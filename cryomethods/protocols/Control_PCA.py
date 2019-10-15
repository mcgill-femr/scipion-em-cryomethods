import os
import re
import copy
import random
import numpy as np
from glob import glob
from collections import Counter

import pyworkflow.em as em
import pyworkflow.em.metadata as md
import pyworkflow.protocol.constants as cons
import pyworkflow.protocol.params as params
from pyworkflow.utils import (makePath, copyFile, replaceBaseExt)

from cryomethods import Plugin
from cryomethods.convert import (writeSetOfParticles, rowToAlignment,
                                 relionToLocation, loadMrc, saveMrc,
                                 alignVolumes, applyTransforms)

from .protocol_base import ProtocolBase
from xmipp3.convert import getImageLocation
from os.path import join
from cryomethods import Plugin
import matplotlib.pyplot as plt
from pyworkflow.object import Float
from cryomethods.convert import writeSetOfParticles
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import *
import sys





class ProtLandscapePCA(ProtocolBase):
    _label = 'Control PCA'
    IS_2D = False
    IS_AUTOCLASSIFY = True

    def _initialize(self):
        """ This function is mean to be called after the
        working dir for the protocol have been set.
        (maybe after recovery from mapper)
        """
        self._createFilenameTemplates()
        self._createIterTemplates()

    def _createFilenameTemplates(self):
        """ Centralize how files are called for iterations and references. """
        self.levDir = self._getExtraPath()
        self.rLevDir = self._getExtraPath('lev_%(lev)02d/rLev_%(rLev)02d/')
        self.rLevIter = self.rLevDir + 'relion_it%(iter)03d_'
        # add to keys, data.star, optimiser.star and sampling.star
        myDict = {
            'input_star': self.levDir + 'input_rLev-%(rLev)03d.star',
            'outputData': self.levDir + 'output_data.star',
            'map': self.levDir + 'map_id-%(id)s.mrc',
            'avgMap': self.levDir+ 'map_average.mrc',
            'modelFinal': self.levDir + 'model.star',
            'relionMap': self.rLevDir + 'relion_it%(iter)03d_class%(ref3d)03d.mrc',
            'outputModel': self.levDir + 'output_model.star',
            'data': self.rLevIter + 'data.star',
            'rawFinalModel': self._getExtraPath('raw_final_model.star'),
            'rawFinalData': self._getExtraPath('raw_final_data.star'),
            'finalModel': self._getExtraPath('final_model.star'),
            'finalData': self._getExtraPath('final_data.star'),
            'finalAvgMap': self._getExtraPath('map_average.mrc'),
            'optimiser': self.rLevIter + 'optimiser.star',
            'all_avgPmax_xmipp': self._getTmpPath(
                'iterations_avgPmax_xmipp.xmd'),
            'all_changes_xmipp': self._getTmpPath(
                'iterations_changes_xmipp.xmd'),
        }
        for key in self.FILE_KEYS:
            myDict[key] = self.rLevIter + '%s.star' % key
            key_xmipp = key + '_xmipp'
            myDict[key_xmipp] = self.rLevDir + '%s.xmd' % key
        # add other keys that depends on prefixes
        for p in self.PREFIXES:
            myDict['%smodel' % p] = self.rLevIter + '%smodel.star' % p
            myDict[
                '%svolume' % p] = self.rLevDir + p + 'class%(ref3d)03d.mrc:mrc'

        self._updateFilenamesDict(myDict)

    def _createIterTemplates(self, rLev=None):
        """ Setup the regex on how to find iterations. """
        rLev = self._rLev if rLev is None else rLev
        print("level rLev in _createIterTemplates:", self._level, rLev)

        self._iterTemplate = self._getFileName('data', lev=self._level,
                                               rLev=rLev,
                                               iter=0).replace('000', '???')
        # Iterations will be identify by _itXXX_ where XXX is the iteration
        # number and is restricted to only 3 digits.
        self._iterRegex = re.compile('_it(\d{3,3})_')
        self._classRegex = re.compile('_class(\d{2,2}).')

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        self._defineInputParams(form)
        self._defineReferenceParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineCTFParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineOptimizationParams(form, expertLev=cons.LEVEL_NORMAL)
        form.addParam('doImageAlignment', params.BooleanParam, default=True,
                      label='Perform image alignment?',
                      help='If set to No, then rather than performing both '
                           'alignment and classification, only classification '
                           'will be performed. This allows the use of very '
                           'focused masks.This requires that the optimal '
                           'orientations of all particles are already '
                           'calculated.')
        form.addParam('sampling', params.FloatParam, default=1.7,
                      label="Sampling rate (A/px)",
                      help='Sampling rate (Angstroms/pixel)')
        self._defineSamplingParams(form, expertLev=cons.LEVEL_NORMAL,
                                   cond='doImageAlignment')
        self._defineAdditionalParams(form)


    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._level = 1
        self._rLev = 1
        self._evalIdsList = []
        self._doneList = []
        self._constStop = []
        self.stopDict = {}
        self._mapsDict = {}
        self._clsIdDict = {}

        # self._initialize()
        self._insertFunctionStep('estimatePCAStep')
    #-------------------------step function-----------------------------------
    def _getRefStar(self):
        return self._getExtraPath("input_references.star")

    def _getAverageVol(self,  volSet, listVol=[]):
        self._createFilenameTemplates()
        Plugin.setEnviron()

        inputObj = self.inputVolumes.get()
        fnIn = []
        for i in inputObj:

            a = getImageLocation(i).split(':')[0]
            fnIn.append(a)
        listVol= fnIn

        # listVol = self._getPathMaps() if not bool(listVol) else listVol
        print (listVol, "listVolll")
        print('creating average map: ', listVol)
        avgVol = self._getFileName('avgMap')
        print (avgVol, "avgVol")

        print('alignining each volume vs. reference')
        for vol in listVol:
            print (vol, "vol2")
            npVol = loadMrc(vol, writable=False)

            if vol == listVol[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)
            npAvgVol += npVol

        print (npAvgVol, "npAvgVol1")
        npAvgVol = np.divide(npAvgVol, len(listVol))
        print('saving average volume')
        saveMrc(npAvgVol.astype(dType), avgVol)





    def estimatePCAStep(self):
        self._createFilenameTemplates()

        Plugin.setEnviron()
        listNpVol = []

        row = md.Row()
        refMd = md.MetaData()

        ih = em.ImageHandler()
        inputObj = self.inputVolumes.get()

        fnIn= []
        for i in inputObj:
            a = getImageLocation(i).split(':')[0]
            fnIn.append(a)

        for vol in fnIn:
            row.setValue(md.RLN_MLMODEL_REF_IMAGE, vol)
            row.addToMd(refMd)
        refMd.write(self._getRefStar())




        print (fnIn, "fninn")

        # for i in fnIn:
        #     print (i, "vol")
        #     row.setValue(md.RLN_MLMODEL_REF_IMAGE, i)
        #     row.addToMd(refMd)
        # refMd.write(self._getRefStar())

        listVol = fnIn
        self._getAverageVol(listVol)

        avgVol = self._getFileName('avgMap')
        npAvgVol = loadMrc(avgVol, False)
        dType = npAvgVol.dtype
        volNp = loadMrc(listVol.__getitem__(0), False)
        dim = volNp.shape[0]
        lenght = dim ** 3
        cov_matrix = []

        for vol in listVol:
            volNp = loadMrc(vol, False)

            # Now, not using diff volume to estimate PCA
            # diffVol = volNp - npAvgVol
            volList = volNp.reshape(lenght)

            row = []
            b = volList - npAvgVol.reshape(lenght)
            print (b, 'b')
            for j in listVol:
                npVol = loadMrc(j, writable=False)
                volList = npVol.reshape(lenght)
                volList_two = volList - npAvgVol.reshape(lenght)
                print (volList, "vollist")
                temp_a= np.corrcoef(volList_two, b).item(1)
                print (temp_a, "temp_a")
                row.append(temp_a)
                # b= volList_two
                # print (corr, "corr")
            cov_matrix.append(row)

        print (cov_matrix, "covMatrix")
        u, s, vh = np.linalg.svd(cov_matrix)
        cuttOffMatrix = sum(s) * 0.95
        sCut = 0

        print('cuttOffMatrix & s: ', cuttOffMatrix, s)
        for i in s:
            print('cuttOffMatrix: ', cuttOffMatrix)
            if cuttOffMatrix > 0:
                print("Pass, i = %s " % i)
                cuttOffMatrix = cuttOffMatrix - i
                sCut += 1
            else:
                break
        print('sCut: ', sCut)

        eigValsFile ='eigenvalues.txt'
        self._createMFile(s, eigValsFile)

        eigVecsFile = 'eigenvectors.txt'
        self._createMFile(vh, eigVecsFile)

        vhDel = np.transpose(np.delete(vh, np.s_[sCut:vh.shape[1]], axis=0))
        self._createMFile(vhDel, 'matrix_vhDel.txt')

        print(' this is the matrix "vhDel": ', vhDel)


        # insert at 1, 0 is the script path (or '' in REPL)

        mat_one = []
        for vol in listVol:
            volNp = loadMrc(vol, False)
            volList = volNp.reshape(lenght)
            print (volList, "volList")
            row_one = []
            for j in listVol:
                npVol = loadMrc(j, writable=False)
                volList_three = npVol.reshape(lenght)
                j_trans = volList_three.transpose()
                matrix_two = np.dot(volList, j_trans)
                row_one.append(matrix_two)
            mat_one.append(row_one)

        matProj = np.dot(mat_one, vhDel)
        # print (newBaseAxis, "newbase")
        # matProj = np.transpose(np.dot(newBaseAxis, mat_one))
        print (matProj, "matProj")

        volSet=[]
        mf = '/home/josuegbl/PROCESSING/MAPS_FINALE/raw_final_model.star'
        modelFile = md.MataData(mf)
        modelStar = md.MetaData('model_classes@' + modelFile)
        for row in md.iterRows(modelStar):
            fn = row.getValue('rlnReferenceImage')

            itemId = self._getClassId(fn)
            classDistrib = row.getValue('rlnClassDistribution')

            if classDistrib > 0:
                vol = em.Volume()
                vol.setObjId(itemId)
                vol._rlnClassDistribution = em.Float(classDistrib)
                volSet.append(vol)

        # mf = self._getPath('/home/josuegbl/PROCESSING/MAPS_FINALE/raw_final_model.star')
        print (mf, "mf")


        x_proj = [item[0] for item in matProj]
        y_proj = [item[1] for item in matProj]
        print (x_proj, "x_proj")
        print (y_proj, "y_proj")

    # ------------------------------------------------------------------

    # for run in range(numOfRuns):
    #     mf = (self._getExtraPath('run_%02d' % run,
    #                              'relion_it%03d_' % iter +
    #                              'model.star'))
    #     print (mf, "mf")
    #     # mdClass.read(mf)
    #     ab = md.MetaData('model_classes@' + mf)
    #
    #     print (ab, "mdClass")
    #
    # clsDist = []
    #
    # for row in md.iterRows(ab):
    #     classDistrib = row.getValue('rlnClassDistribution')
    #     clsDist.append(classDistrib)
    #
    # print (clsDist, "clsDist")
    #
    # imgSet = self.inputParticles.get()
    # totalPart = imgSet.getSize()
    # print (totalPart, "totalPar")
    # K = self.numOfVols.get()
    # colors = cm.rainbow(np.linspace(0, 1, totalPart))
    # colorList = []
    # for i in clsDist:
    #     index = int(len(colors) * i)
    #     colorList.append(colors[index])
    #
    # x_proj = [item[0] for item in matProj]
    # y_proj = [item[1] for item in matProj]
    # print (x_proj, "x_proj")
    # print (y_proj, "y_proj")



    # plt.scatter(x_proj, y_proj, c=colorList, alpha=0.5)
    #
    # plt.show()


    def _getPathMaps(self):
        inputObj = self.inputVolumes.get()
        filesPath = []
        for i in inputObj:
            a = getImageLocation(i)
            filesPath.append(a)

        print (filesPath, "fninn")
        return sorted(glob(filesPath))

    def _createMFile(self, matrix, name='matrix.txt'):
        print (name, "name")
        f = open(name, 'w')        # f = open(name, 'w')
        for list in matrix:
            s = "%s\n" % list
            f.write(s)
        f.close()

    def _getClassId(self, volFile):
        result = None
        s = self._classRegex.search(volFile)
        if s:
            result = int(s.group(1)) # group 1 is 2 digits class number
        return self.volDict[result]

    def _fillVolSetIter(self, volSet):
        volSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._createFilenameTemplates()
        Plugin.setEnviron()
        modelStar = md.MetaData('model_classes@' +
                                self._getFileName('modelFinal'))


        for row in md.iterRows(modelStar):
            fn = row.getValue('rlnReferenceImage')
            fnMrc = fn + ":mrc"
            itemId = self._getClassId(fn)
            classDistrib = row.getValue('rlnClassDistribution')
            accurracyRot = row.getValue('rlnAccuracyRotations')
            accurracyTras = row.getValue('rlnAccuracyTranslations')
            resol = row.getValue('rlnEstimatedResolution')
            if classDistrib > 0:
                vol = em.Volume()
                self._invertScaleVol(fnMrc)
                vol.setFileName(self._getOutputVolFn(fnMrc))
                vol.setObjId(itemId)
                vol._rlnClassDistributionl = em.Float(classDistrib)
                vol._rlnAccuracyRotations = em.Foat(accurracyRot)
                vol._rlnAccuracyTranslations = em.Float(accurracyTras)
                vol._rlnEstimatedResolution = em.Float(resol)
                volSet.append(vol)