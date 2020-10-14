import matplotlib


from glob import glob

import numpy as np
from itertools import *
from matplotlib import *

import pyworkflow.em as em
import pyworkflow.em.metadata as md
import pyworkflow.protocol.params as params
from cryomethods import Plugin
from cryomethods.convert import (loadMrc, saveMrc, alignVolumes, applyTransforms)
# from xmipp3.convert import getImageLocation
from .protocol_base import ProtocolBase
import collections


PCA_THRESHOLD = 0
PCA_COUNT=1
ALIGN = 0
NOTALIGN = 1



class ProtLandscapePCA(em.EMProtocol):
    _label = 'Control PCA'
    FILE_KEYS = ['data', 'optimiser', 'sampling']
    PREFIXES = ['']

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
            'all_avgPmax_xmipp': self._getTmpPath('iterations_avgPmax_xmipp.xmd'),
            'all_changes_xmipp': self._getTmpPath(
                'iterations_changes_xmipp.xmd')
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

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes',
                      important=True,
                      label='Input volumes',
                      help='Initial reference 3D maps')
        form.addParam('resLimit', params.FloatParam, default=20,
                      label="Resolution Limit (A)",
                      help="Resolution limit used to low pass filter both "
                           "input and reference map(s).")
        form.addParam('alignment', params.EnumParam, default=0,
                      choices=['Yes, align Volumes',
                               'No volumes alignment'],
                      label='Align Input Volumes?')

        group = form.addSection('machine leaning components')
        group.addParam('thresholdMode', params.EnumParam,
                      choices=['thr', 'pcaCount'],
                      default=PCA_THRESHOLD,
                      label='Cut-off mode',
                      help='Threshold value will allow you to select the\n'
                           'principle components above this value.\n'
                           'sCut will allow you to select number of\n'
                           'principle components you want to select.')
        group.addParam('thr', params.FloatParam, default=0.95,
                      important=True,
                      condition='thresholdMode==%d' % PCA_THRESHOLD,
                      label='THreshold percentage')
        group.addParam('pcaCount', params.FloatParam, default=2,
                      label="count of PCA",
                      condition='thresholdMode==%d' % PCA_COUNT,
                      help='Number of PCA you want to select.')

        # group.addParam('addWeights', params.FileParam, label="Weight File path",
        #               allowsNull=True,
        #               help='Specify a path to weights for volumes.')
        #

        form.addParallelSection(threads=0, mpi=0)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        inputVols = self.inputVolumes.get()
        volId = inputVols.getObjId()
        self._insertFunctionStep('convertInputStep', volId)
        self._insertFunctionStep('analyzePCAStep')

    #-------------------------step function-------------------------------------
    def convertInputStep(self, resetId):
        inputVols = self.inputVolumes.get()
        ih = em.ImageHandler()
        for i, vol in enumerate(inputVols):
            num = vol.getObjId()
            newFn = self._getExtraPath('volume_id_%03d.mrc' % num)
            ih.convert(vol, newFn)

        sampling_rate = inputVols.getSamplingRate()
        print (sampling_rate, "sampling rate before")
        resLimitDig = self.resLimit.get()
        print (resLimitDig, "resLimitDig")
        inputVols.setSamplingRate(resLimitDig)
        print (inputVols.getSamplingRate(), "after")

    #     ----------------alignment---------------------------
    def alignVols(self):
        self._getAverageVol()
        avgVol= self._getFileName('avgMap')
        npAvgVol = loadMrc(avgVol, False)
        dType = npAvgVol.dtype
        fnIn = self._getMrcVolumes()
        for vols in fnIn:
            npVolAlign = f(vols, False)
            npVolFlipAlign = np.fliplr(npVolAlign)

            axis, shifts, angles, score = alignVolumes(npVolAlign, npAvgVol)
            axisf, shiftsf, anglesf, scoref = alignVolumes(npVolFlipAlign,
                                                           npAvgVol)
            if scoref > score:
                npVol = applyTransforms(npVolFlipAlign, shiftsf, anglesf, axisf)
            else:
                npVol = applyTransforms(npVolAlign, shifts, angles, axis)

            saveMrc(npVol.astype(dType), vols)


    def analyzePCAStep(self):
        self._createFilenameTemplates()
        Plugin.setEnviron()
        if self.alignment.get()==0:
            self.alignVols()
            fnIn= self._getMrcVolumes()
        else:
            fnIn = self._getMrcVolumes()

        self._getAverageVol()

        avgVol = self._getFileName('avgMap')
        npAvgVol = loadMrc(avgVol, False)
        dType = npAvgVol.dtype
        iniVolNp = loadMrc(fnIn[0], False)
        dim = iniVolNp.shape[0]
        lenght = dim ** 3

        u, s, vh = np.linalg.svd(self._getCovMatrix())
        vhDel = self._getvhDel(vh, s)
        # -------------NEWBASE_AXIS-------------------------------------------
        counter = 0

        for eignRow in vhDel.T:
            base = np.zeros(lenght)
            for (vol, eigenCoef) in izip(fnIn,eignRow):
                volInp = loadMrc(vol, False)
                volInpR = volInp.reshape(lenght)
                volSubs= volInpR - npAvgVol.reshape(lenght)
                base += volSubs*eigenCoef
                volBase = base.reshape((dim, dim, dim))
            nameVol = 'volume_base_%02d.mrc' % (counter)
            print('-------------saving map %s-----------------' % nameVol)
            saveMrc(volBase.astype(dType),self._getExtraPath(nameVol))
            counter += 1

        matProj = []
        baseMrc = self._getExtraPath("volume_base_??.mrc")
        baseMrcFile = sorted(glob(baseMrc))
        for vol in fnIn:
            volNp = loadMrc(vol, False)
            restNpVol = volNp.reshape(lenght) - npAvgVol.reshape(lenght)
            volRow = restNpVol.reshape(lenght)
            rowCoef = []
            for baseVol in baseMrcFile:
                npVol = loadMrc(baseVol, writable=False)
                baseVol_row= npVol.reshape(lenght)
                baseVol_col = baseVol_row.transpose()
                projCoef = np.dot(volRow, baseVol_col)
                rowCoef.append(projCoef)
            matProj.append(rowCoef)

        # obtaining volumes from coordinates-----------------------------------
        os.makedirs(self._getExtraPath('reconstructed_vols'))
        orignCount=0
        for projRow in matProj:
            vol = np.zeros((dim, dim,dim))
            for baseVol, proj in zip(baseMrcFile, projRow):
                volNpo = loadMrc(baseVol, False)
                vol += volNpo * proj
            finalVol= vol + npAvgVol
            nameVol = 'volume_reconstructed_%02d.mrc' % (orignCount)
            print('-------------saving original_vols %s-----------------' % nameVol)
            saveMrc(finalVol.astype(dType), self._getExtraPath('reconstructed_vols', nameVol))
            orignCount += 1

        # difference b/w input vol and original vol-----------------------------
        reconstMrc = self._getExtraPath("original_vols","*.mrc")
        reconstMrcFile = sorted(glob(reconstMrc))
        diffCount=0
        os.makedirs(self._getExtraPath('volDiff'))
        for a, b in zip(reconstMrcFile, fnIn):
            volRec = loadMrc(a, False)
            volInpThree = loadMrc(b, False)
            volDiff= volRec - volInpThree
            nameVol = 'volDiff_%02d.mrc' % (diffCount)
            print('-------------saving original_vols %s-----------------' % nameVol)
            saveMrc(volDiff.astype(dType), self._getExtraPath('volDiff', nameVol))
            diffCount += 1

        #save coordinates:
        os.makedirs(self._getExtraPath('Coordinates'))
        coorPath = self._getExtraPath('Coordinates')

        mat_file = os.path.join(coorPath, 'matProj_splic')
        coordNumpy= np.save(mat_file, matProj)

    # -------------------------- UTILS functions ------------------------------
    def _getMrcVolumes(self):
        return sorted(glob(self._getExtraPath('volume_id_*.mrc')))

    def _getAverageVol(self):
        self._createFilenameTemplates()
        Plugin.setEnviron()

        listVol = self._getMrcVolumes()
        avgVol = self._getFileName('avgMap')
        npVol = loadMrc(listVol[0], writable=False)
        dType = npVol.dtype
        npAvgVol = np.zeros(npVol.shape)

        for vol in listVol:
            npVol = loadMrc(vol, writable=False)
            npAvgVol += npVol

        npAvgVol = np.divide(npAvgVol, len(listVol))
        saveMrc(npAvgVol.astype(dType), avgVol)

    def _getCovMatrix(self):
        if self.alignment.get()==0:
            self.alignVols()
            fnIn= self._getMrcVolumes()
        else:
            fnIn = self._getMrcVolumes()

        self._getAverageVol()

        avgVol = self._getFileName('avgMap')
        npAvgVol = loadMrc(avgVol, False)
        dType = npAvgVol.dtype
        iniVolNp = loadMrc(fnIn[0], False)
        dim = iniVolNp.shape[0]
        lenght = dim ** 3
        cov_matrix = []
        for vol in fnIn:
            volNp = loadMrc(vol, False)
            volList = volNp.reshape(lenght)

            row = []
            # Now, using diff volume to estimate PCA
            b = volList - npAvgVol.reshape(lenght)
            for j in fnIn:
                npVol = loadMrc(j, writable=False)
                volList_a = npVol.reshape(lenght)
                volList_two = volList_a - npAvgVol.reshape(lenght)
                temp_a= np.corrcoef(volList_two, b).item(1)
                row.append(temp_a)
            cov_matrix.append(row)
        os.makedirs(self._getExtraPath('CovMatrix'))
        covPath = self._getExtraPath('CovMatrix')
        CovMatrix = os.path.join(covPath, 'covMatrix')
        np.save(CovMatrix, cov_matrix)
        CovMatData = np.load(
            self._getExtraPath('CovMatrix', 'covMatrix.npy'))
        return CovMatData

    # def getParticlesPca(self):
    #     z_part= np.loadtxt(self.addWeights.get())
    #     return z_part

    def _getPathMaps(self):
        inputObj = self.inputVolumes.get()
        filesPath = []
        for i in inputObj:
            a = getImageLocation(i)
            filesPath.append(a)

        return sorted(glob(filesPath))

    def _createMFile(self, matrix, name='matrix.txt'):
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


    def _getvhDel(self, vh, s):
        if self.thresholdMode == PCA_THRESHOLD:
            thr= self.thr.get()
            if thr < 1:
                cuttOffMatrix = sum(s) * thr
                sCut = 0

                for i in s:
                    if cuttOffMatrix > 0:
                        cuttOffMatrix = cuttOffMatrix - i
                        sCut += 1
                    else:
                        break

                vhDel = self._geteigen(vh, sCut, s)
                return vhDel
            else:
                os.makedirs(self._getExtraPath('EigenFile'))
                eigPath = self._getExtraPath('EigenFile')
                eigValsFile = os.path.join(eigPath, 'eigenvalues')
                np.save(eigValsFile, s)
                eignValData = np.load(
                    self._getExtraPath('EigenFile', 'eigenvalues.npy'))

                eigVecsFile = os.path.join(eigPath, 'eigenvectors')
                np.save(eigVecsFile, vh)
                eignVecData = np.load(
                    self._getExtraPath('EigenFile', 'eigenvectors.npy'))
                vhdelPath = os.path.join(eigPath, 'matrix_vhDel')
                np.save(vhdelPath, vh.T)
                vhDelData = np.load(
                    self._getExtraPath('EigenFile', 'matrix_vhDel.npy'))
                return vh.T
        else:

            sCut= int(self.pcaCount.get())
            vhDel = self._geteigen(vh, sCut, s)
            return vhDel

    def _geteigen(self, vh, sCut, s):
        os.makedirs(self._getExtraPath('EigenFile'))
        eigPath = self._getExtraPath('EigenFile')
        eigValsFile = os.path.join(eigPath, 'eigenvalues')
        np.save(eigValsFile, s)
        eignValData = np.load(
            self._getExtraPath('EigenFile', 'eigenvalues.npy'))

        eigVecsFile = os.path.join(eigPath, 'eigenvectors')
        np.save(eigVecsFile, vh)
        eignVecData = np.load(
            self._getExtraPath('EigenFile', 'eigenvectors.npy'))

        vhDel = np.transpose(np.delete(vh, np.s_[sCut:vh.shape[1]], axis=0))
        vhdelPath = os.path.join(eigPath, 'matrix_vhDel')
        np.save(vhdelPath, vhDel)
        vhDelData = np.load(self._getExtraPath('EigenFile', 'matrix_vhDel.npy'))

        return vhDel

    def _getPcaCount(self, s):
        cuttOffMatrix = sum(s) * 0.95
        sCut = 0

        for i in s:
            if cuttOffMatrix > 0:
                cuttOffMatrix = cuttOffMatrix - i
                sCut += 1
            else:
                break
        return sCut




    def _validate(self):
        errors = []
        return errors
