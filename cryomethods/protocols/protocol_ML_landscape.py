# **************************************************************************
# *
# * Authors:  Satinder Kaur (satinder.kaur@mail.mcgill.ca), May 2019
# *
# *
# *
# * Department of Anatomy and Cell Biology, McGill University, Montreal
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'satinder.kaur@mail.mcgill.ca'
# *
# **************************************************************************
from glob import glob
try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np
from matplotlib import *

import pyworkflow.protocol.params as params

from pwem.emlib.image import ImageHandler
from pwem.protocols import EMProtocol

from cryomethods import Plugin
from cryomethods.functions import NumpyImgHandler as npih

PCA_THRESHOLD = 0
PCA_COUNT=1
ALIGN = 0
NOTALIGN = 1


class ProtLandscapePCA(EMProtocol):
    _label = 'Energy Landscape'
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
            'map': self.levDir + 'map_id-%(id)s.mrc',
            'avgMap': self.levDir+ 'map_average.mrc',
            'data': self.rLevIter + 'data.star',
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
        form.addParam('inputClasses', params.PointerParam,
                      pointerClass='SetOfClasses3D',
                      important=True,
                      label='Input classes',
                      help='Please, pick a set of classes 3D.')
        form.addParam('resLimit', params.FloatParam, default=20,
                      label="Resolution Limit (A)",
                      help="Resolution limit used to low pass filter both "
                           "input and reference map(s).")
        form.addParam('alignment', params.EnumParam, default=1,
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
        form.addParallelSection(threads=0, mpi=0)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        inputCls = self.inputClasses.get()
        self._insertFunctionStep('convertInputStep', inputCls.getObjId())
        self._insertFunctionStep('analyzePCAStep')

    #-------------------------step function-------------------------------------
    def convertInputStep(self, resetId):
        """ Preprocess all volumes prior to clustering"""
        inputCls = self.inputClasses.get()
        ih = ImageHandler()
        for i, class3D in enumerate(inputCls):
            num = class3D.getObjId()
            vol = class3D.getRepresentative()
            newFn = self._getExtraPath('volume_id_%03d.mrc' % num)
            ih.convert(vol, newFn)

        # sampling_rate = inputCls.getSamplingRate()
        # print(sampling_rate, "sampling rate before")
        # resLimitDig = self.resLimit.get()
        # print(resLimitDig, "resLimitDig")
        # inputCls.setSamplingRate(resLimitDig)
        # print(inputCls.getSamplingRate(), "after")

    def analyzePCAStep(self):
        self._createFilenameTemplates()
        Plugin.setEnviron()
        if self.alignment.get()==0:
            self.alignVols()
        fnIn = self._getMrcVolumes()
        self._getAverageVol()

        avgVol = self._getFileName('avgMap')
        npAvgVol = npih.loadMrc(avgVol, False)
        dType = npAvgVol.dtype
        iniVolNp = npih.loadMrc(fnIn[0], False)
        dim = iniVolNp.shape[0]
        lenght = dim ** 3

        u, s, vh = np.linalg.svd(self._getCovMatrix())
        vhDel = self._getvhDel(vh, s)
        # -------------NEWBASE_AXIS-------------------------------------------
        counter = 0

        for eignRow in vhDel.T:
            base = np.zeros(lenght)
            for (vol, eigenCoef) in izip(fnIn,eignRow):
                volInp = npih.loadMrc(vol, False)
                volInpR = volInp.reshape(lenght)
                volSubs= volInpR - npAvgVol.reshape(lenght)
                base += volSubs*eigenCoef
                volBase = base.reshape((dim, dim, dim))
            nameVol = 'volume_base_%02d.mrc' % (counter)
            print('-------------saving map %s-----------------' % nameVol)
            npih.saveMrc(volBase.astype(dType),self._getExtraPath(nameVol))
            counter += 1

        matProj = []
        baseMrc = self._getExtraPath("volume_base_??.mrc")
        baseMrcFile = sorted(glob(baseMrc))
        for vol in fnIn:
            volNp = npih.loadMrc(vol, False)
            restNpVol = volNp.reshape(lenght) - npAvgVol.reshape(lenght)
            volRow = restNpVol.reshape(lenght)
            rowCoef = []
            for baseVol in baseMrcFile:
                npVol = npih.loadMrc(baseVol, writable=False)
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
                volNpo = npih.loadMrc(baseVol, False)
                vol += volNpo * proj
            finalVol= vol + npAvgVol
            nameVol = 'volume_reconstructed_%02d.mrc' % (orignCount)
            print('----------saving original_vols %s-------------' % nameVol)
            npih.saveMrc(finalVol.astype(dType),
                         self._getExtraPath('reconstructed_vols', nameVol))
            orignCount += 1

        # difference b/w input vol and original vol-----------------------------
        reconstMrc = self._getExtraPath("original_vols","*.mrc")
        reconstMrcFile = sorted(glob(reconstMrc))
        diffCount=0
        os.makedirs(self._getExtraPath('volDiff'))
        for a, b in zip(reconstMrcFile, fnIn):
            volRec = npih.loadMrc(a, False)
            volInpThree = npih.loadMrc(b, False)
            volDiff= volRec - volInpThree
            nameVol = 'volDiff_%02d.mrc' % (diffCount)
            print('---------saving original_vols %s--------------' % nameVol)
            npih.saveMrc(volDiff.astype(dType),
                         self._getExtraPath('volDiff', nameVol))
            diffCount += 1

        #save coordinates:
        os.makedirs(self._getExtraPath('Coordinates'))
        coorPath = self._getExtraPath('Coordinates')

        mat_file = os.path.join(coorPath, 'matProj_splic')
        np.save(mat_file, matProj)

    # -------------------------- UTILS functions ------------------------------
    def alignVols(self):
        self._getAverageVol()
        avgVol= self._getFileName('avgMap')
        npAvgVol = npih.loadMrc(avgVol, False)
        dType = npAvgVol.dtype
        fnIn = self._getMrcVolumes()
        for vols in fnIn:
            npVolAlign = npih.loadMrc(vols, False)
            npVolFlipAlign = np.fliplr(npVolAlign)

            axis, shifts, angles, score = npih.alignVolumes(npVolAlign, npAvgVol)
            axisf, shiftsf, anglesf, scoref = npih.alignVolumes(npVolFlipAlign,
                                                           npAvgVol)
            if scoref > score:
                npVol = npih.applyTransforms(npVolFlipAlign, shiftsf,
                                             anglesf, axisf)
            else:
                npVol = npih.applyTransforms(npVolAlign, shifts,
                                             angles, axis)
            npih.saveMrc(npVol.astype(dType), vols)

    def _getMrcVolumes(self):
        return sorted(glob(self._getExtraPath('volume_id_*.mrc')))

    def _getAverageVol(self):
        self._createFilenameTemplates()
        Plugin.setEnviron()

        listVol = self._getMrcVolumes()
        avgVol = self._getFileName('avgMap')
        npVol = npih.loadMrc(listVol[0], writable=False)
        dType = npVol.dtype
        npAvgVol = np.zeros(npVol.shape)

        for vol in listVol:
            npVol = npih.loadMrc(vol, writable=False)
            npAvgVol += npVol

        npAvgVol = np.divide(npAvgVol, len(listVol))
        npih.saveMrc(npAvgVol.astype(dType), avgVol)

    def _getCovMatrix(self):
        if self.alignment.get()==0:
            self.alignVols()
            fnIn= self._getMrcVolumes()
        else:
            fnIn = self._getMrcVolumes()

        self._getAverageVol()

        avgVol = self._getFileName('avgMap')
        npAvgVol = npih.loadMrc(avgVol, False)
        dType = npAvgVol.dtype
        iniVolNp = npih.loadMrc(fnIn[0], False)
        dim = iniVolNp.shape[0]
        lenght = dim ** 3
        cov_matrix = []
        for vol in fnIn:
            volNp = npih.loadMrc(vol, False)
            volList = volNp.reshape(lenght)

            row = []
            # Now, using diff volume to estimate PCA
            b = volList - npAvgVol.reshape(lenght)
            for j in fnIn:
                npVol = npih.loadMrc(j, writable=False)
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
            a = npih.getImageLocation(i)
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
