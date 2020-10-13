# **************************************************************************
# *
# * Authors:     Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
# *              Javier Vargas Balbuena (javier.vargasbalbuena@mcgill.ca)
# *
# * Department of Anatomy and Cell Biology, McGill University
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
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
from collections import defaultdict

from pwem.emlib.image import ImageHandler

try:
    from itertools import izip
except ImportError:
    izip = zip
import numpy as np
from glob import glob

import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons
from pyworkflow.utils import makePath

from pwem.protocols import EMProtocol

from cryomethods import Plugin
from cryomethods.constants import METHOD

from cryomethods.functions import NumpyImgHandler


class ProtVolClustering(EMProtocol):
    _label = 'clustering volumes'
    def __init__(self, **args):
        EMProtocol.__init__(self, **args)

    def _initialize(self):
        """ This function is mean to be called after the
        working dir for the protocol have been set.
        (maybe after recovery from mapper)
        """
        pass

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes',
                      important=True,
                      label='Input volumes',
                      help='Initial reference 3D maps')

        group = form.addGroup('Preprocessing')
        group.addParam('lowPass', params.FloatParam, default=25,
                       label='low-pass filter (A)',
                       help='Low pass filter value in Angstroms.')
        group.addParam('lowRaised', params.FloatParam, default=0.01,
                       expertLevel=cons.LEVEL_ADVANCED,
                       label='Transition bandwidth',
                       help="This value represent how long you extend the low "
                            "pass filter to avoid unwanted effects in the maps.")
        group.addParam('alignVolumes', params.BooleanParam,
                      default=False,
                      label="Do you want align the volumes?",
                      help='If set Yes, volumes will be aligned against the '
                           'average volume.')
        form.addParam('classMethod', params.EnumParam, default=1,
                      choices=METHOD,
                      label='Method to determine the classes:',
                      help='')

    # -------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep', self._getResetDeps())
        self._insertFunctionStep('createOutputStep', self.classMethod.get())

    # -------------------------- STEPS functions -------------------------------
    def convertInputStep(self, resetDeps):
        """ Preprocess all volumes prior to clustering"""
        Plugin.setEnviron()
        self._convertVolumes()
        pxSize = self.inputVolumes.get().getSamplingRate()
        listVol = self._getPathMaps()
        lowPass = self.lowPass.get()
        lowRaised = self.lowRaised.get()

        for volFn in listVol:
            inputMap = volFn + ':mrc'
            outMap = volFn.split('.')[0] + '_filtered.mrc:mrc'

            dictParam = {'-i': inputMap,
                         '-o': outMap,
                         '--sampling': pxSize
                         }
            args = ' '.join('%s %s' %(k,v) for k,v in dictParam.items())
            args += ' --fourier low_pass %s %s' %(lowPass, lowRaised)
            self.runJob('xmipp_transform_filter', args)

        print("Saving average volume")
        self._saveAverageVol()
        if self.alignVolumes:
            self._alignVolumes()
            
    def createOutputStep(self, method):
        """do the clustering and generates different classes"""
        dictNames = defaultdict(list)
        inputVol = self.inputVolumes.get()
        volList = self._getPathMaps('volume_????_filtered.mrc')

        npMask, _ = self._doAvgMapsMask(volList)
        matrix, _ = self._mrcToNp(volList, npMask)
        matrix, _ = self._doPCA(volList)
        labels = self._clusteringData(matrix)

        for vol, label in izip(inputVol, labels):
            dictNames[label].append(vol)

        for key, volumeList in dictNames.items():
            subset = '%03d' %key
            volSet = self._createSetOfVolumes(subset)
            for vol in volumeList:
                volSet.append(vol)

            result = {'outputVolumes' + subset: volSet}
            self._defineOutputs(**result)
            self._defineSourceRelation(inputVol, volSet)

    # -------------------------- UTILS functions -------------------------------
    def _getResetDeps(self):
        volId = self.inputVolumes.get().getObjId()

        dictParams = {'lowPass': self.lowPass.get(),
                      'lowRaised': self.lowRaised.get(),
                      'alignVolumes': self.alignVolumes}
        deps = ' '.join(['%s' % str(v) for k, v in dictParams.items()])
        deps += ' %s' % volId
        return deps

    def _getAvgMapFn(self):
        return self._getExtraPath('map_average.mrc')

    def _getPathMaps(self, filename="volume_????.mrc"):
        filesPath = self._getInputPath(filename)
        return sorted(glob(filesPath))

    def _getInputPath(self, *paths):
        """ Return a path inside the tmp folder. """
        return self._getExtraPath("input", *paths)

    def _getOutputFn(self, num):
        """ Return a new name if the inputFn is not .mrc """
        return self._getInputPath('volume_%04d.mrc' % num)

    def _convertVolumes(self):
        ih = ImageHandler()
        makePath(self._getExtraPath('input'))
        inputVols = self.inputVolumes.get()
        for vol in inputVols:
            map = ih.read(vol)
            outputFn = self._getOutputFn(vol.getObjId())
            map.write(outputFn)

    def _saveAverageVol(self):
        listVol = self._getPathMaps('volume_????_filtered.mrc')
        avgVol = self._getAvgMapFn()
        npAvgVol, dType = self._doAverageMap(listVol)
        print("Dtype: ", dType)
        if dType == 'float64':
            dType = 'float32'
        NumpyImgHandler.saveMrc(npAvgVol.astype(dType), avgVol)

    def _doAvgMapsMask(self, listVol):
        for vol in listVol:
            npVol = self._getVolNp(vol)
            npMask = 1 * (npVol > 0)

            if vol == listVol[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)

            npAvgVol += npMask

        npAvgVol = npAvgVol / len(listVol)
        npAvgMask = 1 * (npAvgVol < 0.99)
        # npAvgVol *= npAvgMask
        return npAvgMask, dType

    def _doAverageMap(self, listVol):
        for vol in listVol:
            npVol = self._getVolNp(vol)
            if vol == listVol[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)
            npAvgVol += npVol

        npAvgVol = int(npAvgVol / len(listVol))
        return npAvgVol, dType

    def _getVolNp(self, vol):
        volNp = NumpyImgHandler.loadMrc(vol, False)
        std = 2.5 * volNp.std()
        npMask = 1 * (volNp >= std)
        mapNp = volNp * npMask
        return mapNp

    def _alignVolumes(self):
        # Align all volumes
        Plugin.setEnviron()
        listVol = self._getPathMaps('volume_????_filtered.mrc')
        avgVol = self._getAvgMapFn()
        npAvgVol = NumpyImgHandler.loadMrc(avgVol, writable=False)
        dType = npAvgVol.dtype

        for vol in listVol:
            npVolAlign = self._getVolNp(vol)
            npVolFlipAlign = np.fliplr(npVolAlign)

            axis, shifts, angles, score = NumpyImgHandler.alignVolumes(npVolAlign, npAvgVol)
            axisf, shiftsf, anglesf, scoref = NumpyImgHandler.alignVolumes(npVolFlipAlign,
                                                           npAvgVol)
            if scoref > score:
                npVol = NumpyImgHandler.applyTransforms(npVolFlipAlign, shiftsf, anglesf, axisf)
            else:
                npVol = NumpyImgHandler.applyTransforms(npVolAlign, shifts, angles, axis)

            NumpyImgHandler.saveMrc(npVol.astype(dType), vol)

    def _mrcToNp(self, volList, avgVol=None):
        listNpVol = []
        for vol in volList:
            volNp = self._getVolNp(vol)
            dim = volNp.shape[0]
            lenght = dim**3
            if avgVol is None:
                volNpList = volNp.reshape(lenght)
            else:
                volNpSub = volNp * avgVol
                npMask = 1 * (volNpSub > 0)
                volNpSub *= npMask
                volNpList = volNpSub.reshape(lenght)

            listNpVol.append(volNpList)
        return listNpVol, listNpVol[0].dtype

    def _doPCA(self, listVol):
        npAvgVol, _ = self._doAverageMap(listVol)
        listNpVol, _ = self._mrcToNp(listVol, avgVol=npAvgVol)

        covMatrix = np.cov(listNpVol)
        u, s, vh = np.linalg.svd(covMatrix)
        cuttOffMatrix = sum(s) * 0.95
        sCut = 0

        for i in s:
            if cuttOffMatrix > 0:
                cuttOffMatrix = cuttOffMatrix - i
                sCut += 1
            else:
                break

        vhDel = np.transpose(np.delete(vh, np.s_[sCut:vh.shape[1]], axis=0))
        newBaseAxis = vhDel.T.dot(listNpVol)
        matProj = np.transpose(np.dot(newBaseAxis, np.transpose(listNpVol)))

        projFile = self._getPath('projection_matrix.txt')
        self._createMFile(matProj, projFile)
        return matProj, newBaseAxis

    def _clusteringData(self, matProj):
        method = self.classMethod.get()
        if method == 0:
            return self._doSklearnKmeans(matProj)
        else:
            return self._doSklearnAffProp(matProj)

    def _doSklearnKmeans(self, matProj):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=matProj.shape[1]).fit(matProj)
        return kmeans.labels_

    def _doSklearnAffProp(self, matProj):
        from sklearn.cluster import AffinityPropagation
        ap = AffinityPropagation(damping=0.5).fit(matProj)
        return ap.labels_

    def _createMFile(self, matrix, name='matrix.txt'):
        f = open(name, 'w')
        for list in matrix:
            s = "%s\n" % list
            f.write(s)
        f.close()
