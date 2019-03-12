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
import sys
from glob import glob
import numpy as np

from pyworkflow.utils import basename
from pyworkflow.tests import *
from cryomethods import Plugin
from cryomethods.protocols import Prot3DAutoClassifier
from cryomethods.convert import loadMrc, alignVolumes, saveMrc, applyTransforms


class TestBase(BaseTest):
    @classmethod
    def setData(cls, dataProject='relion_tutorial'):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.volumes = cls.dataset.getFile('import/case2/*class00?.mrc')
        cls.clsVols = cls.dataset.getFile('classVols/map_rLev-0??.mrc')


class TestAlignVolumes(TestBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestBase.setData()

    def testAlignVolumes(self):
        Plugin.setEnviron()
        volList = sorted(glob(self.volumes))
        volRef = volList.pop(0)
        maxScore = 0

        for vol in volList:
            volRefNp = loadMrc(volRef)
            volNp = loadMrc(vol)
            volNpFp = np.fliplr(volNp)

            axis, shifts, angles, score = alignVolumes(volNp, volRefNp)
            axisf, shiftsf, anglesf, scoref = alignVolumes(volNpFp, volRefNp)
            print('scores : w/o flip- %03f w flip %03f' %(score, scoref))
            if scoref > score:
                print('angles:', anglesf[0], anglesf[1], anglesf[2],)
                print('shifts:', shiftsf[0], shiftsf[1], shiftsf[2],)
                npVol = applyTransforms(volNpFp, shiftsf, anglesf, axisf)
                print('flipped map is better: ', vol)
            else:
                print('angles:', angles[0], angles[1], angles[2],)
                print('shifts:', shifts[0], shifts[1], shifts[2],)
                npVol = applyTransforms(volNp, shifts, angles, axis)
                print('original map is better ', vol)

            saveMrc(npVol, '/home/josuegbl/'+basename(vol))

    def testPCA(self):
        Plugin.setEnviron()
        volList = sorted(glob(self.volumes))
        mList = []
        for vol in volList:
            volNp = loadMrc(vol)
            lenght = volNp.shape[0]**3,
            volList = volNp.reshape(lenght)
            mList.append(volList)

        covMatrix = np.cov(mList)
        # print('Covariance : ', covMatrix)

        eigValues, eigVectors = np.linalg.eig(covMatrix)

        # Make a list of (eigenvalue, eigenvector) tuples
        eigPairs = [(np.abs(eigValues[i]), eigVectors[:,i]) for i in range(len(eigValues))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eigPairs.sort(key=lambda x: x[0], reverse=True)

        matrix_w = np.hstack((eigPairs[0][1].reshape(3,1), eigPairs[1][1].reshape(3,1)))
        # print('Matrix W:\n', matrix_w)

        transformed = matrix_w.T.dot(mList)

        matProj = np.transpose(np.dot(transformed, np.transpose(mList)))

        matDist = []
        for list1 in matProj:
            rows = []
            for list2 in matProj:
                v = 0
                for i,j in izip(list1, list2):
                    v += (i - j)**2
                rows.append(v)
            matDist.append(rows)
        print(matDist)

    def testClustering(self):
        from itertools import izip
        Plugin.setEnviron()
        volList = sorted(glob('/home/josuegbl/PROCESSING/CAJAL/44S_TestBank/Runs'
                              '/002624_ProtAutoClassifier/extra/lev_03/'
                              'map_id-03.0??.mrc'))
        self._getAverageVol(volList)

        # for i in range(4):
        dictNames = {}
        groupDict = {}
        prot = Prot3DAutoClassifier(classMethod=3)
        print("Mehod: ", prot.classMethod.get())
        # matrix = self._estimatePCA(volList)
        matrix, _ = self._mrcToNp(volList)
        labels = prot._clusteringData(matrix)
        if labels is not None:
            f = open('method_%s.txt' % 3, 'w')
            for vol, label in izip (volList, labels):
                dictNames[vol] = label

            for key, value in sorted(dictNames.iteritems()):
                groupDict.setdefault(value, []).append(key)

            for key, value in groupDict.iteritems():
                line = '%s %s\n' % (key, value)
                f.write(line)
            f.close()

            print(labels)

    def testAffinityProp(self):
        from itertools import izip
        Plugin.setEnviron()
        volList = sorted(glob('/home/josuegbl/PROCESSING/CAJAL/44S_TestBank/Runs'
                              '/002624_ProtAutoClassifier/extra/lev_03/'
                              'map_id-03.0??.mrc'))
        self._getAverageVol(volList)

        # for i in range(4):
        dictNames = {}
        groupDict = {}
        prot = Prot3DAutoClassifier(classMethod=3)
        print("Mehod: ", prot.classMethod.get())
        matrix, _ = self._mrcToNp(volList)
        labels = prot._clusteringData(matrix)
        if labels is not None:
            f = open('method_%s.txt' % 3, 'w')
            for vol, label in izip (volList, labels):
                dictNames[vol] = label

            for key, value in sorted(dictNames.iteritems()):
                groupDict.setdefault(value, []).append(key)

            for key, value in groupDict.iteritems():
                line = '%s %s\n' % (key, value)
                f.write(line)
            f.close()

            print(labels)

    def _getAverageVol(self, volList):
        print('creating average map')
        avgVol = ('average_map.mrc')
        print('alignining each volume vs. reference')
        for vol in volList:
            npVol = loadMrc(vol, False)
            if vol == volList[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)
            npAvgVol += npVol

        npAvgVol = np.divide(npAvgVol, len(volList))
        print('saving average volume')
        saveMrc(npAvgVol.astype(dType), avgVol)

    def _estimatePCA(self, volList):
        avgVol = ('average_map.mrc')
        npAvgVol = loadMrc(avgVol, False)
        listDiffVol = []
        m = []
        dType = npAvgVol.dtype
        dim = npAvgVol.shape[0]

        sortedList = sorted(volList)
        listNpVol, _ = self._mrcToNp(sortedList)
        for volNp in listNpVol:
            diffVol = volNp - npAvgVol.reshape(dim**3)
            listDiffVol.append(diffVol)

        covMatrix = np.cov(listDiffVol)
        u, s, vh = np.linalg.svd(covMatrix)
        cuttOffMatrix = sum(s) * 0.95
        sCut = 0

        for i in s:
            if cuttOffMatrix > 0:
                cuttOffMatrix = cuttOffMatrix - i
                sCut += 1
            else:
                break
        print('sCut: ', sCut)

        eigValsFile = 'eigenvalues.txt'
        self._createMFile(s, eigValsFile)

        eigVecsFile = 'eigenvectors.txt'
        self._createMFile(vh, eigVecsFile)

        vhDel = np.transpose(np.delete(vh, np.s_[sCut:vh.shape[1]], axis=0))
        self._createMFile(vhDel, 'matrix_vhDel.txt')

        # print(' this is the matrix "vhDel": ', vhDel)

        newBaseAxis = vhDel.T.dot(listNpVol)

        for i, volNewBaseList in enumerate(newBaseAxis):
            volBase = volNewBaseList.reshape((dim, dim, dim))
            nameVol = 'volume_base_%02d.mrc' % (i+1)
            saveMrc(volBase.astype(dType), nameVol)
        matProj = np.transpose(np.dot(newBaseAxis, np.transpose(listNpVol)))

        projFile = 'projection_matrix.txt'
        self._createMFile(matProj, projFile)
        return matProj

    def _reconstructMap(self, matProj):
        from glob import glob
        listBaseVol = glob('volume_base*.mrc')
        sortedList = sorted(listBaseVol)
        listNpBase, dType = self._mrcToNp(sortedList)

        volNpList = np.dot(matProj, listNpBase)
        dim = int(round(volNpList.shape[1]**(1./3)))
        for i, npVol in enumerate(volNpList):
            npVolT = np.transpose(npVol)
            volNp = npVolT.reshape((dim, dim, dim))
            nameVol = 'volume_reconstructed_%02d.mrc' % (i + 1)
            saveMrc(volNp.astype(dType), nameVol)

    def _mrcToNp(self, volList):
        listNpVol = []
        for vol in volList:
            volNp = loadMrc(vol, False)
            dim = volNp.shape[0]
            lenght = dim**3
            volList = volNp.reshape(lenght)
            listNpVol.append(volList)
        return listNpVol, listNpVol[0].dtype

    def _createMFile(self, matrix, name='matrix.txt'):
        f = open(name, 'w')
        for list in matrix:
            s = "%s\n" % list
            f.write(s)
        f.close()