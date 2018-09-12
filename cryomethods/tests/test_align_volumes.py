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
from cryomethods.convert import loadMrc, alignVolumes, saveMrc, applyTransforms


class TestBase(BaseTest):
    @classmethod
    def setData(cls, dataProject='relion_tutorial'):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.volumes = cls.dataset.getFile('import/case2/*class00?.mrc')


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
