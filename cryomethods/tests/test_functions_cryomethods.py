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
import pyworkflow.em.metadata as md
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
        projName = cls.__name__
        Manager().deleteProject(projName)
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
        prot = Prot3DAutoClassifier(classMethod=1)
        volList = sorted(glob(self.volumes))
        matProj, _ = prot._doPCA(volList)
        print(matProj)

    def testClustering(self):
        from itertools import izip
        Plugin.setEnviron()
        volList = self._getVolList()
        self._getAverageVol(volList)

        dictNames = {}
        groupDict = {}
        prot = Prot3DAutoClassifier(classMethod=1)
        print("Mehod: ", prot.classMethod.get())
        # matrix = self._estimatePCA(volList)
        matrix, _ = self._mrcToNp(volList)
        labels = prot._clusteringData(matrix)
        if labels is not None:
            f = open('method_%s.txt' % 1, 'w')
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
        volList = self._getVolList()

        dictNames = {}
        groupDict = {}
        prot = Prot3DAutoClassifier(classMethod=1)
        print("Mehod: ", prot.classMethod.get())
        npAvgMap, _ = prot._doAverageMaps(volList)
        # matrix, _ = prot._mrcToNp(volList, npAvgMap)
        matrix, _ = prot._doPCA(volList)
        labels = prot._clusteringData(matrix)
        if labels is not None:
            f = open('volumes_clustered.txt', 'w')
            for vol, label in izip (volList, labels):
                dictNames[vol] = label

            for key, value in sorted(dictNames.iteritems()):
                groupDict.setdefault(value, []).append(key)

            for key, value in groupDict.iteritems():
                valueStr = ' '.join(value)
                valueStr.replace('ScipionUserData', 'PROCESSING/TESLA')
                line = 'chimera %s\n' % valueStr
                f.write(line)
            f.close()
            print(labels)

    def testHandlingMd(self):
        from collections import defaultdict
        starFn = '/home/josuegbl/raw_final_data.star'
        fn = '/home/josuegbl/random_data.star'
        # labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
        mdData = md.MetaData(starFn)
        mdAux = md.MetaData()
        mdFinal = md.MetaData()

        mdAux.randomize(mdData)
        mdFinal.selectPart(mdAux, 1, 100)
        # listMd = []
        #
        # for row in md.iterRows(mdData, sortByLabel=md.RLN_PARTICLE_CLASS):
        #     clsPart = row.getValue(md.RLN_PARTICLE_CLASS)
        #     newClass = labels[clsPart-1] + 1
        #     row.setValue(md.RLN_PARTICLE_CLASS, newClass)
        #
        #     listMd.append((row, newClass))
        # res = defaultdict(list)
        # for v, k in listMd: res[k].append(v)
        #
        # for key, listMd in res.iteritems():
        #     mdInput = md.MetaData()
        #     for rowMd in listMd:
        #         objId = mdInput.addObject()
        #         rowMd.writeToMd(mdInput, objId)

        mdFinal.write(fn)


        # print(mdInput)
        # outMd.aggregate(mdInput, md.AGGR_MAX, md.RLN_PARTICLE_CLASS,
        #                 md.RLN_IMAGE_NAME, md.RLN_IMAGE_NAME)
        # outMd.aggregateSingleInt(mdInput, md.AGGR_MAX, md.RLN_PARTICLE_CLASS)
        # outMd.aggregateMdGroupBy(mdInput, md.AGGR_COUNT,
        #                          [md.RLN_PARTICLE_CLASS],
        #                          md.RLN_PARTICLE_CLASS, md.MDL_WEIGHT)
        # print(outMd)

            # if newClass != lastCls:
            #     # levelRuns.append(newClass)
            #     # makePath(self._getRunPath(self._level, newClass))
            #
            #     # if lastCls is not None:
            #     #     print("writing %s" % fn)
            #     #     mdInput.write(fn)
            #
            #     fn = '/home/josuegbl/rawclass_final_data.star'
            #     lastCls = newClass
            #
            #
            # print("writing %s and ending the loop" % fn)
            # mdInput.write(fn)

        # mapIds = self._getFinalMapIds()
        # claseId = 0

    def _getVolList(self):
        volList = glob('/home/josuegbl/ScipionUserData/projects/Spliceosome_Tesla/MAPS/*_fil20.mrc')
#        fixedPath = '/mnt/tesla/data/josuegbl/ScipionUserData/projects/Spliceosome_Tesla/'
#        filePath = 'Runs/001594_Prot3DAutoClassifier/extra/raw_final_model.star'
#        wholePath = fixedPath + filePath
#        mdModel = md.MetaData(wholePath)
#        for row in md.iterRows(mdModel):
#            volFn = row.getValue('rlnReferenceImage')
#            fullVolFn = fixedPath + volFn
#            volList.append(fullVolFn)
        return volList

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

    def _createMFile(self, matrix, name='matrix.txt'):
        f = open(name, 'w')
        for list in matrix:
            s = "%s\n" % list
            f.write(s)
        f.close()
