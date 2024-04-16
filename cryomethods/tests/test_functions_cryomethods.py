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
from glob import glob
from os.path import basename

import numpy as np
from pyworkflow.utils import copyFile
from pyworkflow.tests import *
import pwem.emlib.metadata as md
from cryomethods import Plugin
from cryomethods.protocols import Prot3DAutoClassifier,ProtVolClustering
from cryomethods.functions import NumpyImgHandler
from cryomethods.functions import correctAnisotropy
from cryomethods.functions import compute_ctf_np, compute_ctf_torch
from cryomethods.functions import MlMethods

class TestBase(BaseTest):
    @classmethod
    def setData(cls, dataProject='relion_tutorial'):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.volume1 = cls.dataset.getFile('import/case2/relion_it015_class003.mrc')
        cls.volumes = cls.dataset.getFile('import/case2/*class00?.mrc')
        cls.clsVols = cls.dataset.getFile('classVols/map_rLev-0??.mrc')
        cls.particles = cls.dataset.getFile('import/case2/relion_it015_data.star')



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
        print(volList)
        volRef = volList.pop(0)
        maxScore = 0
        npIh = NumpyImgHandler()
        for vol in volList:
            volRefNp = npIh.loadMrc(volRef)
            volNp = npIh.loadMrc(vol)
            volNpFp = np.fliplr(volNp)

            axis, shifts, angles, score = npIh.alignVolumes(volNp, volRefNp)
            axisf, shiftsf, anglesf, scoref = npIh.alignVolumes(volNpFp, volRefNp)
            print('scores : w/o flip- %03f w flip %03f' %(score, scoref))
            if scoref > score:
                print('angles:', anglesf[0], anglesf[1], anglesf[2],)
                print('shifts:', shiftsf[0], shiftsf[1], shiftsf[2],)
                npVol = npIh.applyTransforms(volNpFp, shiftsf, anglesf, axisf)
                print('flipped map is better: ', vol)
            else:
                print('angles:', angles[0], angles[1], angles[2],)
                print('shifts:', shifts[0], shifts[1], shifts[2],)
                npVol = npIh.applyTransforms(volNp, shifts, angles, axis)
                print('original map is better ', vol)

            #npIh.saveMrc(npVol, '/home/josuegbl/'+ basename(vol))

    def testPCA(self):
        Plugin.setEnviron()
        prot = Prot3DAutoClassifier(classMethod=1)
        volList = sorted(glob(self.volumes))
        covMatrix, listNpVol = MlMethods.getCovMatrixAuto(volList,1)
        matProj, _ = MlMethods.doPCA(covMatrix)
        print(matProj)

    def testClustering(self):
        Plugin.setEnviron()

        volList = sorted(glob(self.volumes))
        #volList = self._getVolList()
        #self._getAverageVol(volList)
        npIh = NumpyImgHandler()
        npIh.getAverageMap(volList)

        dictNames = {}
        groupDict = {}
        #prot = Prot3DAutoClassifier(classMethod=1)
        prot = ProtVolClustering(classMethod=1)
        print("Mehod: ", prot.classMethod.get())
        # matrix = self._estimatePCA(volList)
        matrix, _ = prot._mrcToNp(volList)
        labels = prot._clusteringData(matrix)
        if labels is not None:
            f = open('method_%s.txt' % 1, 'w')
            for vol, label in zip (volList, labels):
                dictNames[vol] = label

            for key, value in sorted(dictNames.items()):
                groupDict.setdefault(value, []).append(key)

            for key, value in groupDict.items():
                line = '%s %s\n' % (key, value)
                f.write(line)
            f.close()
            print(labels)

    def testAffinityProp(self):
        from cryomethods.functions import MlMethods, NumpyImgHandler
        Plugin.setEnviron()

        volList = sorted(glob(self.volumes))
        #volList = self._getVolList()
        ml = MlMethods()
        npIh = NumpyImgHandler()

        dictNames = {}
        groupDict = {}
        matrix = npIh.getAllNpList(volList, 2)

        # covMatrix, listNpVol = ml.getCovMatrixAuto(volList, 2)
        # eigenVec, eigVal = ml.doPCA(covMatrix, 1)
        # matrix = ml.getMatProjAuto(listNpVol, eigenVec)

        labels = ml.doSklearnAffProp(matrix)
        # labels = ml.doSklearnKmeans(matrix)
        # labels = ml.doSklearnDBSCAN(matrix)
        print(labels)

        if labels is not None:
            f = open('volumes_clustered.txt', 'w')
            for vol, label in zip(volList, labels):
                dictNames[vol] = label

                #destFn = '/home/josuegbl/PROCESSING/TESLA/projects/RNC_HTLnd2/MAPS' + basename(vol)
                destFn = basename(vol)
                copyFile(vol, destFn)
            for key, value in sorted(dictNames.items()):
                groupDict.setdefault(value, []).append(key)

            counter = 0
            for key, value in groupDict.items():
                valueStr = ' '.join(value)
                line = 'chimera %s\n' % valueStr
                f.write(line)
                counter += 1
                avgFn = 'map_average_class_%02d.mrc' %counter
                avgNp,_ = npIh.getAverageMap(value)
                npIh.saveMrc(avgNp, avgFn)
            f.close()

        # import shutil
        # for fn in volList:
        #     dir = '/home/josuegbl/PROCESSING/30S_delta_yjeQ/'
        #     newFn = dir + 'map_id-4.' + fn.split('map_id-')[1]
        #     shutil.move(fn, newFn)

        # for line in matrix:
        #     l = map(abs, line)
        #     fig = plt.figure()
        #     plt.subplot(1, 2, 1)
        #     plt.plot(eigenVals)
        #     plt.subplot(1, 2, 2)
        #     plt.plot(l)
        #     plt.show()

    def testHandlingMd(self):
        #starFn = '/home/josuegbl/raw_final_data.star'
        starFn =  self.particles
        #fn = '/home/josuegbl/random_data.star'
        fn = ''
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
       # volList = glob('/home/josuegbl/PROCESSING/30S_delta_yjeQ/map_id'
       #                '-?.??.???.mrc')
       volList = []
       fixedPath = '/home/josuegbl/PROCESSING/TESLA/projects/RNC_HTLnd2/'
       filePath = 'Runs/001034_Prot3DAutoClassifier/extra/final_model.star'
       wholePath = fixedPath + filePath
       mdModel = md.MetaData(wholePath)
       for row in md.iterRows(mdModel):
           volFn = row.getValue('rlnReferenceImage')
           fullVolFn = fixedPath + volFn
           volList.append(fullVolFn)
       return volList

    def _reconstructMap(self, matProj):
        from glob import glob
        listBaseVol = glob('volume_base*.mrc')
        sortedList = sorted(listBaseVol)
        listNpBase, dType = self._mrcToNp(sortedList)
        volNpList = np.dot(matProj, listNpBase)
        dim = int(round(volNpList.shape[1]**(1./3)))
        npIh = NumpyImgHandler()
        for i, npVol in enumerate(volNpList):
            npVolT = np.transpose(npVol)
            volNp = npVolT.reshape((dim, dim, dim))
            nameVol = 'volume_reconstructed_%02d.mrc' % (i + 1)
            npIh.saveMrc(volNp.astype(dType), nameVol)

    def _createMFile(self, matrix, name='matrix.txt'):
        f = open(name, 'w')
        for list in matrix:
            s = "%s\n" % list
            f.write(s)
        f.close()


class TestCorrection(TestBase):
    @classmethod
    def setUpClass(cls):
        projName = cls.__name__
        Manager().deleteProject(projName)
        setupTestProject(cls)
        TestBase.setData()


    def testCorrectAnisotrophy(self):
        Plugin.setEnviron()
        npIh = NumpyImgHandler()
        volNp = npIh.loadMrc(self.volume1)
        vol, volFt = correctAnisotropy(volNp, 0.5, 0.55, 0.1,0.45)
        print ("shape: ", vol.shape)
        npIh.saveMrc(vol.astype(volNp.dtype), "corrected.mrc")
        npIh.saveMrc(volFt.astype(volNp.dtype), "corrected_ft.mrc")

class TestCTF(TestBase):

    @classmethod
    def setUpClass(cls):
        projName = cls.__name__
        Manager().deleteProject(projName)
        setupTestProject(cls)
        TestBase.setData()

    def testCTF_NP_Generate(self):
        import torch
        Plugin.setEnviron()
        sampling_rate = 2  # A/px
        #Nyquist frequency
        nyquist_freq = 1 / (2 * sampling_rate)
        N = 100  # Matrix size
        freqs = np.linspace(-nyquist_freq, nyquist_freq, N)
        freqX, freqY = np.meshgrid(freqs, freqs)

        freqs = np.column_stack((freqX.flatten(), freqY.flatten()))
        dfu = float(10000)
        dfv = float(10000)
        angle = float(10000)
        volt = float(300)
        cs = float(2.6)
        w = float(0.1)
        phase_shift = float(0)
        bfactor = float(250)

        ctf = compute_ctf_np(freqs,dfu,dfv,angle,volt,cs,w,phase_shift,bfactor)
        ctf_2d = ctf.reshape(N,N)
        npIh = NumpyImgHandler()
        print('ctf_calculated numpy:',ctf)
        npIh.saveMrc(ctf_2d,'CTF.mrc')


    def testCTF_TORCH_Generate(self):
        import torch
        Plugin.setEnviron()
        sampling_rate = 2  # A/px
        #Nyquist frequency
        nyquist_freq = 1 / (2 * sampling_rate)
        N = 100  # Matrix size
        freqs = np.linspace(-nyquist_freq, nyquist_freq, N)
        freqX, freqY = np.meshgrid(freqs, freqs)

        freqs = torch.from_numpy(np.column_stack((freqX.flatten(), freqY.flatten())))
        freqs.to(torch.float64)
        dfu = torch.tensor(0.2)
        dfv = torch.tensor(0.8)
        angle = torch.tensor(-0.5)
        volt = torch.tensor(300)
        cs = torch.tensor(0)
        w = torch.tensor(0)
        phase_shift = torch.tensor(0)
        bfactor = torch.tensor(250)

        ctf = compute_ctf_torch(freqs,dfu,dfv,angle,volt,cs,w,phase_shift,bfactor)

        ctf_2d = ctf.numpy().reshape(N,N)
        npIh = NumpyImgHandler()
        print('ctf_calculated torch:',ctf)
        npIh.saveMrc(ctf_2d,'CTF2.mrc')

    def testCTF_TORCH_Generate_Stack(self):
        import torch
        Plugin.setEnviron()
        sampling_rate = 2  # A/px
        #Nyquist frequency
        nyquist_freq = 1 / (2 * sampling_rate)
        N = 100  # Matrix size
        freqs = np.linspace(-nyquist_freq, nyquist_freq, N)
        freqX, freqY = np.meshgrid(freqs, freqs)

        freqs = torch.from_numpy(np.column_stack((freqX.flatten(), freqY.flatten())))
        freqs.to(torch.float64)

        numCTFs = 50
        dfu = torch.randint(1000, 30000, (numCTFs, 1))
        dfv = torch.randint(1000, 30000, (numCTFs, 1))
        angle = torch.randint(0, 360, (numCTFs, 1))
        volt = torch.tensor(300)
        cs = torch.tensor(2.6)
        w = torch.tensor(0.1)
        phase_shift = torch.tensor(0)
        bfactor = torch.tensor(250)

        ctf = compute_ctf_torch(freqs,dfu,dfv,angle,volt,cs,w,phase_shift,bfactor)

        ctf_2d = ctf.numpy().reshape(numCTFs,N,N)
        npIh = NumpyImgHandler()
        print('ctf_calculated torch:',ctf)
        npIh.saveMrc(ctf_2d,'CTF3.mrcs')

