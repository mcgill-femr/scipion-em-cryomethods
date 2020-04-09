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
from pyworkflow.utils import Environ
from pyworkflow.tests import *

from pyworkflow.em import ProtImportParticles, ProtImportVolumes, ProtSubSet
from cryomethods.protocols import *

class TestBase(BaseTest):
    @classmethod
    def setData(cls, dataProject='relion_tutorial'):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.particlesFn = cls.dataset.getFile('import/case2/particles.sqlite')
        cls.volumes = cls.dataset.getFile('import/case2/')

    def checkOutput(self, prot, outputName, conditions=[]):
        """ Check that an output was generated and
        the condition is valid.
        """
        o = getattr(prot, outputName, None)
        locals()[outputName] = o
        self.assertIsNotNone(o, "Output: %s is None" % outputName)
        for cond in conditions:
            self.assertTrue(eval(cond), 'Condition failed: ' + cond)

    @classmethod
    def runImportParticles(cls, pattern, samplingRate, checkStack=False):
        """ Run an Import particles protocol. """
        protImport = cls.newProtocol(ProtImportParticles,
                                     importFrom=4,
                                     sqliteFile=pattern,
                                     samplingRate=samplingRate,
                                     checkStack=checkStack)
        cls.launchProtocol(protImport)
        # check that input images have been imported (a better way to do this?)
        if protImport.outputParticles is None:
            raise Exception('Import of images: %s, failed. outputParticles '
                            'is None.' % pattern)
        return protImport

    @classmethod
    def runImportVolumes(cls, pattern, samplingRate):
        """ Run an Import particles protocol. """
        protImport = cls.newProtocol(ProtImportVolumes,
                                     filesPath=pattern,
                                     filesPattern='relion*_class*.mrc',
                                     samplingRate=samplingRate)
        cls.launchProtocol(protImport)
        return protImport

    @classmethod
    def runImportSingleVolume(cls, pattern, samplingRate):
        """ Run an Import particles protocol. """
        protImport = cls.newProtocol(ProtImportVolumes,
                                     filesPath=pattern,
                                     filesPattern='relion*_class001.mrc',
                                     samplingRate=samplingRate)
        cls.launchProtocol(protImport)
        return protImport


class Test3DAutoClasifier(TestBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestBase.setData()
        cls.protImport = cls.runImportParticles(cls.particlesFn, 7.08)
        cls.protImportVol = cls.runImportSingleVolume(cls.volumes, 7.08)

    def testAutoClassify(self):
        def _runAutoClassifier(doGpu=False, label=''):
            print label
            autoClassifierProt = self.newProtocol(Prot3DAutoClassifier,
                                                  numberOfIterations=10,
                                                  resolToStop=27.0,
                                                  minPartsToStop=2000,
                                                  classMethod=1,
                                                  numberOfMpi=4,
                                                  numberOfThreads=1)
            autoClassifierProt.setObjLabel(label)
            autoClassifierProt.inputParticles.set(
                self.protImport.outputParticles)
            autoClassifierProt.inputVolumes.set(self.protImportVol.outputVolume)

            autoClassifierProt.doGpu.set(doGpu)

            self.launchProtocol(autoClassifierProt)
            return autoClassifierProt

        def _checkAsserts(relionProt):
            self.assertIsNotNone(relionProt.outputVolumes, "There was a "
                                                           "problem")

        volSelGpu = _runAutoClassifier(True, "Run Auto-classifier GPU")
        _checkAsserts(volSelGpu)


class Test2DAutoClasifier(TestBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestBase.setData()
        cls.protImport = cls.runImportParticles(cls.particlesFn, 7.08)

    def testAutoClassify(self):
        def _runAutoClassifier(doGpu=False, label=''):
            print label
            autoClassifierProt = self.newProtocol(Prot2DAutoClassifier,
                                                  numberOfIterations=10,
                                                  minPartsToStop=700,
                                                  classMethod=1,
                                                  numberOfMpi=4,
                                                  numberOfThreads=1)
            autoClassifierProt.setObjLabel(label)
            autoClassifierProt.inputParticles.set(
                self.protImport.outputParticles)

            autoClassifierProt.doGpu.set(doGpu)

            self.launchProtocol(autoClassifierProt)
            return autoClassifierProt

        def _checkAsserts(relionProt):
            self.assertIsNotNone(relionProt.outputClasses, "There was a "
                                                           "problem")

        volSelGpu = _runAutoClassifier(True, "Run Auto-classifier GPU")
        _checkAsserts(volSelGpu)


class TestVolumeSelector(TestBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestBase.setData()
        cls.protImport = cls.runImportParticles(cls.particlesFn, 7.08)
        cls.protImportVol = cls.runImportVolumes(cls.volumes, 7.08)

    def testInitialVolumeSelector(self):
        def _runVolumeSelector(doGpu=False, label=''):
            volSelectorProt = self.newProtocol(ProtInitialVolumeSelector,
                                               targetResol=28.32,
                                               numOfVols=2,
                                               numberOfMpi=3, numberOfThreads=1)

            volSelectorProt.setObjLabel(label)
            volSelectorProt.inputParticles.set(self.protImport.outputParticles)
            volSelectorProt.inputVolumes.set(self.protImportVol.outputVolumes)

            volSelectorProt.doGpu.set(doGpu)
            return volSelectorProt

        def _checkAsserts(prot):
            self.assertIsNotNone(prot.outputVolumes, "There was a problem with "
                                                      "Initial Volume Selector")

        environ = Environ(os.environ)
        cudaPath = environ.getFirst(('RELION_CUDA_LIB', 'CUDA_LIB'))

        if cudaPath is not None and os.path.exists(cudaPath):
            volSelGpu = _runVolumeSelector(True, "Run Volume Selector GPU")
            self.launchProtocol(volSelGpu)
            _checkAsserts(volSelGpu)

        else:
            volSelNoGPU = _runVolumeSelector(False, "Volume Selector No GPU")
            volSelNoGPU.numberOfMpi.set(4)
            volSelNoGPU.numberOfThreads.set(2)
            self.launchProtocol(volSelNoGPU)
            _checkAsserts(volSelNoGPU)


class TestDirectionalPruning(TestBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestBase.setData()

        cls.protImport = cls.runImportParticles(cls.particlesFn, 7.08)
        cls.protImportVol = cls.runImportSingleVolume(cls.volumes, 7.08)

    def test_solidAndSplit(self):
        # Let's keep a smaller subset of particles to speed-up computations
        protSubset = self.newProtocol(ProtSubSet,
                                      objLabel='subset 1K',
                                      chooseAtRandom=True,
                                      nElements=1000)

        protSubset.inputFullSet.set(self.protImport.outputParticles)
        self.launchProtocol(protSubset)

        # We use a coarse angular sampling of 20 to speed-up test
        protSolid = self.newProtocol(ProtDirectionalPruning,
                                objLabel='directional classes 1',
                                angularSampling=20,
                                angularDistance=25,
                                numberOfMpi=4
                                )

        protSolid.inputVolume.set(self.protImportVol.outputVolume)
        protSolid.inputParticles.set(protSubset.outputParticles)
        self.launchProtocol(protSolid)
        self.checkOutput(protSolid, 'outputParticles')

        protSolid1 = self.newProtocol(ProtDirectionalPruning,
                                objLabel='directional classes 1',
                                classMethod=1,
                                angularSampling=20,
                                angularDistance=25,

                                numberOfMpi=4
                                )

        protSolid1.inputVolume.set(self.protImportVol.outputVolume)
        protSolid1.inputParticles.set(protSubset.outputParticles)
        self.launchProtocol(protSolid1)
        self.checkOutput(protSolid1, 'outputParticles')

        protSolid2 = self.newProtocol(ProtDirectionalPruning,
                                      objLabel='directional classes 1',
                                      classMethod=2,
                                      numberOfIterations=5,
                                      regularisationParamT=2,
                                      numberOfMpi=4
                                      )

        protSolid2.inputVolume.set(self.protImportVol.outputVolume)
        protSolid2.inputParticles.set(protSubset.outputParticles)
        self.launchProtocol(protSolid2)
        self.checkOutput(protSolid2, 'outputParticles')


class TestClass3DRansac(TestBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestBase.setData()

        cls.protImport = cls.runImportParticles(cls.particlesFn, 7.08)
        cls.protImportVol = cls.runImportSingleVolume(cls.volumes, 7.08)

    def test_solidAndSplit(self):

        # Let's keep a smaller subset of particles to speed-up computations
        protSubset = self.newProtocol(ProtSubSet,
                                      objLabel='subset 1K',
                                      chooseAtRandom=True,
                                      nElements=1000)

        protSubset.inputFullSet.set(self.protImport.outputParticles)
        self.launchProtocol(protSubset)


        # We use a coarse angular sampling of 20 to speed-up test
        DransacProt = self.newProtocol(ProtClass3DRansac,
                                        objLabel='directional classes 1',
                                        Class2D=2,
                                        angularSampling=20,
                                        angularDistance=25,
                                        numberOfIterations=5,
                                        regularisationParamT=2,
                                        numClasses=5,
                                        numberOfMpi=4
                                        )

        DransacProt.inputVolume.set(self.protImportVol.outputVolume)
        DransacProt.inputParticles.set(protSubset.outputParticles)
        self.launchProtocol(DransacProt)


        # DransacProt1 = self.newProtocol(ProtClass3DRansac,
        #                             objLabel='directional classes 1',
        #                             Class2D=0,
        #                             angularSampling=20,
        #                             angularDistance=25,
        #                             numClasses=5,
        #                             numberOfMpi=4
        #                             )
        #
        #
        # DransacProt1.inputVolume.set(self.protImportVol.outputVolume)
        # DransacProt1.inputParticles.set(protSubset.outputParticles)
        # self.launchProtocol(DransacProt1)
        #
        #
        # DransacProt2= self.newProtocol(ProtClass3DRansac,
        #                             objLabel='directional classes 1',
        #                             Class2D=1,
        #                             angularSampling=20,
        #                             angularDistance=25,
        #                             numClasses=5,
        #                             numberOfMpi=4
        #                             )
        #
        # DransacProt2.inputVolume.set(self.protImportVol.outputVolume)
        # DransacProt2.inputParticles.set(protSubset.outputParticles)
        # self.launchProtocol(DransacProt2)

class TestVolumeClustering(TestBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testVolumeClustering(self):
        protImport = self.newProtocol(ProtImportVolumes,
                                     filesPath='/home/josuegbl/SOFTWARE/SCIPION/scipion/data/tests/BetaGClass',
                                     filesPattern='*.mrc',
                                     samplingRate=1.7)
        self.launchProtocol(protImport)

        prot = self.newProtocol(ProtVolClustering,
                                alignVolumes=False)
        prot.setObjLabel('test')
        prot.inputVolumes.set(protImport.outputVolumes)
        self.launchProtocol(prot)
        self.assertIsNotNone(prot.outputVolumes, "There was a problem...")

