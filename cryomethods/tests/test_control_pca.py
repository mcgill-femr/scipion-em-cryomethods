from pyworkflow.utils import Environ
from pyworkflow.tests import *

from pyworkflow.em import ProtImportParticles, ProtImportVolumes
from cryomethods.protocols import ProtLandscapePCA
from cryomethods.protocols import ProtInitialVolumeSelector

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

class ControlTestPCA(TestBase):
    """ Check the images are converted properly to spider format. """

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestBase.setData()
        cls.protImport = cls.runImportParticles(cls.particlesFn, 7.08)
        cls.protImportVol = cls.runImportSingleVolume(cls.volumes, 7.08)

    def testControlPCA(self):
        def _runAutoClassifier(doGpu=False, label=''):
            print label
            autoClassifierProt = self.newProtocol(ProtLandscapePCA,
                                                  numberOfIterations=10,
                                                  resolToStop=27.0,
                                                  minPartsToStop=1000,
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



