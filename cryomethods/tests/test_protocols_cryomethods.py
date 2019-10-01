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

from pyworkflow.em import ProtImportParticles, ProtImportVolumes
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

