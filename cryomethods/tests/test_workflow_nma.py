# **************************************************************************
# *
# * Authors:     J.M. de la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

from pyworkflow.tests.em.workflows.test_workflow import TestWorkflow



from cryomethods.protocols.Merge_NMA import ProtNmaMerge

from xmipp3.protocols.pdb.protocol_pseudoatoms_base import NMA_MASK_THRE
from pyworkflow.tests import *

from pyworkflow.em import ProtImportParticles, ProtImportVolumes


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
    def runImportSingleVolume(cls, pattern, samplingRate):
        """ Run an Import particles protocol. """
        protImport = cls.newProtocol(ProtImportVolumes,
                                     filesPath=pattern,
                                     filesPattern='relion*_class001.mrc',
                                     samplingRate=samplingRate)
        cls.launchProtocol(protImport)
        return protImport



class CryoMetTestNMA(TestBase):
    """ Check the images are converted properly to spider format. """
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestBase.setData()
        cls.protImport = cls.runImportParticles(cls.particlesFn, 7.08)
        cls.protImportVol = cls.runImportSingleVolume(cls.volumes, 7.08)

    #def test_nma1(self):

        #""" Run NMA simple workflow for both Atomic and Pseudoatoms. """

        # ------------------------------------------------
        # Case 1. Import a Pdb -> NMA
        # ------------------------------------------------

        # Import a PDB
        #protImportPdb = self.newProtocol(ProtImportPdb, inputPdbData=1,
        #                                 pdbFile=self.ds.getFile('pdb'))
        #self.launchProtocol(protImportPdb)

        # Launch NMA for PDB imported
        #protNMA1 = self.newProtocol(CryoMethProtConvertToPseudoAtoms,
         #                           cutoffMode=0)
        #protNMA1.inputStructure.set(protImportPdb.outputPdb)
        #self.launchProtocol(protNMA1)

        # # Import the set of particles
        # # (in this order just to be in the middle in the tree)
        # protImportParts = self.newProtocol(ProtImportParticles,
        #                                   filesPath=self.ds.getFile(
        #                                        'particles'),
        #                                    samplingRate=1.0)
        # self.launchProtocol(protImportParts)
        #
        # # Launch NMA alignment, but just reading result from a previous metadata
        # protAlignment = self.newProtocol(CryoMethProtAlignmentNMA,
        #                                  modeList='7-9',
        #                                   copyDeformations=self.ds.getFile(
        #                                       'gold/pseudo_run1_images.xmd'))
        # protAlignment.inputModes.set(protNMA1.outputModes)
        # protAlignment.inputParticles.set(protImportParts.outputParticles)
        # self.launchProtocol(protAlignment)
        #
        # # Launch Dimred after NMA alignment
        # protDimRed = self.newProtocol(CryoMethProtDimredNMA,
        #             dimredMethod=0,  # PCA
        #            reducedDim=2)
        # protDimRed.inputNMA.set(protAlignment)
        # self.launchProtocol(protDimRed)
    def test_nma2(self):
        # ------------------------------------------------
        # Case 2. Import Vol -> Pdb -> NMA
        # ------------------------------------------------
        # Import the set of particles
        # (in this order just to be in the middle in the tree)
        #protImportParts = self.newProtocol(ProtImportParticles,
         #                                   filesPath=self.ds.getFile(
         #                                       'particles'),
         #                                   samplingRate=1.0)
        #self.launchProtocol(protImportParts)

        # Convert the Volume to Pdb
        protConvertVol = self.newProtocol(ProtNmaMerge)
        protConvertVol.inputParticles.set(self.protImport.outputParticles)
        protConvertVol.subsetSize.set(100)
        protConvertVol.inputVolume.set(self.protImportVol.outputVolume)
        protConvertVol.maskMode.set(NMA_MASK_THRE)
        protConvertVol.maskThreshold.set(0.2)
        protConvertVol.pseudoAtomRadius.set(2.5)
        self.launchProtocol(protConvertVol)

        # Import the set of particles
        # (in this order just to be in the middle in the tree)

        # # convert pdb to volume
        #
        #
        # protConvert = self.newProtocol(CryoMethProtConvertPdb,
        #                                inputPdbData=ProtImportPdb.IMPORT_FROM_FILES,
        #                                pdbFile= self.ds.getFile( ))
        # protConvert.pdbFile.set(protNMA2.outputModes)
        # self.launchProtocol(protConvert)

        #protAlignment = self.newProtocol(
        #    CryoMethProtAlignmentNMA,
        #    modeList='7-9',
        #    copyDeformations=self.ds.getFile(
        #        'gold/pseudo_run1_images.xmd'))
        #protAlignment.inputModes.set(protNMA2.outputModes)
        #protAlignment.inputParticles.set(protImportParts.outputParticles)
        #self.launchProtocol(protAlignment)
        #self.assertIsNotNone(protConvert.outputVolume.getFileName(),
         #                    "There was a problem with the conversion")
        #self.assertAlmostEqual(protConvert.outputVolume.getSamplingRate(),
         #                      protConvert.sampling.get(), places=1,
         #                      msg="wrong sampling rate")
        #self.assertAlmostEqual(protConvert.outputVolume.getDim()[0],
         #                      protConvert.size.get(), places=1,
         #                      msg="wrong size")

        # Launch NMA alignment, but just reading result from a previous metadata




# Launch Dimred after NMA alignment
         #protDimRed = self.newProtocol(CryoMethProtDimredNMA,
         #                              dimredMethod=0,  # PCA
         #                              reducedDim=2)
         #protDimRed.inputNMA.set(protAlignment)
         #self.launchProtocol(protDimRed)


