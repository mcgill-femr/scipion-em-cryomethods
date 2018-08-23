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
from pyworkflow.tests import *
from cryomethods import Plugin
from cryomethods.convert import loadMrc, alignVolumes

# from pyworkflow.em import ProtImportParticles, ProtImportVolumes
# from cryomethods.protocols import ProtInitialVolumeSelector

class TestBase(BaseTest):
    @classmethod
    def setData(cls, dataProject='relion_tutorial'):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.volumes = cls.dataset.getFile('import/case2/*class00?.mrc')

    # def checkOutput(self, prot, outputName, conditions=[]):
    #     """ Check that an output was generated and
    #     the condition is valid.
    #     """
    #     o = getattr(prot, outputName, None)
    #     locals()[outputName] = o
    #     self.assertIsNotNone(o, "Output: %s is None" % outputName)
    #     for cond in conditions:
    #         self.assertTrue(eval(cond), 'Condition failed: ' + cond)
    #
    # @classmethod
    # def runImportParticles(cls, pattern, samplingRate, checkStack=False):
    #     """ Run an Import particles protocol. """
    #     protImport = cls.newProtocol(ProtImportParticles,
    #                                  importFrom=4,
    #                                  sqliteFile=pattern,
    #                                  samplingRate=samplingRate,
    #                                  checkStack=checkStack)
    #     cls.launchProtocol(protImport)
    #     # check that input images have been imported (a better way to do this?)
    #     if protImport.outputParticles is None:
    #         raise Exception('Import of images: %s, failed. outputParticles '
    #                         'is None.' % pattern)
    #     return protImport
    #
    # @classmethod
    # def runImportVolumes(cls, pattern, samplingRate):
    #     """ Run an Import particles protocol. """
    #     protImport = cls.newProtocol(ProtImportVolumes,
    #                                  filesPath=pattern,
    #                                  filesPattern='relion*_class*.mrc',
    #                                  samplingRate=samplingRate)
    #     cls.launchProtocol(protImport)
    #     return protImport


class TestAlignVolumes(TestBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestBase.setData()
        # cls.protImport = cls.runImportParticles(cls.particlesFn, 7.08)
        # cls.protImportVol = cls.runImportVolumes(cls.volumes, 7.08)


    def testAlignVolumes(self):
        Plugin.setEnviron()

        volList = sorted(glob(self.volumes))
        volRef = volList.pop(0)

        for vol in volList:
            volRefNp = loadMrc(volRef)
            volNp = loadMrc(vol)
            axis, shifts, angles, score = alignVolumes(volNp, volRefNp)
            print(axis, shifts, angles, score)
