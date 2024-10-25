import warnings

from pyworkflow.protocol.params import (PointerParam, FloatParam,
                                        StringParam, BooleanParam,
                                        EnumParam, IntParam, LEVEL_ADVANCED)

from pwem.protocols import ProtAnalysis3D
from pwem import ALIGN_NONE, ALIGN_PROJ
from cryomethods.convert import writeSetOfParticles
from xmipp3.convert import writeSetOfParticles as writeSetOfParticlesXmipp
from cryomethods.functions import NumpyImgHandler
import numpy as np
import os
from pwem.constants import NO_INDEX
from cryomethods import Plugin

class ProtLocProb(ProtAnalysis3D):
    """
    Given a map and the number of moments, the protocol estimates the local probability map.
    """
    _label = 'local probability map'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        group = form.addGroup('Reconstruction')
        group.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles",
                      help='Select the input images from the project.')

        group.addParam('reconstructRelion', BooleanParam, default=False,
                      label="Apply Relion (Yes), apply Xmipp (No)",
                      help='If set Yes, Relion reconstruction will be \n'
                           'applied. If set No, xmipp reconstruction will be')

        groupRelion = form.addGroup('Relion', condition="reconstructRelion")
        groupRelion.addParam('symmetryGroup', StringParam, default='c1',
                      label="Symmetry group",
                      help='See [[https://relion.readthedocs.io/'
                           'en/latest/Reference/Conventions.html#symmetry]'
                           '[Relion Symmetry]] page for a description '
                           'of the symmetry format accepted by Relion')

        groupRelion.addParam('maxResRelion', FloatParam, default=-1,
                      label="Maximum resolution (A)",
                      help='Maximum resolution (in Angstrom) to consider \n'
                           'in Fourier space (default Nyquist).')

        groupRelion.addParam('paddingFactorRelion', FloatParam, default=2.0,
                      label="Padding Factor Relion",
                      help='Padding factor for Relion reconstruction.')

        groupRelion.addParam('numMom', IntParam, default=5,
                      label="Number of moments",
                      help='Number of moments to estimate. Maximum 4')

        groupRelion.addParam('extraParametersRelion', StringParam, default='',
                 label="Extra parameters", help='Extra parameters for Relion \n'
                                                'recontruction')

        groupXmipp = form.addGroup('Xmipp', condition="not reconstructRelion")
        groupXmipp.addParam('symmetryGroup', StringParam, default='c1',
                       label="Symmetry group",
                       help='Enforce symmetry in projections')

        groupXmipp.addParam('maxRes', FloatParam, default=-1,
                       label="Maximum resolution (A)",
                       help='Maximum resolution (in Angstrom) to consider \n'
                            'in Fourier space (default Nyquist).')

        groupXmipp.addParam('paddingFactorXmipp', BooleanParam, default=True,
                     label="Padding Factor",
                     help='If selected, you can adjust projection and volume for Xmipp \n'
                          'reconstruction')

        groupXmipp.addParam('projection', FloatParam, default=2.0,
                       label="Number of Projections",
                       help='Number of projections for Xmipp reconstruction.',
                       condition="paddingFactorXmipp")

        groupXmipp.addParam('volume', FloatParam, default=2.0,
                       label="Volume Number",
                       help='Volume number for Xmipp reconstruction.',
                       condition="paddingFactorXmipp")

        groupXmipp.addParam('numMom', IntParam, default=5,
                           label="Number of moments",
                           help='Number of moments to estimate. Maximum 4')

        groupXmipp.addParam('extraParameters', StringParam, default='',
                 label="Extra parameters", help='Extra parameters for \n'
                                                'Xmipp reconstruction')

        form.addParallelSection(threads=1, mpi=1)


    # --------------------------- INSERT steps functions ----------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')

        num_mom = self.numMom.get()

        self.mom_order = []

        for m in range(1, num_mom+1):
            self._insertFunctionStep('_processParticles',m)
            self._insertFunctionStep('reconstructStep',m)

        self._insertFunctionStep('createOutputStep')


    def convertInputStep(self):
        """ Create the input file in STAR format as expected by Relion.
        If the input particles comes from Relion, just link the file.
        """
        imgSet = self.inputParticles.get()
        imgStar = self._getExtraPath('inputParticles.star')

        # Pass stack file as None to avoid write the images files
        writeSetOfParticles(imgSet, imgStar,
                                    outputDir=self._getTmpPath(),
                                    alignType=ALIGN_PROJ)

        imgXmd = self._getExtraPath('inputParticles.xmd')
        writeSetOfParticlesXmipp(imgSet, imgXmd)


    def createOutputStep(self):
        mom_order = self.mom_order

        m1 = NumpyImgHandler.loadMrc(os.path.join(self._getExtraPath(), mom_order[0]))
        m1[m1 < 0 ] = 0
        NumpyImgHandler.saveMrc(m1, os.path.join(self._getPath(),"1_mean.mrc"))


        m2 = NumpyImgHandler.loadMrc(os.path.join(self._getExtraPath(), mom_order[1]))
        m2[m2 < 0] = 0

        m12 = m1*m1
        m13 = m1*m1*m1
        m14 = m1 * m1 * m1 * m1

        variance = (m2 - m12)
        variance[variance < 0] = 0
        NumpyImgHandler.saveMrc(variance, os.path.join(self._getPath(), "2_variance.mrc"))


        m3 = NumpyImgHandler.loadMrc(os.path.join(self._getExtraPath(), mom_order[2]))
        m3[m3 < 0] = 0
        variance32 = np.sqrt(variance*variance*variance)
        skewness = (m3 - 3 * m1 * variance - m13) / (variance32+0.001)
        NumpyImgHandler.saveMrc(skewness, os.path.join(self._getPath(), "3_skewness.mrc"))


        del(skewness)
        m4 = NumpyImgHandler.loadMrc(os.path.join(self._getExtraPath(), mom_order[3]))
        m4[m4 < 0] = 0
        kurtosis = (m4 - 4 * m1 * m3 + 6 * m12 * m2 - 3 * m14 )/ (variance*variance+0.001)
        #kurtosis = np.nan_to_num(kurtosis)
        NumpyImgHandler.saveMrc(kurtosis, os.path.join(self._getPath(), "4_kurtosis.mrc"))


    def reconstructStep(self, m_index=1):

        env = Plugin.getEnviron()

        volume_name = 'm' + str(m_index) + '.mrc'
        imgSet = self.inputParticles.get()


        if self.reconstructRelion.get() == True:

            params_relion = ' --i %s' % self._getExtraPath('inputParticles.star') #input_particles
            params_relion += ' --o %s' % self._getExtraPath(volume_name) #output_volume
            params_relion += ' --sym %s' % self.symmetryGroup.get()
            params_relion += ' --pad %0.1f' % self.paddingFactorRelion.get()
            params_relion += ' --subset -1 --class -1'

            # Addition of the Sampling rate and the maximum resolution
            params_relion += ' --angpix %0.5f' % imgSet.getSamplingRate()
            params_relion += ' --maxres %0.3f' % self.maxResRelion.get()
            params_relion += ' %s' % self.extraParametersRelion.get()

            self.runJob('relion_reconstruct', params_relion)


        else:

            params = ' -i %s' % self._getExtraPath('inputParticles.xmd') #input_particles
            params += ' -o %s' % self._getExtraPath(volume_name) #output_volume
            params += ' --sym %s' % self.symmetryGroup.get()
            params += ' --padding %0.1f %0.1f' % (self.projection.get(), self.volume.get())


            # Addition of the Sampling rate, the maximum resolution and extra parameters (if needed)
            params += ' --sampling %0.5f' % imgSet.getSamplingRate()
            params += ' --max_resolution %0.3f' % self.maxRes.get()
            params += ' %s' % self.extraParameters.get()

            self.runJob('xmipp_reconstruct_fourier_accel', params, env=env)


        self.mom_order.append(volume_name)
        print('----------------------------------------')
        print(f'List of moments: ´{self.mom_order}')



    def _processParticles(self, m_index=1):

        imgSet = self.inputParticles.get()

        # New folder created where temporal images will be located
        mypath = os.path.join(self._getPath(), "temporal_images")
        os.makedirs(mypath, exist_ok=True)

        # We make a copy of the initial particle set, which will point to the new location
        imgSetNew = self._createSetOfParticles()
        imgSetNew.copyInfo(imgSet)

        for i, img in enumerate(imgSet):
            loc = img.getLocation()
            nombre_part = os.path.basename(loc[1])

            part = NumpyImgHandler.loadMrcSlice(str(loc[0])+'@'+loc[1], writable=True)
            loc_new_folder = (loc[0], os.path.join(mypath, str(loc[0]) + '_' + nombre_part))


            if m_index == 1:
                part_n = part
            elif m_index == 2:
                part_n = part * part
            elif m_index == 3:
                part_n = part * part * part
            elif m_index == 4:
                part_n = part * part * part * part

            NumpyImgHandler.saveMrc(part_n, loc_new_folder[1])


        if self.reconstructRelion.get() == True:
            imgStar = self._getExtraPath('inputParticles.star')
            print(f'------------------SAVE MOMENT ORDER {m_index}------------------')
            imgSetNew.copyItems(imgSet, updateItemCallback=self._setFileName)
            writeSetOfParticles(imgSetNew, imgStar,
                                outputDir=self._getExtraPath(),
                                alignType=ALIGN_PROJ)

        else:
            imgXmd = self._getExtraPath('inputParticles.xmd')
            print(f'------------------SAVE MOMENT ORDER {m_index}------------------')
            imgSetNew.copyItems(imgSet, updateItemCallback=self._setFileName)
            writeSetOfParticlesXmipp(imgSetNew, imgXmd,
                                outputDir=self._getExtraPath())



    # --------------------------- INFO functions ------------------------------
    def _getOutStack(self, index, fn):
        """ Return the output stack filename based on the input. """
        mypath = os.path.join(self._getPath(), "temporal_images")
        nombre_part = os.path.basename(fn)
        loc_new_folder = (index, os.path.join(mypath, str(index) + '_' + nombre_part))
        return loc_new_folder[1]


    def _setFileName(self, item, row=None):
        index, fn = item.getLocation()

        index = 1 if index == NO_INDEX else index
        item.setLocation(0, self._getOutStack(index, fn))

    def _methods(self):
        messages = []
        if hasattr(self, 'resolution_Volume'):
            messages.append(
                'Information about the method/article in ')
        return messages

    def _validate(self):
        return []

    def _summary(self):
        summary = []
        summary.append("Input volume: %s" % self.inputParticles.getNameId())
        summary.append("Number of moments: %s" % self.numMom.get())
        summary.append(" ")
        return summary

    def _citations(self):
        return ['Vargas2021']



