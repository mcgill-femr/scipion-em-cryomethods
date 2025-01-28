from pyworkflow.protocol.params import (PointerParam, FloatParam,
                                        StringParam, BooleanParam,
                                        EnumParam, IntParam)

from pwem.protocols import ProtAnalysis3D
from pwem import ALIGN_PROJ
from cryomethods.convert import writeSetOfParticles
from xmipp3.convert import writeSetOfParticles as writeSetOfParticlesXmipp
from cryomethods.functions import NumpyImgHandler
import numpy as np
import os
from pwem.constants import NO_INDEX
from cryomethods import Plugin
import matplotlib.pyplot as plt

PROB_DENSITY_FUNCT = 0
ACC_MOMENTS = 1
RELION_RECONSTRUCTION = 0
XMIPP_RECONSTRUCTION = 1


class ProtLocPDF(ProtAnalysis3D):
    """
    Given a map and the number of moments, the protocol estimates the local probability map.
    """
    _label = 'locPDF'
        # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles",
                      help='Select the input images from the project.')

        form.addParam('methodApply', EnumParam,
                  choices=['pdf', 'accumulative moments'],
                  default=PROB_DENSITY_FUNCT,
                  label='Method to apply', display=EnumParam.DISPLAY_COMBO,
                  help='Decide which method you want to apply to your particles.\n'
                       '1. Probability density function, calculates it based on the range.\n'
                       '2. Accumulative moments. Apply the method of moments.\n'
                      )

        # -------------------------------- Pdf ----------------------------------------
        form.addParam('numBins', IntParam, default=10,
                      condition='methodApply==%d' % PROB_DENSITY_FUNCT,
                      label="Number of bins",
                      help='Number of bins')

        form.addParam('minRange', FloatParam, default=-1.0,
                      condition='methodApply==%d' % PROB_DENSITY_FUNCT,
                      label="Minimum value of the range",
                      help='Minimum value of the range')

        form.addParam('maxRange', FloatParam, default=1.0,
                      condition='methodApply==%d' % PROB_DENSITY_FUNCT,
                      label="Maximum value of the range",
                      help='Maximum value of the range')

        # -------------------------------- Moments ----------------------------------------
        #form.addParam('numMom', IntParam, default=4,
        #              condition='methodApply==%d' % ACC_MOMENTS,
        #               label="Number of moments to estimate\n "
        #                     "(only 4)",
        #               help='Number of moments to estimate. Only 4')

        form.addParam('numMom', BooleanParam, default=True,
                      condition='methodApply==%d' % ACC_MOMENTS,
                      label="4 moments will be estimated",
                      help='If selected, 4 moments will be calculated')

        # ----------------------------------Bootstrao---------------------------------
        group = form.addGroup('Bootstrap')
        group.addParam('numBatches', IntParam, default=10,
                             label="Number of batches",
                             help='Number of batches to apply bootstrap to')


        group.addParam('numSamples', IntParam, default=50,
                             label="Number of samples/particles per batch",
                             help='Number of samples/particles each batch will contain')

        group.addParam('reconstruction', EnumParam,
                      choices=['Relion reconstruction', 'Xmipp reconstruction'],
                      default=XMIPP_RECONSTRUCTION,
                      label='Reconstruction to be applied', display=EnumParam.DISPLAY_COMBO,
                      help='Select which reconstruction you want to apply\n'
                           '1. Relion reconstruction.\n'
                           '2. Xmipp reconstruction. \n')

        # ----------------------------------Relion reconstruction---------------------------------
        groupRelion = form.addGroup('Relion', condition="reconstruction==%d" % RELION_RECONSTRUCTION)
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

        groupRelion.addParam('extraParametersRelion', StringParam, default='',
                 label="Extra parameters", help='Extra parameters for Relion \n'
                                                'recontruction')

        #----------------------------------Xmipp reconstruction---------------------------------
        groupXmipp = form.addGroup('Xmipp', condition="reconstruction==%d" % XMIPP_RECONSTRUCTION) #"not reconstructRelion"
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

        groupXmipp.addParam('extraParameters', StringParam, default='',
                 label="Extra parameters", help='Extra parameters for \n'
                                                'Xmipp reconstruction')

        form.addParallelSection(threads=1, mpi=1)




    # --------------------------- INSERT steps functions ----------------------

    def _insertAllSteps(self):
        #XmippProtPreprocessParticles()
        #self.vol_moments = []    #JV
        #self.one_volume = []   #JV

        self._insertFunctionStep('convertInputStep')

        num_batches = self.numBatches.get()

        #self.num_volumes = []

        if self.methodApply == PROB_DENSITY_FUNCT:
            for m in range(1, num_batches + 1):
                self._insertFunctionStep('_processParticles', m)
                self._insertFunctionStep('reconstructStep', m)
                self._insertFunctionStep('_calculatePDF', m)
                #self._insertFunctionStep('calculateStatistics')

        elif self.methodApply == ACC_MOMENTS:
            m = 1
            self._insertFunctionStep('_processParticles', m)
            self._insertFunctionStep('reconstructStep', m)
            self._insertFunctionStep('_calculateMoments', m)
            #self._calculateMoments(m, mean, m2, m3, m4)
            #self._insertFunctionStep('_calculateMoments', m)
            #for m in range(1, num_batches + 1):
            #    print("entro")
            #    self._insertFunctionStep('_processParticles', m)
            #    self._insertFunctionStep('reconstructStep', m)
            #    self.one_volume = self._insertFunctionStep('reconstructStep', m)
            #    print('self.one_volume', self.one_volume)
            #    if m == 1:
            #        print("ENTRO 0")
            #        mean = m2 = m3 = m4 = np.zeros_like(self.one_volume, dtype=float) #np.zeros_like(self.one_volume, dtype=float)
            #        self._calculateMoments(m, mean, m2, m3, m4)
            #        #self._insertFunctionStep('_calculateMoments', m, mean, m2, m3, m4)
            #        print("ENTRO 1")

            #    else:
            #        #n = m - 1
            #        print("ENTRO 2")

            #        #self._insertFunctionStep('_calculateMoments', m, self.vol_moments[0],
            #        #                         self.vol_moments[1], self.vol_moments[2], self.vol_moments[3])
            #        self._calculateMoments(m, self.vol_moments[0],
            #                                 self.vol_moments[1], self.vol_moments[2], self.vol_moments[3])

            #        print("ENTRO 3")






    def _calculateMoments(self, m_index): #, mean, m2, m3, m4):

        #if self.numMom.get() == True:
        num_mom = 4

        print(num_mom)
        #n = np.ones_like(self.one_volume, dtype=float)*m_index - 1
        #n = m_index - 1

        #if m_index == 1:
        self.vol_moments = [np.zeros_like(self.one_volume, dtype=float) for _ in range(1, num_mom + 1)]
        n = mean = m2 = m3 = m4 = np.zeros_like(self.one_volume, dtype=float)  # np.zeros_like(self.one_volume, dtype=float)


        n, mean, m2, m3, m4 = np.array(n), np.array(mean), np.array(m2), np.array(m3), np.array(m4)
        print(mean)

        #mean = np.array(mean)
        print('======================MEAN=================================', mean)

        x = np.array(self.one_volume)
        #x = np.array(x)
        print('======================X=================================', x)
        #n = np.array(n)
        n1 = n.copy()

        print('======================n1======================', n1)
        n += np.ones_like(n, dtype=float)

        # delta = np.subtract(x, mean)
        delta = x - mean
        #delta = np.array(delta)
        print('======================DELTA=================================',delta)

        delta_n = delta / n
        #delta_n = np.array(delta_n)
        print('======================DELTA_N=================================', delta_n)
        #print('n', n, 'delta_n', delta_n)

        delta_n2 = delta_n ** 2
        #delta_n2 = np.array(delta_n2)
        print('======================DELTA_N2=================================', delta_n2)

        term1 = delta * delta_n * n1
        #term1 = np.array(term1)
        print('======================TERM1=================================', term1)

        mean += delta_n
        print('======================MEAN NUEVA=================================', mean)

        #m4 = np.array(m4)
        m4 += term1 * delta_n2 * (n ** 2 - 3 * n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3
        print('======================M4=================================', m4)

        #m3 = np.array(m3)
        m3 += term1 * delta_n * (n - 2) - 3 * delta_n * m2
        print('======================m3=================================', m3)

        #m2 = np.array(m2)
        m2 += term1
        print('======================m2=================================', m2)

        #print('----------------------------')
        if np.all(m2 > 0):
            variance = m2 / n
            skewness = (np.sqrt(n) * m3) / (m2 ** (3 / 2))
            kurtosis = ((n * m4) / (m2 ** 2)) - 3

            self.vol_moments[0] = mean
            self.vol_moments[1] = variance
            self.vol_moments[2] = skewness
            self.vol_moments[3] = kurtosis

        else:
            ##### añadir otro if donde especifico qie si m=1 y M2 es 0 es por ser la primera iteracion
            # si ya lleva varias, entonces poner el mensaje que aparece debajo
            print('M2 is 0, so skewness and kurtosis cannot be calculated as they would be divided by 0')
            self.vol_moments[0] = mean
            self.vol_moments[1] = np.zeros_like(self.one_volume, dtype=float)
            self.vol_moments[2] = np.zeros_like(self.one_volume, dtype=float)
            self.vol_moments[3] = np.zeros_like(self.one_volume, dtype=float)

        print('mean=========================================', self.vol_moments[0])
        print('variance', self.vol_moments[1])
        print('skewness', self.vol_moments[2])
        print('kurtosis', self.vol_moments[3])

        np.save(os.path.join(self._getExtraPath(), "1_mean.npy"), self.vol_moments[0])
        np.save(os.path.join(self._getExtraPath(), "2_variance.npy"), self.vol_moments[1])
        np.save(os.path.join(self._getExtraPath(), "3_skewness.npy"), self.vol_moments[2])
        np.save(os.path.join(self._getExtraPath(), "4_kurtosis.npy"), self.vol_moments[3])
        #NumpyImgHandler.saveMrc(self.vol_moments[0], os.path.join(self._getExtraPath(), "1_mean.mrc"))
        #NumpyImgHandler.saveMrc(self.vol_moments[1], os.path.join(self._getExtraPath(), "2_variance.mrc"))
        #NumpyImgHandler.saveMrc(self.vol_moments[2], os.path.join(self._getExtraPath(), "3_skewness.mrc"))
        #NumpyImgHandler.saveMrc(self.vol_moments[3], os.path.join(self._getExtraPath(), "4_kurtosis.mrc"))


        #return mean
        #else:
        #    print('This protocol cannot be executed')


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


    def _processParticles(self, m_index=1):

        if self.reconstruction == RELION_RECONSTRUCTION:
            print(True)
            #input_star = self._getExtraPath('inputParticles.star')
            #output_star = self._getExtraPath(f'sample_{m_index}.star')
            #self.runJob("xmipp_metadata_utilities", f"-i %s -o %s "
            #                                        f"--operate random_subset {self.numSamples.get()} --mode overwrite " % (input_star, output_star))


            #self.runJob("xmipp_metadata_convert", f"-i {output_xmd} -o inputParticles_{m_index}.star")
            #self.runJob("xmipp_metadata_convert", f"-i {input_star} -o output.xmd")
        #imgSet = self.inputParticles.get()


        else:
            input_xmd = self._getExtraPath('inputParticles.xmd')
            output_xmd = self._getTmpPath(f'sample_{m_index}.xmd')


            self.runJob("xmipp_metadata_utilities", f"-i %s -o %s "
                        f"--operate random_subset {self.numSamples.get()} --mode overwrite "%(input_xmd, output_xmd))



    def reconstructStep(self, m_index=1):

        env = Plugin.getEnviron()

        #self.num_volumes = []
        #volume_name = 'vol_' + str(m_index) + '.mrc'
        volume_name = 'vol_1.mrc'
        imgSet = self.inputParticles.get()
        print(f'VOLUME NAME =========================== {volume_name}')

        if self.reconstruction == RELION_RECONSTRUCTION:

            params_relion = ' --i %s' % self._getExtraPath(f'inputParticles.star') #'inputParticles.star'
            params_relion += ' --o %s' % self._getPath(volume_name) #output_volume
            params_relion += ' --sym %s' % self.symmetryGroup.get()
            params_relion += ' --pad %0.1f' % self.paddingFactorRelion.get()
            params_relion += ' --subset -1 --class -1'

            # Addition of the Sampling rate and the maximum resolution
            params_relion += ' --angpix %0.5f' % imgSet.getSamplingRate()
            params_relion += ' --maxres %0.3f' % self.maxResRelion.get()
            params_relion += ' %s' % self.extraParametersRelion.get()

            self.runJob('relion_reconstruct', params_relion)


        else:

            params = ' -i %s' % self._getTmpPath(f'sample_{m_index}.xmd') #input_particles
            params += ' -o %s' % self._getPath(volume_name) #output_volume
            params += ' --sym %s' % self.symmetryGroup.get()
            params += ' --padding %0.1f %0.1f' % (self.projection.get(), self.volume.get())


            # Addition of the Sampling rate, the maximum resolution and extra parameters (if needed)
            params += ' --sampling %0.5f' % imgSet.getSamplingRate()

            if self.maxRes.get() == -1.0:
                params += ' --max_resolution %0.3f' % 0.5
            else:
                params += ' --max_resolution %0.3f' % (imgSet.getSamplingRate()/self.maxRes.get())

            params += ' %s' % self.extraParameters.get()

            #print(params)
            self.runJob('xmipp_reconstruct_fourier_accel', params, env=env)

        self.one_volume = NumpyImgHandler.loadMrc(os.path.join(self._getPath(), volume_name))

        print(os.path.join(self._getPath(), volume_name))
        print(self.one_volume)

        #return self.one_volume


    def _calculatePDF(self, m_index):

        if m_index == 1:
            num_bins = self.numBins.get()
            min_value = self.minRange.get()
            max_value = self.maxRange.get()

            self.len_interv = (max_value - min_value)/num_bins

            self.rango = []
            for i in range(num_bins + 1):
               self.rango.append(min_value + i * self.len_interv)

            np.save(self._getExtraPath('rango.npy'), np.array(self.rango))

            print(self.rango)
            self.range_volumes = [np.zeros_like(self.one_volume) for _ in range(len(self.rango) - 1)]

        for i in range(len(self.rango) - 1):
            inf_limit = self.rango[i]
            sup_limit = self.rango[i + 1]

            #for volume in self.num_volumes:
            mask = (self.one_volume >= inf_limit) & (self.one_volume < sup_limit)

            self.range_volumes[i][mask] += 1

            NumpyImgHandler.saveMrc(self.range_volumes[i], os.path.join(self._getExtraPath(), f'rangeVol_{i+1}.mrc'))
            #np.save(os.path.join(self._getExtraPath(), f'rangeVol_{i+1}.npy'), self.range_volumes[i])

        #print(self.range_volumes[0][:, :, 200][200])



    def calculateStatistics(self):

        #volumes = self.num_volumes

        x, y, z = 191, 183, 200   # Ejemplo: voxel en la posición (50, 50, 50)

        voxel_values = []
        for i in range(len(self.range_volumes)):
            voxel_values.append(self.range_volumes[i][x ,y, z])

        print("Voxel values:", voxel_values)

        #hist, bin_edges = np.histogram(voxel_values, bins=self.rango)

        plt.figure(figsize=(8, 6))
        plt.bar(self.rango[:-1], voxel_values, width=0.00001, edgecolor="black", align="edge", color="blue",
                alpha=0.7)
        plt.plot(self.rango[:-1], voxel_values, 'o')
        plt.plot(self.rango[:-1], voxel_values)
        plt.xlabel(f"Intensidad del voxel {x, y, z}", fontsize=12)
        plt.ylabel("Frecuencia", fontsize=12)
        plt.title("Histograma de intensidades del voxel", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
        #pass



    def createOutputStep(self):
        #mom_order = self.mom_order

        #m1 = NumpyImgHandler.loadMrc(os.path.join(self._getExtraPath(), mom_order[0]))
        #m1[m1 < 0 ] = 0
        #NumpyImgHandler.saveMrc(m1, os.path.join(self._getPath(),"1_mean.mrc"))


        #m2 = NumpyImgHandler.loadMrc(os.path.join(self._getExtraPath(), mom_order[1]))
        #m2[m2 < 0] = 0

        #m12 = m1*m1
        #m13 = m1*m1*m1
        #m14 = m1 * m1 * m1 * m1

        #variance = (m2 - m12)
        #variance[variance < 0] = 0
        #NumpyImgHandler.saveMrc(variance, os.path.join(self._getPath(), "2_variance.mrc"))


        #m3 = NumpyImgHandler.loadMrc(os.path.join(self._getExtraPath(), mom_order[2]))
        #m3[m3 < 0] = 0
        #variance32 = np.sqrt(variance*variance*variance)
        #skewness = (m3 - 3 * m1 * variance - m13) / (variance32+0.001)
        #NumpyImgHandler.saveMrc(skewness, os.path.join(self._getPath(), "3_skewness.mrc"))


        #del(skewness)
        #m4 = NumpyImgHandler.loadMrc(os.path.join(self._getExtraPath(), mom_order[3]))
        #m4[m4 < 0] = 0
        #kurtosis = (m4 - 4 * m1 * m3 + 6 * m12 * m2 - 3 * m14 )/ (variance*variance+0.001)
        ##kurtosis = np.nan_to_num(kurtosis)
        #NumpyImgHandler.saveMrc(kurtosis, os.path.join(self._getPath(), "4_kurtosis.mrc"))
        pass


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



