from turtledemo.round_dance import stop

from pyworkflow.protocol import TupleParam
from pyworkflow.protocol.params import (PointerParam, FloatParam,
                                        StringParam, BooleanParam,
                                        EnumParam, IntParam)

from pwem.protocols import ProtAnalysis3D
from pwem import ALIGN_PROJ
from scipy.ndimage import maximum

from xmipp3.convert import *
from xmipp3.convert import writeSetOfParticles as writeSetOfParticlesXmipp
from cryomethods.convert import writeSetOfParticles
from cryomethods.functions import NumpyImgHandler
import numpy as np
import os, random
from pwem.constants import NO_INDEX
from cryomethods import Plugin
import mrcfile, starfile, sqlite3
import pandas as pd
from pwem.emlib.metadata import MetaData
import matplotlib.pyplot as plt

PARTICLE_ID = 199
PROB_DENSITY_FUNCT = 0
ACC_MOMENTS = 1
RELION_RECONSTRUCTION = 0
XMIPP_RECONSTRUCTION = 1


class ProtLocPDF_classes(ProtAnalysis3D):
    """
    Given a map and the number of moments, the protocol estimates the local probability map.
    """
    _label = 'locPDF_classes'
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
                       '2. Accumulative moments. Apply the method of moments, where 4 moments '
                       'will be calculated (mean, variance, skewness and kurtosis).\n'
                      )

        # -------------------------------- Pdf ----------------------------------------
        form.addParam('numBins', IntParam, default=10,
                      condition='methodApply==%d' % PROB_DENSITY_FUNCT,
                      label="Number of bins",
                      help='Number of bins.')

        form.addParam('minRange', FloatParam, default=-1.0,
                      condition='methodApply==%d' % PROB_DENSITY_FUNCT,
                      label="Minimum value of the range",
                      help='Minimum value of the range.')

        form.addParam('maxRange', FloatParam, default=1.0,
                      condition='methodApply==%d' % PROB_DENSITY_FUNCT,
                      label="Maximum value of the range",
                      help='Maximum value of the range.')

        # -------------------------------- Moments ----------------------------------------
        #form.addParam('numMom', BooleanParam, default=True,
        #              condition='methodApply==%d' % ACC_MOMENTS,
        #              label="4 moments will be estimated",
        #              help='If selected, 4 moments will be calculated')

        # ----------------------------------Bootstrap---------------------------------
        group = form.addGroup('Bootstrap')
        group.addParam('inputProt', PointerParam,
                       label="Input 2D classes",
                       pointerClass='SetOfClasses2D',
                       help='Select the 2D classification output. '
                             'Particles will be sampled from each class to generate '
                             'bootstrap reconstructions.')

        group.addParam('numBatches', IntParam, default=10,
                             label="Number of reconstructions",
                             help='Number of independent particle subsets to generate. '
                                  'Each subset will be used to compute a separate reconstruction '
                                  'using random sampling (bootstrap).')

        group.addParam('numParticlesPerClass', IntParam, default=5,
                             label="Particles per class",
                             help='Maximum number of particles to sample from each class for each reconstruction. '
                                  'If a class contains fewer particles, all will be included. '
                                  'The total number of particles per reconstruction depends '
                                  'on the number of classes.')

        group.addParam('reconstruction', EnumParam,
                      choices=['Relion reconstruction', 'Xmipp reconstruction'],
                      default=XMIPP_RECONSTRUCTION,
                      label='Reconstruction to be applied', display=EnumParam.DISPLAY_COMBO,
                      help='Select which reconstruction you want to apply\n'
                           '1. Relion reconstruction.\n'
                           '2. Xmipp reconstruction. \n')

        # ----------------------------------Relion reconstruction---------------------------------
        groupRelion = form.addGroup('Relion', condition="reconstruction==%d" % RELION_RECONSTRUCTION)

        ##===================== use GPU ========================
        #groupRelion.addParam('gpu', BooleanParam, default=True,
        #                    label='Do you want to use GPU?')

        #groupRelion.addParam('gpuIds', IntParam, default=0,
        #                    condition='gpu',
        #                    label='GPU Id')

        #groupRelion.addParam('inputParticlesStar', PointerParam,
        #               label="Input particles star",
        #               pointerClass='SetOfParticles',
        #               pointerCondition='hasAlignmentProj',
        #               help='Select input particles from a Star file')

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
                          'reconstruction.')

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
                                                'Xmipp reconstruction.')

        form.addParallelSection(threads=1, mpi=1)


    # --------------------------- INSERT steps functions ----------------------

    def _insertAllSteps(self):

        self._insertFunctionStep('convertInputStep')

        num_batches = self.numBatches.get()

        if self.methodApply == PROB_DENSITY_FUNCT:
            for m in range(1, num_batches + 1):
                self._insertFunctionStep('_processParticles', m)
                self._insertFunctionStep('reconstructStep', m)
                self._insertFunctionStep('_calculatePDF', m)

            self._insertFunctionStep('statistic_volumes')


        elif self.methodApply == ACC_MOMENTS:
            for m in range(1, num_batches + 1):
                self._insertFunctionStep('_processParticles', m)
                #self._insertFunctionStep('reconstructStep', m)
                #self._insertFunctionStep('_calculateMoments', m)
                #self._insertFunctionStep('_calculateFourierMoments', m)


    def convertInputStep(self):
        """ Create the input file in STAR format as expected by Relion.
        If the input particles comes from Relion, just link the file.
        """
        imgSet = self.inputParticles.get()
        imgStar = self._getExtraPath('inputParticles.star')

        # Pass stack file as None to avoid write the images files
        writeSetOfParticles(imgSet, imgStar,
                                    outputDir=self._getExtraPath(), #Tmp
                                    alignType=ALIGN_PROJ)

        imgXmd = self._getExtraPath('inputParticles.xmd')
        writeSetOfParticlesXmipp(imgSet, imgXmd)


    def _processParticles(self, m_index=1):

        prot_classes = self.inputProt.get()
        print(prot_classes)
        print(type(prot_classes))

        inputParticles = prot_classes.getImages()

        print(type(inputParticles))

        #outputParticles = SetOfParticles()
        outputParticles = self._createSetOfParticles()
        outputParticles.copyInfo(inputParticles)
        print('outputParticles', outputParticles, type(outputParticles))

        for cls in prot_classes:
            ids = list(cls.getIdSet())
            print('ids', sorted(ids))

            if not ids:
                continue

            random.seed(42 + cls.getObjId())

            selected_ids = random.sample(ids, min(5, len(ids)))
            print('selected_ids', selected_ids)

            for objId in selected_ids:
                particle = inputParticles[objId]
                print("---- PARTICLE ----\n")
                #print(particle)
                print("Class:", cls.getObjId())
                print("Particle ID:", particle.getObjId())
                print("File:", particle.getFileName())
                print("Index:", particle.getIndex())

                # Sampling rate
                if hasattr(particle, 'getSamplingRate'):
                    print("Sampling:", particle.getSamplingRate())

                # Alineamiento
                if particle.hasTransform():
                    print("Has alignment")

                # CTF
                if particle.hasCTF():
                    print("Has CTF")

                print("------------------\n")

                outputParticles.append(particle.clone())

        self._defineOutputs(outputParticles=outputParticles)
        outputParticles.write()
        print('outputparticles', type(outputParticles))
        #self._defineOutputs(outputParticles=outputParticles)
        outputParticles.close()

        if self.reconstruction == RELION_RECONSTRUCTION:
            # Save new .star file
            output_star = self._getExtraPath(f'sample_{m_index}.star') #cambiar al temporal
            writeSetOfParticles(
                outputParticles,
                output_star,
                outputDir=self._getExtraPath(),
                alignType=ALIGN_PROJ
            )

            print("STAR written:", output_star)

        else:
            # Save new .xmd
            output_xmd = self._getExtraPath(f"sample_{m_index}.xmd") #_getExtraPath  _getTmpPath
            writeSetOfParticlesXmipp(outputParticles, output_xmd)


    def reconstructStep(self, m_index=1):

        env = Plugin.getEnviron()

        volume_name = 'vol_' + str(m_index) + '.mrc'
        #volume_name = 'vol_1.mrc'
        imgSet = self.inputParticles.get()

        self.voxel_size = imgSet.getSamplingRate()
        print(f'VOLUME NAME =========================== {volume_name}')

        if self.reconstruction == RELION_RECONSTRUCTION:

            params_relion = ' --i %s' % self._getTmpPath(f'sample_{m_index}.star') #'inputParticles.star'
            params_relion += ' --o %s' % self._getPath(volume_name) #output_volume
            params_relion += ' --sym %s' % self.symmetryGroup.get()
            params_relion += ' --pad %0.1f' % self.paddingFactorRelion.get()
            #params_relion += ' --subset -1 --class -1'

            # Addition of the Sampling rate and the maximum resolution
            params_relion += ' --angpix %0.5f' % self.voxel_size
            #params_relion += ' --maxres %0.3f' % (1/self.maxResRelion.get())

            if self.maxResRelion.get() == -1.0:
                params_relion += ' --maxres %0.3f' % (2.0 * self.voxel_size)
                #0.5
            else:
                params_relion += ' --maxres %0.3f' % self.maxResRelion.get()
                ##(1/self.maxResRelion.get())

            #if self.gpuIds.get() is not None:
            #    params_relion += ' --gpu %i' % self.gpuIds.get()

            params_relion += ' %s' % self.extraParametersRelion.get()

            print('============parametros de relion ==============', params_relion)

            self.runJob('relion_reconstruct', params_relion, env=env)

        else:

            params = ' -i %s' % self._getTmpPath(f'sample_{m_index}.xmd') #input_particles
            params += ' -o %s' % self._getPath(volume_name) #output_volume
            params += ' --sym %s' % self.symmetryGroup.get()
            params += ' --padding %0.1f %0.1f' % (self.projection.get(), self.volume.get())

            # Addition of the Sampling rate, the maximum resolution and extra parameters (if needed)
            params += ' --sampling %0.5f' % self.voxel_size

            if self.maxRes.get() == -1.0:
                params += ' --max_resolution %0.3f' % (1/(2 * self.voxel_size))
                #0.5
            else:
                params += ' --max_resolution %0.3f' % (self.voxel_size/self.maxRes.get())

            params += ' %s' % self.extraParameters.get()

            #print(params)
            self.runJob('xmipp_reconstruct_fourier_accel', params, env=env)

        print('TAMAÑO DEL VOXEL', self.voxel_size)
        self.one_volume = NumpyImgHandler.loadMrc(os.path.join(self._getPath(), volume_name))

        print(self.one_volume)
        print('tipo de dato de los self.one_volume', self.one_volume.dtype)


    def _calculateMoments(self, m_index):

        num_mom = 4

        if m_index == 1:
            self.vol_moments = [np.zeros_like(self.one_volume, dtype=float) for _ in range(1, num_mom + 1)]
            n = np.zeros_like(self.one_volume, dtype=float)
            mean = np.zeros_like(self.one_volume, dtype=float)
            m2 = np.zeros_like(self.one_volume, dtype=float)
            m3 = np.zeros_like(self.one_volume, dtype=float)
            m4 = np.zeros_like(self.one_volume, dtype=float)

        else:
            mean = np.load(os.path.join(self._getExtraPath(), "1_mean.npy"))
            m2 = np.load(os.path.join(self._getPath(), "m2.npy"))
            m3 = np.load(os.path.join(self._getPath(), "m3.npy"))
            m4 = np.load(os.path.join(self._getPath(), "m4.npy"))
            n = np.ones_like(self.one_volume, dtype=float) * m_index - 1

        # Convert all volumes into array
        n, mean, m2, m3, m4 = np.array(n), np.array(mean), np.array(m2), np.array(m3), np.array(m4)

        #print('======================MEAN=================================', mean)
        x = np.array(self.one_volume)
        #print('======================X=================================', x)
        n1 = n.copy()

        #print('======================n1======================', n1)
        n += np.ones_like(n, dtype=float)

        #print('======================N======================', n)
        delta = x - mean
        #print('======================DELTA=================================',delta)

        delta_n = delta / n
        #print('======================DELTA_N=================================', delta_n)

        delta_n2 = delta_n ** 2
        #print('======================DELTA_N2=================================', delta_n2)

        term1 = delta * delta_n * n1
        #print('======================TERM1=================================', term1)

        mean += delta_n
        #print('======================NEW MEAN =================================', mean)

        m4 += term1 * delta_n2 * (n ** 2 - 3 * n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3
        #print('======================M4=================================', m4)

        m3 += term1 * delta_n * (n - 2) - 3 * delta_n * m2
        #print('======================m3=================================', m3)

        m2 += term1
        #print('======================m2=================================', m2)

        np.save(os.path.join(self._getPath(), "m2.npy"), m2)
        np.save(os.path.join(self._getPath(), "m3.npy"), m3)
        np.save(os.path.join(self._getPath(), "m4.npy"), m4)

        # Calculate moments
        if np.any(m2 > 0):
           variance = m2 / n
           skewness = (np.sqrt(n) * m3) / (m2 ** (3 / 2))
           kurtosis = ((n * m4) / (m2 ** 2)) - 3
           print('KURTOSIS TIEMPO REAL \n', kurtosis)

           self.vol_moments[0] = mean
           self.vol_moments[1] = variance
           self.vol_moments[2] = skewness
           self.vol_moments[3] = kurtosis

        else:
            print('M2 is 0, so skewness and kurtosis cannot be calculated as they would be divided by 0')
            self.vol_moments[0] = mean
            self.vol_moments[1] = np.zeros_like(self.one_volume, dtype=float)
            self.vol_moments[2] = np.zeros_like(self.one_volume, dtype=float)
            self.vol_moments[3] = np.zeros_like(self.one_volume, dtype=float)

        #print('mean', self.vol_moments[0])
        #print('variance', self.vol_moments[1])
        #print('skewness', self.vol_moments[2])
        #print('kurtosis', self.vol_moments[3])

        np.save(os.path.join(self._getExtraPath(), "1_mean.npy"), self.vol_moments[0])

        # Save volumes of the moments for further visualization
        if m_index == self.numBatches.get():

            mrcfile.write(os.path.join(self._getExtraPath(), "1_mean.mrc"), self.vol_moments[0].astype(np.float32),
                          voxel_size=self.voxel_size)
            mrcfile.write(os.path.join(self._getExtraPath(), "2_variance.mrc"), self.vol_moments[1].astype(np.float32),
                          voxel_size=self.voxel_size)
            mrcfile.write(os.path.join(self._getExtraPath(), "3_skewness.mrc"), self.vol_moments[2].astype(np.float32),
                          voxel_size=self.voxel_size)
            mrcfile.write(os.path.join(self._getExtraPath(), "4_kurtosis.mrc"), self.vol_moments[3].astype(np.float32),
                          voxel_size=self.voxel_size)


    def _calculateFourierMoments(self, m_index):

        num_mom = 4

        # FFT de la reconstrucción actual
        vol_fft = np.fft.fftn(self.one_volume)
        vol_fft = vol_fft.astype(np.complex128)
        #vol_fft = np.fft.fftshift(np.fft.fftn(self.one_volume)) #solo si vas a interpretar el espectro visualmente o radialmente
        print(vol_fft)

        if m_index == 1:
            self.vol_moments_fft = [np.zeros_like(vol_fft) for _ in range(num_mom)] #1, num_mom + 1
            n_fft = np.zeros_like(vol_fft, dtype=float)
            mean_fft = np.zeros_like(vol_fft, dtype=np.complex128)
            m2_fft = np.zeros_like(vol_fft, dtype=np.complex128)
            m3_fft = np.zeros_like(vol_fft, dtype=np.complex128)
            m4_fft = np.zeros_like(vol_fft, dtype=np.complex128)

        else:
            mean_fft = np.load(os.path.join(self._getExtraPath(), "1_mean_fft.npy"))
            m2_fft = np.load(os.path.join(self._getPath(), "m2_fft.npy"))
            m3_fft = np.load(os.path.join(self._getPath(), "m3_fft.npy"))
            m4_fft = np.load(os.path.join(self._getPath(), "m4_fft.npy"))
            n_fft = np.load(os.path.join(self._getPath(), "n_fft.npy"))
            #n_fft = np.ones_like(self.one_volume, dtype=float) * m_index - 1

        print('======================MEAN FFT=================================', mean_fft)
        x_fft = vol_fft
        print('======================X FFT=================================', x_fft)

        n_fft = n_fft + 1
        print('======================N FFT======================', n_fft)

        delta_fft = x_fft - mean_fft
        print('======================DELTA FFT=================================',delta_fft)

        delta_n_fft = delta_fft / n_fft
        print('======================DELTA_N FFT=================================', delta_n_fft)

        print('======================DELTA_N2 FFT=================================', np.abs(delta_n_fft) ** 2)


        term1_fft = delta_fft * np.conj(delta_n_fft) * (n_fft - 1)
        print('======================TERM1 FFT=================================', term1_fft)

        mean_fft = mean_fft + delta_n_fft
        print('======================NEW MEAN FFT=================================', mean_fft)


        m4_fft = m4_fft + term1_fft * (np.abs(delta_n_fft) ** 2) * (n_fft ** 2 - 3 * n_fft + 3) \
             + 6 * np.abs(delta_n_fft) ** 2 * m2_fft \
             - 4 * delta_n_fft * m3_fft
        print('======================M4 FFT=================================', m4_fft)


        m3_fft = m3_fft + term1_fft * delta_n_fft * (n_fft - 2) - 3 * delta_n_fft * m2_fft
        print('======================m3 FFT=================================', m3_fft)

        m2_fft = m2_fft + term1_fft
        print('======================m2 FFT=================================', m2_fft)

        np.save(os.path.join(self._getPath(), "m2_fft.npy"), m2_fft)
        np.save(os.path.join(self._getPath(), "m3_fft.npy"), m3_fft)
        np.save(os.path.join(self._getPath(), "m4_fft.npy"), m4_fft)
        np.save(os.path.join(self._getPath(), "n_fft.npy"), n_fft)
        np.save(os.path.join(self._getExtraPath(), "1_mean_fft.npy"), mean_fft)

        variance_fft = m2_fft / n_fft
        print('======================VARIANCE FFT=================================')
        print(variance_fft)

        skewness_fft = (np.sqrt(n_fft) * m3_fft) / (np.abs(m2_fft) ** (3 / 2) + 1e-12)
        print('======================SKEWNESS FFT=================================\n',
              skewness_fft)

        kurtosis_fft = ((n_fft * m4_fft) / (np.abs(m2_fft) ** 2 + 1e-12)) - 3
        print('======================KURTOSIS FFT=================================\n',
              kurtosis_fft)

        self.vol_moments[0] = mean_fft
        self.vol_moments[1] = variance_fft
        self.vol_moments[2] = skewness_fft
        self.vol_moments[3] = kurtosis_fft

        #if m_index == self.numBatches.get():
        #    mrcfile.write(os.path.join(self._getExtraPath(), "1_mean_fft.mrc"),
        #                  np.abs(self.vol_moments[0]).astype(np.float32),
        #                  voxel_size=self.voxel_size)

        #    mrcfile.write(os.path.join(self._getExtraPath(), "2_variance_fft.mrc"),
        #                  np.abs(self.vol_moments[1]).astype(np.float32),
        #                  voxel_size=self.voxel_size)

        #    mrcfile.write(os.path.join(self._getExtraPath(), "3_skewness_fft.mrc"),
        #                  np.abs(self.vol_moments[2]).astype(np.float32),
        #                  voxel_size=self.voxel_size)

        #    mrcfile.write(os.path.join(self._getExtraPath(), "4_kurtosis_fft.mrc"),
        #                  np.abs(self.vol_moments[3]).astype(np.float32),
        #                  voxel_size=self.voxel_size)


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
            self.range_volumes = [np.zeros_like(self.one_volume, dtype=float) for _ in range(len(self.rango) - 1)]

        #else:
        #    self.rango = np.load(self._getExtraPath('rango.npy'))
        #    self.range_volumes = [
        #        NumpyImgHandler.loadMrc(os.path.join(self._getExtraPath(), f'rangeVol_{i + 1}.mrc'))
        #        for i in range(len(self.rango) - 1)]

        for i in range(len(self.rango) - 1):
            inf_limit = self.rango[i]
            sup_limit = self.rango[i + 1]

            mask = (self.one_volume >= inf_limit) & (self.one_volume < sup_limit)
            #print(f'Voxel count in range: {np.count_nonzero(mask)}')

            self.range_volumes[i][mask] += 1
            #print(f'Non-zero count in range_volumes[{i}]: {np.count_nonzero(self.range_volumes[i])}')

            np.save(self._getExtraPath('first_bin.npy'), self.range_volumes[0])
            mrcfile.write(os.path.join(self._getExtraPath(), f'rangeVol_{i+1}.mrc'), self.range_volumes[i].astype(np.float32)
                          ,voxel_size=self.voxel_size, overwrite=True)

            #voxel_distribution = [vol[200, 200, 200] for vol in self.range_volumes]
            #print("Distribución del voxel (200, 200, 200) entre bins:", voxel_distribution)
            #plt.bar(range(len(voxel_distribution)), voxel_distribution)
            #plt.title("Histograma del voxel (200, 200, 200)")
            #plt.xlabel("Bin")
            #plt.ylabel("Frecuencia")
            #plt.show()
            #print('tipo de dato de los self.range_volumes', self.range_volumes[0].dtype)


    def statistic_volumes(self):
        ''''si para relion hay que añadir el valor de 10**-15 en el denominador del calculo de las 
        ponderaciones, pero para xmipp no es necesario porque se realizan correctamente todos los
        calculos, lo ideal es crear una funcion que tenga ese parametro, por ejemplo, epsilon, 
        de tal manera que cuando se haga la reconstr por relion valga 10**-15, pero cuando sea por
        medio de xmipp entonces valga 0 '''''


        self.range_volumes = []
        for i in range(1, self.numBins.get() + 1):
            volume = self._getExtraPath("rangeVol_%s.mrc" % i)
            vol = NumpyImgHandler.loadMrc(volume)
            self.range_volumes.append(NumpyImgHandler.loadMrc(volume).copy())

        bin_centers = np.array([(self.rango[i] + self.rango[i + 1]) / 2.0 for i in range(len(self.rango) - 1)],
                               dtype=np.float32)
        print(f'bin_centers {bin_centers}')
        print(f'bin_centers SHAPE {bin_centers.shape}')

        bin_centers_expanded = bin_centers[:, np.newaxis, np.newaxis, np.newaxis]
        print(f'bin_centers_expanded SHAPE {bin_centers_expanded.shape}')
        print('TIPO DE DATOS DE bin_centers_expanded', type(bin_centers_expanded))
        print('TIPO DE DATOS DE range_volumes', type(self.range_volumes))


        # ------------------------ WEIGHTED MEAN ----------------------------------------
        sum_mean = np.sum(self.range_volumes * bin_centers_expanded, axis=0)
        print(f'weighted_sum {sum_mean}')

        weighted_mean = sum_mean/(np.sum(self.range_volumes, axis=0)) #+ 10**-15
        print(f'valor de media ponderada {weighted_mean}')
        print(f'TAMAÑO de media ponderada {weighted_mean.shape}')

        # ------------------------ MOST PROBABLE BIN ----------------------------------------
        most_probable_freq = np.max(self.range_volumes, axis=0)
        print(f'Valor más probable por voxel:\n {most_probable_freq}'
              f'\n {most_probable_freq.shape}')

        max_index = np.argmax(self.range_volumes, axis=0)
        print(f'VALOR DE LOS INDICES: {max_index}')

        bin_values = np.squeeze(bin_centers_expanded[max_index])
        print(f'DIMENSIONES DE VALORES_BIN {bin_values.shape}')
        print(f'Valor real correspondiente al máximo de frecuencia:{bin_values}')

        # ------------------------ WEIGHTED VARIANCE AND STD ----------------------------------------
        sum_weighted_variance = np.sum(self.range_volumes * (bin_centers_expanded - weighted_mean) ** 2, axis=0)
        print(f'suma de varianza ponderada \n{sum_weighted_variance}')
        weighted_variance = sum_weighted_variance/(np.sum(self.range_volumes, axis=0))# + 10**-15)
        print(f'Varianza ponderada SEPARADA: \n{weighted_variance}')
        print(f'TAMAÑO de varianza ponderada {weighted_variance.shape}')

        #weighted_variance = np.average((bin_centers_expanded - weighted_mean) ** 2, axis=0, weights=self.range_volumes)
        #print(f'Varianza ponderada: \n{weighted_variance}')

        weighted_std = np.sqrt(weighted_variance)
        print(f'desviacion tipica \n{weighted_std}')

        # ------------------------ WEIGHTED SKEWNESS ----------------------------------------
        sum_weighted_skew = np.sum(self.range_volumes * ((bin_centers_expanded - weighted_mean)/weighted_std) ** 3, axis=0)
        weighted_skewness = sum_weighted_skew/(np.sum(self.range_volumes, axis=0)) # + 10**-15)

        #weighted_skewness = np.average(((bin_centers_expanded - weighted_mean) / weighted_std) ** 3, axis=0, weights=self.range_volumes)
        print(f'Skewness ponderado \n{weighted_skewness}')

        # ------------------------   WEIGHTED KURTOSIS ----------------------------------------
        sum_weighted_kurt = np.sum(self.range_volumes * ((bin_centers_expanded - weighted_mean)/weighted_std) ** 4, axis=0)
        weighted_kurtosis = sum_weighted_kurt/(np.sum(self.range_volumes, axis=0)) - 3
        #weighted_kurtosis = sum_weighted_kurt / (np.sum(self.range_volumes, axis=0) + 10 ** -15) - 3

        #weighted_kurtosis = np.average(((bin_centers_expanded - weighted_mean) / weighted_std) ** 4 - 3, axis=0, weights=self.range_volumes)
        print(f'curtosis ponderada \n{weighted_kurtosis}')


        output_weighted_mean = os.path.join(self._getExtraPath(), 'weighted_mean.mrc')
        output_max_freq = os.path.join(self._getExtraPath(), 'most_probable_freq.mrc')
        output_max_bin = os.path.join(self._getExtraPath(), 'most_probable_bin.mrc')
        output_path_std = os.path.join(self._getExtraPath(), 'weighted_std.mrc')
        output_path_skewness = os.path.join(self._getExtraPath(), 'weighted_skewness.mrc')
        output_path_kurtosis = os.path.join(self._getExtraPath(), 'weighted_kurtosis.mrc')


        mrcfile.write(output_weighted_mean, weighted_mean.astype(np.float32), voxel_size=self.voxel_size)
        mrcfile.write(output_max_freq, most_probable_freq.astype(np.float32), voxel_size=self.voxel_size)
        mrcfile.write(output_max_bin, bin_values.astype(np.float32), voxel_size=self.voxel_size)
        mrcfile.write(output_path_std, weighted_std.astype(np.float32), voxel_size=self.voxel_size)
        mrcfile.write(output_path_skewness, weighted_skewness.astype(np.float32), voxel_size=self.voxel_size)
        mrcfile.write(output_path_kurtosis, weighted_kurtosis.astype(np.float32), voxel_size=self.voxel_size)



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
        summary.append(" ")
        return summary

    def _citations(self):
        return ['Vargas2021']



