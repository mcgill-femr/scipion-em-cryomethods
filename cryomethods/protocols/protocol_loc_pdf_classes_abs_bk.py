from pyworkflow.protocol.params import (PointerParam, FloatParam,
                                        StringParam, BooleanParam,
                                        EnumParam, IntParam)

from pwem.protocols import ProtAnalysis3D

from xmipp3.convert import *
from xmipp3.convert import writeSetOfParticles as writeSetOfParticlesXmipp
from cryomethods.convert import writeSetOfParticles
from cryomethods.functions import NumpyImgHandler
import numpy as np
import os, random, shutil, pickle
from pwem.constants import NO_INDEX
from cryomethods import Plugin
import mrcfile
from scipy.ndimage import gaussian_filter

PROB_DENSITY_FUNCT = 0
ACC_MOMENTS = 1
RELION_RECONSTRUCTION = 0
XMIPP_RECONSTRUCTION = 1
REAL_SPACE = 0
FOURIER_SPACE = 1
BOTH = 2
GAUSSIAN_SCIPY = 0
LOWPASS_RELION = 1

class ProtLocPDF_classes_abs(ProtAnalysis3D):
    """
    Given a map and the number of moments, the protocol estimates the local probability map (absolute).
    """
    _label = 'locPDF_classes_abs'
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

        # -------------------------------- Normalization ----------------------------------------
        form.addParam('normalizeVolumes', BooleanParam, default=False,
                      label="Normalize reconstructed volumes",
                      help='If YES, each reconstructed volume will be Z-score normalized '
                           '(mean=0, std=1) before further processing.\n\n'
                           'For PDF: \n   if ON, use a range around [-4, 4]; '
                           '\n   if OFF, set the range according to your original density scale.\n'
                           'For Moments: \n   if ON, moments capture shape variability only; '
                           '\n   if OFF, they also capture global scale differences between reconstructions.')

        # -------------------------------- Pdf ----------------------------------------
        form.addParam('numBins', IntParam, default=10,
                      condition='methodApply==%d' % PROB_DENSITY_FUNCT,
                      label="Number of bins",
                      help='Number of bins.')

        form.addParam('minRange', FloatParam, default=-1.0,
                      condition='methodApply==%d' % PROB_DENSITY_FUNCT,
                      label="Minimum value of the range",
                      help='Minimum value of the range.\n'
                           'If normalization is ON (recommended): use -4.0 to cover ~99.99% of data.\n'
                           'If normalization is OFF: set according to your original density scale.')

        form.addParam('maxRange', FloatParam, default=1.0,
                      condition='methodApply==%d' % PROB_DENSITY_FUNCT,
                      label="Maximum value of the range",
                      help='Maximum value of the range.\n'
                           'If normalization is ON (recommended): use 4.0 to cover ~99.99% of data.\n'
                           'If normalization is OFF: set according to your original density scale.')

        # -------------------------------- Moments ----------------------------------------
        form.addParam('momentDomain', EnumParam,
                      choices = ['Real space', 'Fourier space', 'Both'],
                      default = REAL_SPACE,
                      condition='methodApply==%d' % ACC_MOMENTS,
                      label="Moment calculation domain",
                      display=EnumParam.DISPLAY_COMBO,
                      help='Select where accumulative moments will be calculated:\n' 
                                '1. Real space.\n'
                                '2. Fourier space, but only aplied on |FFT|\n'
                                '3. Both domains')

        # ----------------------------------Filtering---------------------------------
        groupFilter = form.addGroup('Filtering')
        groupFilter.addParam('applyFilter', BooleanParam,
                      default=False,
                      label='Apply filter?',
                      help='If set to Yes, a smoothing filter will be applied '
                           'to each volume after normalization (if enabled) '
                           'and before the statistical moments calculation.')

        groupFilter.addParam('filterMethod', EnumParam,
                             choices=['Gaussian filter (scipy)', 'Low-pass filter (RELION)'],
                             default=GAUSSIAN_SCIPY,
                             condition='applyFilter',
                             display=EnumParam.DISPLAY_COMBO,
                             label='Filtering method',
                             help='Choose the filtering method:\n'
                                  '1. Gaussian filter: real-space Gaussian smoothing '
                                  '(scipy.ndimage.gaussian_filter), sigma expressed '
                                  'in voxels.\n'
                                  '2. Low-pass filter: Fourier-space low-pass filter '
                                  'applied via RELION\'s relion_image_handler, cutoff '
                                  'expressed as a resolution in Angstroms.')

        groupFilter.addParam('gaussianSigma', FloatParam,
                             default=2.0,
                             condition='applyFilter and filterMethod==%d' % GAUSSIAN_SCIPY,
                             label='Gaussian sigma',
                             help='Sigma of the Gaussian kernel, expressed in voxels.')

        groupFilter.addParam('lowpassResolution', FloatParam,
                             default=20,
                             condition='applyFilter and filterMethod==%d' % LOWPASS_RELION,
                             label='Low-pass resolution cutoff (A)',
                             help='Resolution cutoff, in Angstroms, for the low-pass '
                                  'filter applied via relion_image_handler (--lowpass '
                                  'option). Lower values (finer resolution) preserve '
                                  'more detail; higher values (coarser resolution) '
                                  'apply stronger smoothing.')

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

        groupRelion.addParam('relionSymmetryGroup', StringParam, default='c1',
                      label="Symmetry group Relion",
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
        groupXmipp.addParam('xmippSymmetryGroup', StringParam, default='c1',
                       label="Symmetry group Xmipp",
                       help='Enforce symmetry in projections')

        groupXmipp.addParam('maxRes', FloatParam, default=-1,
                       label="Maximum resolution (A)",
                       help='Maximum resolution (in Angstrom) to consider \n'
                            'in Fourier space (default Nyquist).')

        groupXmipp.addParam('usePaddingFactorXmipp', BooleanParam, default=True,
                     label="Padding Factor",
                     help='If selected, you can adjust projection and volume for Xmipp \n'
                          'reconstruction.')

        groupXmipp.addParam('projection', FloatParam, default=2.0,
                       label="Number of Projections",
                       help='Number of projections for Xmipp reconstruction.',
                       condition="usePaddingFactorXmipp")

        groupXmipp.addParam('volume', FloatParam, default=2.0,
                       label="Volume Number",
                       help='Volume number for Xmipp reconstruction.',
                       condition="usePaddingFactorXmipp")

        groupXmipp.addParam('extraParameters', StringParam, default='',
                 label="Extra parameters", help='Extra parameters for \n'
                                                'Xmipp reconstruction.')

        form.addParallelSection(threads=1, mpi=1)


    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):

        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('_prepareParticleClassesStep')

        num_batches = self.numBatches.get()

        if self.methodApply.get() == PROB_DENSITY_FUNCT:
            for m in range(1, num_batches + 1):
                self._insertFunctionStep('_processParticles', m)
                self._insertFunctionStep('reconstructStep', m)
                if self.normalizeVolumes.get():
                    self._insertFunctionStep('_normalizeVolumeStep', m)
                if self.applyFilter.get():
                    if self.filterMethod.get() == GAUSSIAN_SCIPY:
                        self._insertFunctionStep('_gaussianFilterStep', m)
                    else:
                        self._insertFunctionStep('_gaussianFilterRelionStep', m)
                self._insertFunctionStep('_calculatePDF', m)
                self._insertFunctionStep('_removePreviousVolume', m)

            self._insertFunctionStep('statistic_volumes')


        elif self.methodApply.get() == ACC_MOMENTS:
            for m in range(1, num_batches + 1):
                self._insertFunctionStep('_processParticles', m)
                self._insertFunctionStep('reconstructStep', m)
                if self.normalizeVolumes.get():
                    self._insertFunctionStep('_normalizeVolumeStep', m)


                domain = self.momentDomain.get()
                if domain in [REAL_SPACE, BOTH]:
                    if self.applyFilter.get():
                        if self.filterMethod.get() == GAUSSIAN_SCIPY:
                            self._insertFunctionStep('_gaussianFilterStep', m)
                        else:
                            self._insertFunctionStep('_gaussianFilterRelionStep', m)
                    self._insertFunctionStep('_calculateMoments', m)

                if domain in [FOURIER_SPACE, BOTH]:
                    # Not applying Gaussian filtering in Fourier space
                    self._insertFunctionStep('_calculateFourierMoments', m)

                self._insertFunctionStep('_removePreviousVolume', m)


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

    def _loadVolume(self, m_index):
        return NumpyImgHandler.loadMrc(self._getExtraPath(f'vol_{m_index}.mrc'))

    def _getVolumeForProcessing(self, m_index):
        """
        Returns the volume that should be used for moments/PDF calculation,
        taking into account whether a filter step was applied.
        """
        if self.applyFilter.get():
            vol_fn = self._getExtraPath(f'vol_{m_index}_filtered.mrc')
        else:
            vol_fn = self._getExtraPath(f'vol_{m_index}.mrc')
        return NumpyImgHandler.loadMrc(vol_fn)

    def _prepareParticleClassesStep(self):
        prot_classes = self.inputProt.get()

        # 1. COLLECT CLASS INFORMATION
        # =========================================================
        class_info = []
        for cls in prot_classes:
            # returns the IDs of the particles belonging to that class
            particle_ids = set(cls.getIdSet())

            if not particle_ids:
                continue

            class_info.append({
                "class_id": cls.getObjId(),  #identifies the Class2D object
                "ids": particle_ids,
                "size": len(particle_ids)
            })

        # 2. FIND PARTICLES THAT APPEAR IN MULTIPLE CLASSES
        # =========================================================
        # Map each particle ID to the classes it belongs to
        particle_classes = {}
        for info in class_info:
            class_id = info["class_id"]

            # Assign the current class ID to each particle in the class
            for particle_id in info["ids"]:
                if particle_id not in particle_classes:
                    particle_classes[particle_id] = []

                particle_classes[particle_id].append(class_id)

        # 3. FIND OVERLAPPING PARTICLES
        # =========================================================
        # Identify particles assigned to more than one class
        overlapping_particles = {
            particle_id: class_ids
            for particle_id, class_ids in particle_classes.items() if len(class_ids) > 1
        }
        print("Particle/class overlap check")
        print("==============================================")

        print(f"Total unique particles before resolving overlap: {len(particle_classes)}")
        print(f"Particles present in multiple classes: {len(overlapping_particles)}")

        # If a particle belongs to several classes, keep it in the MINORITY class, i.e. the class
        # containing fewer particles
        # In case of a tie, the class with the smallest class ID wins. This makes the decision deterministic
        allowed_ids_by_class = {
            info["class_id"]: set(info["ids"])
            for info in class_info
        }

        if overlapping_particles:
            print("Resolving overlapping particles...")
            print("----------------------------------------------")

            class_sizes = {
                info["class_id"]: info["size"]
                for info in class_info
            }

            for particle_id, class_ids in overlapping_particles.items():
                # Sort by number of particles in the class and class ID. The smallest class wins
                winning_class = min(
                                class_ids,
                                key=lambda class_id: (
                                    class_sizes[class_id],
                                    class_id)
                                )

                print(f"Particle {particle_id}: "
                      f"classes {class_ids} -> "
                      f"keeping in minority class {winning_class} "
                      f"(size={class_sizes[winning_class]})")

                # Remove the particle from every class except the winning class
                for class_id in class_ids:
                    if class_id != winning_class:
                        allowed_ids_by_class[class_id].discard(particle_id)

            print("----------------------------------------------")
            print(f"Resolved {len(overlapping_particles)} overlapping particles")
            print("----------------------------------------------")

        else:
            print("No particle overlap detected")

        # 4. FINAL VALIDATION
        # After resolving overlaps, no particle should belong to more than one class.
        # =========================================================
        particle_occurrences = {}

        for class_id, particle_ids in allowed_ids_by_class.items():

            for particle_id in particle_ids:
                particle_occurrences.setdefault(
                    particle_id,
                    []
                ).append(class_id)

        remaining_overlaps = {
            particle_id: class_ids
            for particle_id, class_ids in particle_occurrences.items() if len(class_ids) > 1
        }

        if remaining_overlaps:
            raise RuntimeError(
                f"Overlap still exists after resolving classes: {remaining_overlaps}")

        print("----------------------------------------------")
        print("Overlap resolution completed successfully")
        print("----------------------------------------------")

        # 5. SAVE RESULT
        # =========================================================
        output_file_particles = self._getExtraPath('allowed_ids_by_class.pkl')

        with open(output_file_particles, 'wb') as f:
            pickle.dump(allowed_ids_by_class, f)

        print("Saved resolved particle assignment to:", output_file_particles )
        print("==============================================")


    def _processParticles(self, m_index=1):

        assignment_file = self._getExtraPath('allowed_ids_by_class.pkl')
        if not os.path.exists(assignment_file):
            raise RuntimeError(f"Particle assignment file not found: {assignment_file}")

        with open(assignment_file, 'rb') as f:
            allowed_ids_by_class = pickle.load(f)

        prot_classes = self.inputProt.get()
        inputParticles = prot_classes.getImages()
        outputParticles = self._createSetOfParticles()
        outputParticles.copyInfo(inputParticles)

        # IMPORTANT:
        # We sample from allowed_ids_by_class, NOT directly from cls.getIdSet().
        # Therefore particles that were found in several classes are only available in their minority class.
        used_ids = set()
        duplicates_skipped = 0

        for class_id, ids_set in allowed_ids_by_class.items():
            ids = list(ids_set)

            if not ids:
                continue

            # Deterministic seed for reproducibility
            # 42: arbitrary constant
            # class_id: ensures different classes get different seeds
            # m_index: ensures different bootstrap iterations get different samples from the same
            #          class; without it, every iteration would select the same particles, defeating
            #          the purpose of bootstrapping
            random.seed(42 + class_id + m_index)

            # no replacement
            selected_ids = random.sample(ids, min(self.numParticlesPerClass.get(), len(ids)))
            #print('selected_ids', selected_ids)
            #print("------------------\n")

            for particle_id in selected_ids:
                # Even after resolving class overlaps, make sure the same particle can NEVER be added twice to
                # the reconstruction.
                if particle_id in used_ids:
                    duplicates_skipped += 1

                    print(f"WARNING: Particle {particle_id} was already selected in bootstrap {m_index}. "
                          f"Skipping duplicate")
                    continue

                used_ids.add(particle_id)
                particle = inputParticles[particle_id]
                outputParticles.append(particle.clone())

        num_selected = len(used_ids)
        num_output = outputParticles.getSize()

        print("==============================================")
        print(f"BOOTSTRAP {m_index} CHECK")
        print("==============================================")

        print(f"Particles selected: {num_selected}")
        print(f"Duplicates skipped during bootstrap: {duplicates_skipped}")
        print(f"Particles in output: {num_output}")

        # This should always be true because used_ids is a set and duplicates are filtered before append()
        if num_selected != num_output:
            raise RuntimeError(
                f"Internal consistency error in bootstrap "
                f"{m_index}: "
                f"{num_selected} selected particles but "
                f"{num_output} particles in output."
            )

        print("==============================================")

        outputParticles.write()
        outputParticles.close()

        if self.reconstruction == RELION_RECONSTRUCTION:
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
            output_xmd = self._getTmpPath(f"sample_{m_index}.xmd") #_getExtraPath
            writeSetOfParticlesXmipp(outputParticles, output_xmd)
            print("XMD written:", output_xmd)


    def reconstructStep(self, m_index=1):
        env = Plugin.getEnviron()

        volume_name = 'vol_' + str(m_index) + '.mrc'
        imgSet = self.inputParticles.get()

        self.voxel_size = imgSet.getSamplingRate()
        print(f'VOLUME NAME =========================== {volume_name}')

        if self.reconstruction == RELION_RECONSTRUCTION:

            params_relion = ' --i %s' % self._getExtraPath(f'sample_{m_index}.star') #'inputParticles.star'
            params_relion += ' --o %s' % self._getExtraPath(volume_name) #_getTmpPath   #_getPath
            params_relion += ' --sym %s' % self.relionSymmetryGroup.get()
            params_relion += ' --pad %0.1f' % self.paddingFactorRelion.get()
            #params_relion += ' --subset -1 --class -1'

            # Addition of the Sampling rate and the maximum resolution
            params_relion += ' --angpix %0.5f' % self.voxel_size
            #params_relion += ' --maxres %0.3f' % (1/self.maxResRelion.get())

            # resolution (A)
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
            params += ' -o %s' % self._getTmpPath(volume_name) #_getTmpPath   #_getPath
            params += ' --sym %s' % self.xmippSymmetryGroup.get()
            params += ' --padding %0.1f %0.1f' % (self.projection.get(), self.volume.get())

            # Addition of the Sampling rate, the maximum resolution and extra parameters (if needed)
            params += ' --sampling %0.5f' % self.voxel_size

            # normalized frequency (1/A)
            if self.maxRes.get() == -1.0:
                # REVISARRRR
                params += ' --max_resolution %0.3f' % 0.5
                #params += ' --max_resolution %0.3f' % (1/(2 * self.voxel_size)) incorrecto
                #0.5
            else:
                params += ' --max_resolution %0.3f' % (self.voxel_size/self.maxRes.get())

            params += ' %s' % self.extraParameters.get()

            #print(params)
            self.runJob('xmipp_reconstruct_fourier_accel', params, env=env)

        print('TAMAÑO DEL VOXEL', self.voxel_size)
        #self.one_volume = NumpyImgHandler.loadMrc(os.path.join(self._getTmpPath(), volume_name))
        # _getTmpPath  #_getPath

        #print(self.one_volume)
        #print('tipo de dato de los self.one_volume', self.one_volume.dtype)
        #self.one_volume = None


    def _removePreviousVolume(self, m_index):

        files = [
            self._getTmpPath(f'vol_{m_index}.mrc'),
            self._getTmpPath(f'sample_{m_index}.xmd'),
            #self._getTmpPath(f'sample_{m_index}.star')
            #self._getExtraPath(f'vol_{m_index}_filtered.mrc')
        ]

        for fn in files:
            if os.path.exists(fn):
                os.remove(fn)
                print(f"Deleted: {fn}")


    def _backupMaps(self, vol_fn, output_dir, output_name):
        os.makedirs(output_dir, exist_ok=True)
        backup_fn = os.path.join(output_dir, output_name)

        if not os.path.exists(backup_fn):  # only the first time
            shutil.copy2(vol_fn, backup_fn)
            print(f"Backup saved: {backup_fn}")

        return backup_fn


    def _normalizeVolumeStep(self, m_index):

        vol_fn = self._getExtraPath(f'vol_{m_index}.mrc')
        #vol = NumpyImgHandler.loadMrc(vol_fn)
        vol = np.array(self._loadVolume(m_index))

        # Original volume backup
        #original_dir = self._getExtraPath('originals')
        #backup_fn = self._backupMaps(vol_fn,
        #                             original_dir,
        #                             f'vol_{m_index}_orig.mrc')

        mean = np.mean(vol)
        std = np.std(vol)
        eps = 1e-12

        if std > eps:
            vol_norm = (vol - mean)/std
        else:
            print(f'WARNING: std ≈ 0 in volume {m_index}, it is not normalized')
            vol_norm = vol


        mrcfile.write(vol_fn,
                      vol_norm.astype(np.float32),
                      overwrite=True,
                      voxel_size=self.voxel_size
        )
        print(f"Normalization batch {m_index}: mean={mean:.4f}, std={std:.4f}")


    def _gaussianFilterStep(self, m_index):
        # If normalization is enabled, the Gaussian filter is applied on the normalized volume because _normalizeVolumeStep runs first.
        #vol_fn_inicial = self._getExtraPath(f'vol_{m_index}.mrc')
        #vol = NumpyImgHandler.loadMrc(vol_fn_inicial)
        vol_fn = self._getExtraPath(f'vol_{m_index}_filtered.mrc')
        vol = np.array(self._loadVolume(m_index))

        sigma = self.gaussianSigma.get()

        vol_filtered = gaussian_filter(
            vol,
            sigma=sigma
        )

        mrcfile.write(
            vol_fn,
            vol_filtered.astype(np.float32),
            overwrite=True,
            voxel_size=self.voxel_size
        )

        print(f'Gaussian filter applied (batch={m_index}, sigma={sigma})')


    def _gaussianFilterRelionStep(self, m_index):
        # If normalization is enabled, the filter is applied on the normalized volume because _normalizeVolumeStep runs first.
        vol_fn = self._getExtraPath(f'vol_{m_index}.mrc')
        vol_filtered = self._getExtraPath(f'vol_{m_index}_filtered.mrc')

        # Cutoff resolution for the low-pass filter, in Angstroms
        lowpass_res = self.lowpassResolution.get()

        args = (
            f'--i {vol_fn} '
            f'--o {vol_filtered} '
            f'--lowpass {lowpass_res} '
            f'--angpix {self.voxel_size} '
        )

        self.runJob('relion_image_handler', args)
        print(f'RELION lowpass filter applied (batch={m_index}, lowpass={lowpass_res} A)')


    def _calculateMoments(self, m_index):
        """
        Calculate accumulative moments in real space using voxel-wise online update.
        The output is voxel-wise moment volumes:
          - mean
          - variance
          - skewness
          - kurtosis
        """

        eps = 1e-12

        # ------------------------------------------------------------
        # 1. Input reconstructed volume
        # ------------------------------------------------------------
        x = np.asarray(self._getVolumeForProcessing(m_index), dtype=np.float64)
        #x = np.asarray(self.one_volume, dtype=np.float64)
        #x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        print("VOL MIN:", np.nanmin(x))
        print("VOL MAX:", np.nanmax(x))
        print("VOL MEAN:", np.nanmean(x))
        print("NaNs in input volume:", np.sum(np.isnan(x)))
        print("Infs in input volume:", np.sum(np.isinf(x)))

        # ------------------------------------------------------------
        # 2. Initialize or load accumulators
        # ------------------------------------------------------------
        if m_index == 1:
            self.n_real = np.zeros_like(x, dtype=np.float64)
            self.mean_real = np.zeros_like(x, dtype=np.float64)
            self.M2_real = np.zeros_like(x, dtype=np.float64)
            self.M3_real = np.zeros_like(x, dtype=np.float64)
            self.M4_real = np.zeros_like(x, dtype=np.float64)
        else:
            self.n_real = np.load(os.path.join(self._getPath(), "n_real.npy"))
            self.mean_real = np.load(os.path.join(self._getPath(), "mean_real.npy"))
            self.M2_real = np.load(os.path.join(self._getPath(), "M2_real.npy"))
            self.M3_real = np.load(os.path.join(self._getPath(), "M3_real.npy"))
            self.M4_real = np.load(os.path.join(self._getPath(), "M4_real.npy"))

        # Sanity check
        if not (self.n_real.shape == self.mean_real.shape == self.M2_real.shape ==
                self.M3_real.shape == self.M4_real.shape):
            raise ValueError("Accumulator shapes do not match the volume shape.")


        # ------------------------------------------------------------
        # 3. Online update of voxel-wise moments Pébay/Welford
        # ------------------------------------------------------------
        n_old = self.n_real #n1
        n_new = n_old + 1.0

        delta = x - self.mean_real
        delta_n = delta / n_new
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n_old

        M2_old = self.M2_real.copy()
        M3_old = self.M3_real.copy()
        M4_old = self.M4_real.copy()

        self.mean_real += delta_n

        self.M4_real = (
                M4_old
                + term1 * delta_n2 * (n_new ** 2.0 - 3.0 * n_new + 3.0)
                + 6.0 * delta_n2 * M2_old
                - 4.0 * delta_n * M3_old
        )
        print('======================m4=================================', self.M4_real)

        self.M3_real = (
                M3_old
                + term1 * delta_n * (n_new - 2.0)
                - 3.0 * delta_n * M2_old
        )
        print('======================m3=================================', self.M3_real)

        self.M2_real = M2_old + term1
        print('======================m2=================================', self.M2_real)

        self.n_real = n_new

        print("M2 min/max:", np.min(self.M2_real), np.max(self.M2_real))
        print("M3 min/max:", np.min(self.M3_real), np.max(self.M3_real))
        print("M4 min/max:", np.min(self.M4_real), np.max(self.M4_real))

        ratio = (self.n_real * self.M4_real) / (self.M2_real ** 2)

        print("ratio min:", np.min(ratio))
        print("ratio max:", np.max(ratio))
        print("ratio std:", np.std(ratio))

        # ------------------------------------------------------------
        # 4. Derived moments (stable computation)
        # ------------------------------------------------------------
        variance_real = np.divide(
            self.M2_real,
            np.maximum(self.n_real, eps),
            out=np.zeros_like(self.M2_real, dtype=np.float64),
            where=self.n_real > 0
        )
        print('======================VARIANZA=================================', variance_real)

        std_real = np.sqrt(variance_real)

        skewness_real = np.zeros_like(self.M2_real, dtype=np.float64)
        kurtosis_real = np.zeros_like(self.M2_real, dtype=np.float64)

        mask = self.M2_real > eps

        with np.errstate(divide='ignore', invalid='ignore'):
            skewness_real[mask] = (np.sqrt(self.n_real[mask]) * self.M3_real[mask]) / (self.M2_real[mask] ** 1.5)

            # voxels with variance <= eps are excluded from the kurtosis computation and remain equal to 0
            kurtosis_real[mask] = ((self.n_real[mask] * self.M4_real[mask]) / (self.M2_real[mask] ** 2)) - 3.0

        print('======================SKEWNESS=================================', skewness_real)
        print('======================KURTOSIS=================================', kurtosis_real)

        # After masking low-variance voxels, replace any residual NaN/Inf values caused by numerical precision issues
        skewness_real = np.nan_to_num(skewness_real, nan=0.0, posinf=0.0, neginf=0.0)
        kurtosis_real = np.nan_to_num(kurtosis_real, nan=0.0, posinf=0.0, neginf=0.0)

        ##### INICIO COMPROBACIONES
        if getattr(self, 'debug', False):
            print("Kurtosis min:", np.min(kurtosis_real))
            print("Kurtosis max:", np.max(kurtosis_real))
            print("Kurtosis std:", np.std(kurtosis_real))
            print("Percentiles de kurtosis:")
            print(np.percentile(kurtosis_real[np.isfinite(kurtosis_real)],
                                [0, 1, 5, 25, 50, 75, 95, 99, 100]))

            center = kurtosis_real.shape[0] // 2

            print(kurtosis_real[center])
            print("Central slice min:", np.min(kurtosis_real[center]))
            print("Central slice max:", np.max(kurtosis_real[center]))

            high_var_mask = self.M2_real > np.percentile(self.M2_real, 95)
            print("Kurtosis in high variance region:", np.mean(kurtosis_real[high_var_mask]), np.std(kurtosis_real[high_var_mask]))
            ##### FIN COMPROBACIONES

        print('==============SKEWNESS QUITANDO ARTEFACTOS=================', skewness_real)
        print('==============KURTOSIS QUITANDO ARTEFACTOS=================', kurtosis_real)

        # ------------------------------------------------------------
        # 5. Save accumulators for next batch
        # ------------------------------------------------------------
        np.save(os.path.join(self._getPath(), "n_real.npy"), self.n_real)
        np.save(os.path.join(self._getPath(), "mean_real.npy"), self.mean_real)
        np.save(os.path.join(self._getPath(), "M2_real.npy"), self.M2_real)
        np.save(os.path.join(self._getPath(), "M3_real.npy"), self.M3_real)
        np.save(os.path.join(self._getPath(), "M4_real.npy"), self.M4_real)

        # Optional debug
        print("NaNs mean:", np.sum(np.isnan(self.mean_real)))
        print("NaNs variance:", np.sum(np.isnan(variance_real)))
        print("NaNs skewness:", np.sum(np.isnan(skewness_real)))
        print("NaNs kurtosis:", np.sum(np.isnan(kurtosis_real)))
        print("-------------------------------------------------")
        print("Infs skewness:", np.sum(np.isinf(skewness_real)))
        print("Infs kurtosis:", np.sum(np.isinf(kurtosis_real)))
        print("M2 <= eps:", np.sum(self.M2_real <= eps))
        print('Percentiles', np.percentile(self.M2_real,[0, 1, 5, 25, 50, 75, 95, 99, 100]))
        num_small = np.sum(self.M2_real <= eps)
        total = self.M2_real.size

        print(f"{100 * num_small / total:.2f}% voxels have M2 <= eps")

        # ------------------------------------------------------------
        # 6. Save final MRC volumes only at last batch
        # ------------------------------------------------------------
        if m_index == self.numBatches.get():
            mrcfile.write(
                os.path.join(self._getExtraPath(), "1_mean.mrc"),
                self.mean_real.astype(np.float32),
                voxel_size=self.voxel_size
            )
            mrcfile.write(
                os.path.join(self._getExtraPath(), "2_variance.mrc"),
                variance_real.astype(np.float32),
                voxel_size=self.voxel_size
            )
            mrcfile.write(
                os.path.join(self._getExtraPath(), "2_std.mrc"),
                std_real.astype(np.float32),
                voxel_size=self.voxel_size
            )
            mrcfile.write(
                os.path.join(self._getExtraPath(), "3_skewness.mrc"),
                skewness_real.astype(np.float32),
                voxel_size=self.voxel_size
            )
            mrcfile.write(
                os.path.join(self._getExtraPath(), "4_kurtosis.mrc"),
                kurtosis_real.astype(np.float32),
                voxel_size=self.voxel_size
            )

        print("Finished _calculateMoments for batch:", m_index)


    def _calculateFourierMoments(self, m_index):
        """
        Calculate accumulative moments in Fourier space using ONLY the magnitude |FFT|.
        The output is voxel-wise moment volumes:
          - mean
          - variance
          - skewness
          - kurtosis
        """

        eps = 1e-12

        # ------------------------------------------------------------
        # 1. FFT of the reconstructed volume
        # ------------------------------------------------------------
        vol_real = np.asarray(self._loadVolume(m_index), dtype=np.float64)
        #vol_real = np.asarray(self.one_volume, dtype=np.float64)
        #vol_real = np.nan_to_num(vol_real, nan=0.0, posinf=0.0, neginf=0.0)

        vol_fft = np.fft.fftn(vol_real).astype(np.complex128)

        # Work ONLY with magnitude
        mag_fft = np.abs(vol_fft).astype(np.float64)

        print("MAG FFT MIN:", np.min(mag_fft))
        print("MAG FFT MAX:", np.max(mag_fft))
        print("MAG FFT MEAN:", np.mean(mag_fft))

        # ------------------------------------------------------------
        # 2. Initialize accumulators on first batch
        # ------------------------------------------------------------
        if m_index == 1:
            self.n_mag_fft = np.zeros_like(mag_fft, dtype=np.float64)
            self.mean_mag_fft = np.zeros_like(mag_fft, dtype=np.float64)
            self.M2_mag_fft = np.zeros_like(mag_fft, dtype=np.float64)
            self.M3_mag_fft = np.zeros_like(mag_fft, dtype=np.float64)
            self.M4_mag_fft = np.zeros_like(mag_fft, dtype=np.float64)

        else:
            self.n_mag_fft = np.load(os.path.join(self._getPath(), "n_mag_fft.npy"))
            self.mean_mag_fft = np.load(os.path.join(self._getPath(), "mean_mag_fft.npy"))
            self.M2_mag_fft = np.load(os.path.join(self._getPath(), "M2_mag_fft.npy"))
            self.M3_mag_fft = np.load(os.path.join(self._getPath(), "M3_mag_fft.npy"))
            self.M4_mag_fft = np.load(os.path.join(self._getPath(), "M4_mag_fft.npy"))

        # Sanity check
        if not (self.n_mag_fft.shape == mag_fft.shape ==
                self.mean_mag_fft.shape == self.M2_mag_fft.shape ==
                self.M3_mag_fft.shape == self.M4_mag_fft.shape):
            raise ValueError("Accumulator shapes do not match the Fourier volume shape.")

        # ------------------------------------------------------------
        # 3. Online update of voxel-wise moments for |FFT|
        # ------------------------------------------------------------
        x = mag_fft

        n_old = self.n_mag_fft
        n_new = n_old + 1.0

        delta = x - self.mean_mag_fft
        delta_n = delta / n_new
        delta_n2 = delta_n * delta_n

        term1 = delta * delta_n * n_old

        #mean_old = self.mean_mag_fft.copy()
        M2_old = self.M2_mag_fft.copy()
        M3_old = self.M3_mag_fft.copy()
        M4_old = self.M4_mag_fft.copy()

        # Update mean
        self.mean_mag_fft = self.mean_mag_fft + delta_n

        # Update higher moments
        self.M4_mag_fft = (
                M4_old
                + term1 * delta_n2 * (n_new ** 2 - 3.0 * n_new + 3.0)
                + 6.0 * delta_n2 * M2_old
                - 4.0 * delta_n * M3_old
        )

        self.M3_mag_fft = (
                M3_old
                + term1 * delta_n * (n_new - 2.0)
                - 3.0 * delta_n * M2_old
        )

        self.M2_mag_fft = M2_old + term1

        self.n_mag_fft = n_new

        # ------------------------------------------------------------
        # 4. Derived voxel-wise moments
        # ------------------------------------------------------------
        #variance_fft = self.M2_mag_fft / np.maximum(self.n_mag_fft, eps)
        #den_skew = np.power(np.maximum(self.M2_mag_fft, eps), 1.5)
        #den_kurt = np.power(np.maximum(self.M2_mag_fft, eps), 2.0)
        #skewness_fft = (np.sqrt(self.n_mag_fft) * self.M3_mag_fft) / (den_skew + eps)
        #kurtosis_fft = (self.n_mag_fft * self.M4_mag_fft) / (den_kurt + eps) - 3.0

        variance_fft = np.divide(
            self.M2_mag_fft,
            np.maximum(self.n_mag_fft, eps),
            out=np.zeros_like(self.M2_mag_fft),
            where=self.n_mag_fft > 0
        )

        print('======================VARIANZA=================================', variance_fft)

        std_fft = np.sqrt(variance_fft)

        skewness_fft = np.zeros_like(self.M2_mag_fft)
        kurtosis_fft = np.zeros_like(self.M2_mag_fft)

        # Statistical filtering
        # compute skewness/kurtosis only in voxels with non-negligible variance.
        mask = self.M2_mag_fft > eps

        with np.errstate(divide='ignore', invalid='ignore'):
            skewness_fft[mask] = (np.sqrt(self.n_mag_fft[mask]) * self.M3_mag_fft[mask])/(self.M2_mag_fft[mask] ** 1.5)

            # voxels with variance <= eps are excluded from the kurtosis computation and remain equal to 0
            kurtosis_fft[mask] = ((self.n_mag_fft[mask] * self.M4_mag_fft[mask]) / (self.M2_mag_fft[mask] ** 2)) - 3.0

        print('======================SKEWNESS======================', skewness_fft)
        print('======================KURTOSIS======================', kurtosis_fft)

        # Numerical safety
        # after masking low-variance voxels, replace any residual NaN/Inf values caused by numerical precision issues
        skewness_fft = np.nan_to_num(skewness_fft, nan=0.0, posinf=0.0, neginf=0.0)
        kurtosis_fft = np.nan_to_num(kurtosis_fft, nan=0.0, posinf=0.0, neginf=0.0)

        ##### INICIO COMPROBACIONES
        print("Kurtosis fft min:", np.min(kurtosis_fft))
        print("Kurtosis fft max:", np.max(kurtosis_fft))
        print("Kurtosis fft std:", np.std(kurtosis_fft))
        print("Percentiles de kurtosis fft:")
        print(np.percentile(kurtosis_fft[np.isfinite(kurtosis_fft)],
                            [0, 1, 5, 25, 50, 75, 95, 99, 100]))

        center = kurtosis_fft.shape[0] // 2

        print(kurtosis_fft[center])
        print("Central slice min fft:", np.min(kurtosis_fft[center]))
        print("Central slice max fft:", np.max(kurtosis_fft[center]))

        high_var_mask = self.M2_mag_fft > np.percentile(self.M2_mag_fft, 95)
        print("Kurtosis in high variance region fft:",
                np.mean(kurtosis_fft[high_var_mask]),
                np.std(kurtosis_fft[high_var_mask]))
        ##### FIN COMPROBACIONES

        print('============SKEWNESS QUITANDO ARTEFACTOS FFT================', skewness_fft)
        print('============KURTOSIS QUITANDO ARTEFACTOS FFT================', kurtosis_fft)

        # ------------------------------------------------------------
        # 5. Save accumulators for next bootstrap batch
        # ------------------------------------------------------------
        np.save(os.path.join(self._getPath(), "n_mag_fft.npy"), self.n_mag_fft)
        np.save(os.path.join(self._getPath(), "mean_mag_fft.npy"), self.mean_mag_fft)
        np.save(os.path.join(self._getPath(), "M2_mag_fft.npy"), self.M2_mag_fft)
        np.save(os.path.join(self._getPath(), "M3_mag_fft.npy"), self.M3_mag_fft)
        np.save(os.path.join(self._getPath(), "M4_mag_fft.npy"), self.M4_mag_fft)

        # ------------------------------------------------------------
        # 6. Save MRC volumes only on the last batch
        # ------------------------------------------------------------
        if m_index == self.numBatches.get():
            # Center spectrum only for visualization
            vis_mean = np.fft.fftshift(self.mean_mag_fft)
            vis_var = np.fft.fftshift(variance_fft)
            vis_std = np.fft.fftshift(std_fft)
            vis_skew = np.fft.fftshift(skewness_fft)
            vis_kurt = np.fft.fftshift(kurtosis_fft)

            mrcfile.write(
                os.path.join(self._getExtraPath(), "1_mean_fft_mag.mrc"),
                vis_mean.astype(np.float32),
                voxel_size=self.voxel_size
            )

            mrcfile.write(
                os.path.join(self._getExtraPath(), "2_variance_fft_mag.mrc"),
                vis_var.astype(np.float32),
                voxel_size=self.voxel_size
            )

            mrcfile.write(
                os.path.join(self._getExtraPath(), "2_std_fft_mag.mrc"),
                vis_std.astype(np.float32),
                voxel_size=self.voxel_size
            )

            mrcfile.write(
                os.path.join(self._getExtraPath(), "3_skewness_fft_mag.mrc"),
                vis_skew.astype(np.float32),
                voxel_size=self.voxel_size
            )

            mrcfile.write(
                os.path.join(self._getExtraPath(), "4_kurtosis_fft_mag.mrc"),
                vis_kurt.astype(np.float32),
                voxel_size=self.voxel_size
            )

        print("Finished _calculateFourierMoments for batch:", m_index)


    def _calculatePDF(self, m_index): ##REVISAR

        #vol = NumpyImgHandler.loadMrc(self._getExtraPath(f'vol_{m_index}.mrc'))
        vol = self._getVolumeForProcessing(m_index)

        if m_index == 1:
            num_bins = self.numBins.get()
            min_value = self.minRange.get()
            max_value = self.maxRange.get()

            self.rango = np.linspace(min_value, max_value, num_bins + 1)

            np.save(self._getExtraPath('rango.npy'), self.rango)
            print(f'Min real: {min_value}, Max real: {max_value}')
            print(self.rango)
            self.range_volumes = [np.zeros_like(vol, dtype=float) for _ in range(len(self.rango) - 1)]
            #self.range_volumes = [np.zeros_like(self.one_volume, dtype=float) for _ in range(len(self.rango) - 1)]

        #else:
        #    self.rango = np.load(self._getExtraPath('rango.npy'))
        #    self.range_volumes = [
        #        NumpyImgHandler.loadMrc(os.path.join(self._getExtraPath(), f'rangeVol_{i + 1}.mrc'))
        #        for i in range(len(self.rango) - 1)]

        # COMPROBACIONES
        outside_mask = ((vol < self.rango[0]) | (vol > self.rango[-1]))
                       #(self.one_volume < self.rango[0]) | (self.one_volume > self.rango[-1])

        print("Outside voxels:", np.count_nonzero(outside_mask))
        print("Outside fraction:",
                np.count_nonzero(outside_mask) /
                vol.size
                #self.one_volume.size
        )


        for i in range(len(self.rango) - 1):
            inf_limit = self.rango[i]
            sup_limit = self.rango[i + 1]

            if i == len(self.rango) - 2:    #last bin
                mask = (vol >= inf_limit) & (vol <= sup_limit)
                #mask = (self.one_volume >= inf_limit) & (self.one_volume <= sup_limit)
            else:
                mask = (vol >= inf_limit) & (vol < sup_limit)
                #mask = (self.one_volume >= inf_limit) & (self.one_volume < sup_limit)

            ##mask = (self.one_volume >= inf_limit) & (self.one_volume < sup_limit)
            ##print(f'Voxel count in range: {np.count_nonzero(mask)}')

            self.range_volumes[i][mask] += 1
            #print(f'Non-zero count in range_volumes[{i}]: {np.count_nonzero(self.range_volumes[i])}')
            mrcfile.write(os.path.join(self._getExtraPath(), f'rangeVol_{i+1}.mrc'), self.range_volumes[i].astype(np.float32)
                          ,voxel_size=self.voxel_size, overwrite=True)

        np.save(self._getExtraPath('first_bin.npy'), self.range_volumes[0])


    #def _calculatePDF(self, m_index): ##RANGO INFLUENCIADO POR PRIMER VOLUMEN

    #    if m_index == 1:
    #        num_bins = self.numBins.get()
    #        min_value = self.minRange.get()
    #        max_value = self.maxRange.get()

    #        vol_min = np.min(self.one_volume)
    #        vol_max = np.max(self.one_volume)
    #        print(f'VOL_MIN {vol_min}, VOL_MAX {vol_max}')

    #        if min_value > vol_min or max_value < vol_max:
    #            print(f"WARNING: rango [{min_value}, {max_value}] is outside range "
    #                  f"[{vol_min}, {vol_max}]")
    #            min_value = min(min_value, vol_min)
    #            max_value = max(max_value, vol_max)

    #        self.rango = np.linspace(min_value, max_value, num_bins + 1)

    #        np.save(self._getExtraPath('rango.npy'), np.array(self.rango))
    #        print(f'Min real: {min_value}, Max real: {max_value}')
    #        print(self.rango)
    #        self.range_volumes = [np.zeros_like(self.one_volume, dtype=float) for _ in range(len(self.rango) - 1)]

    #    for i in range(len(self.rango) - 1):
    #        inf_limit = self.rango[i]
    #        sup_limit = self.rango[i + 1]

    #        if i == len(self.rango) - 2: #last bin (39 of 40)
    #            mask = (self.one_volume >= inf_limit) & (self.one_volume <= sup_limit)
    #        else:
    #            mask = (self.one_volume >= inf_limit) & (self.one_volume < sup_limit)

    #        self.range_volumes[i][mask] += 1

    #        mrcfile.write(os.path.join(self._getExtraPath(), f'rangeVol_{i + 1}.mrc'),
    #                  self.range_volumes[i].astype(np.float32)
    #                  , voxel_size=self.voxel_size, overwrite=True)

    #    np.save(self._getExtraPath('first_bin.npy'), self.range_volumes[0])


    def statistic_volumes(self):
        self.range_volumes = []
        for i in range(1, self.numBins.get() + 1):
            volume = self._getExtraPath("rangeVol_%s.mrc" % i)
            self.range_volumes.append(NumpyImgHandler.loadMrc(volume).copy())

        bin_centers = np.array([(self.rango[i] + self.rango[i + 1]) / 2.0 for i in range(len(self.rango) - 1)],
                               dtype=np.float32)
        print(f'bin_centers {bin_centers}')
        print(f'bin_centers SHAPE {bin_centers.shape}')

        bin_centers_expanded = bin_centers[:, np.newaxis, np.newaxis, np.newaxis]
        print(f'bin_centers_expanded SHAPE {bin_centers_expanded.shape}')
        print('TIPO DE DATOS DE bin_centers_expanded', type(bin_centers_expanded))
        print('TIPO DE DATOS DE range_volumes', type(self.range_volumes))

        # total_count stores the number of times each voxel was assigned to any bin
        total_count = np.sum(self.range_volumes, axis=0)
        # nonzero_mask identifies voxels with at least one histogram observation
        nonzero_mask = total_count > 0

        print(f'min total_count {np.min(total_count)}')
        print(f'max total_count {np.max(total_count)}')
        print(f'unique total_count {np.unique(total_count)}')

        unique, counts = np.unique(total_count, return_counts=True)

        for value, count in zip(unique, counts):
            print(
                f"total_count = {value}: "
                f"{count} voxels "
                f"({100 * count / total_count.size:.6f} %)"
            )

        # ------------------------ WEIGHTED MEAN ----------------------------------------
        sum_mean = np.sum(self.range_volumes * bin_centers_expanded, axis=0)
        print(f'weighted_sum {sum_mean}')

        weighted_mean = np.divide(sum_mean, total_count,
                                  out=np.zeros_like(total_count, dtype=np.float32),
                                  where=nonzero_mask)   #total_count > 0

        #weighted_mean = sum_mean/(np.sum(self.range_volumes, axis=0)) #+ 10**-15
        print(f'valor de media ponderada {weighted_mean}')
        print(f'TAMAÑO de media ponderada {weighted_mean.shape}')

        # ------------------------ MOST PROBABLE BIN ----------------------------------------
        most_probable_freq = np.max(self.range_volumes, axis=0)
        print(f'Valor más probable por voxel:\n {most_probable_freq}'
              f'\n {most_probable_freq.shape}')

        max_index = np.argmax(self.range_volumes, axis=0)   # index bin with most counts for each voxel
        print(f'VALOR DE LOS INDICES: {max_index}')

        bin_values = np.squeeze(bin_centers_expanded[max_index])    # center of the bin
        print(f'DIMENSIONES DE VALORES_BIN {bin_values.shape}')
        print(f'Valor real correspondiente al máximo de frecuencia:{bin_values}')

        # Set to 0 for voxels with no data (all bins empty) to avoid false positive from argmax
        most_probable_freq[~nonzero_mask] = 0   #frequency = 0
        bin_values[~nonzero_mask] = 0           #value = 0

        # ------------------------ WEIGHTED VARIANCE AND STD ----------------------------------------
        sum_weighted_variance = np.sum(self.range_volumes * (bin_centers_expanded - weighted_mean) ** 2, axis=0)
        print(f'suma de varianza ponderada \n{sum_weighted_variance}')

        weighted_variance = np.divide(sum_weighted_variance, total_count,
                                      out=np.zeros_like(total_count, dtype=np.float32),
                                      where=nonzero_mask)

        print("Variance min:", np.min(weighted_variance))
        print("Variance max:", np.max(weighted_variance))
        print("Variance mean:", np.mean(weighted_variance))

        print("Non-zero variance voxels:", np.count_nonzero(weighted_variance > 0))
        print("Fraction non-zero variance:", np.count_nonzero(weighted_variance > 0) / weighted_variance.size)

        #weighted_variance = sum_weighted_variance/(np.sum(self.range_volumes, axis=0))# + 10**-15)
        print(f'Varianza ponderada SEPARADA: \n{weighted_variance}')
        print(f'TAMAÑO de varianza ponderada {weighted_variance.shape}')

        ##weighted_variance = np.average((bin_centers_expanded - weighted_mean) ** 2, axis=0, weights=self.range_volumes)
        ##print(f'Varianza ponderada: \n{weighted_variance}')

        #weighted_std = np.sqrt(weighted_variance)
        weighted_std = np.zeros_like(weighted_variance)
        std_mask = weighted_variance > 0
        weighted_std[std_mask] = np.sqrt(weighted_variance[std_mask])
        print(f'desviacion tipica \n{weighted_std}')

        # ------------------------ WEIGHTED SKEWNESS ----------------------------------------
        numerator = bin_centers_expanded - weighted_mean
        diff_over_std = np.divide(numerator, weighted_std,
                                  out=np.zeros_like(numerator, dtype=np.float32),
                                  where=std_mask[np.newaxis, :, :, :]) # std_mask condition to all dimensions
        sum_weighted_skew = np.sum(self.range_volumes * (diff_over_std) ** 3, axis=0)
        # Compute skewness/kurtosis only for voxels with valid observations (total_count > 0) and non-zero variance (std > 0).
        weighted_skewness = np.divide(sum_weighted_skew, total_count,
                                      out=np.zeros_like(total_count, dtype=np.float32),
                                      where=nonzero_mask & std_mask)

        #sum_weighted_skew = np.sum(self.range_volumes * ((bin_centers_expanded - weighted_mean)/weighted_std) ** 3, axis=0)
        #weighted_skewness = sum_weighted_skew/(np.sum(self.range_volumes, axis=0)) # + 10**-15)

        #weighted_skewness = np.average(((bin_centers_expanded - weighted_mean) / weighted_std) ** 3, axis=0, weights=self.range_volumes)
        print(f'Skewness ponderado \n{weighted_skewness}')

        # ------------------------   WEIGHTED KURTOSIS ----------------------------------------
        sum_weighted_kurt = np.sum(self.range_volumes * (diff_over_std) ** 4, axis=0)

        # Voxels with insufficient observations or zero variance remain equal to -3 after excess-kurtosis conversion.
        # This value is used as a sentinel indicating that the kurtosis could not be estimated.
        # Compute skewness/kurtosis only for voxels with valid observations (total_count > 0) and non-zero variance (std > 0).
        weighted_kurtosis = np.divide(sum_weighted_kurt, total_count,
                                      out=np.zeros_like(total_count, dtype=np.float32),
                                      where=nonzero_mask & std_mask) - 3

        print("Valid skewness voxels:", np.count_nonzero(nonzero_mask & std_mask))
        print("Invalid skewness voxels:", np.count_nonzero(~(nonzero_mask & std_mask)))
        valid_mask = nonzero_mask & std_mask

        print('min weighted_skewness[valid_mask]', np.min(weighted_skewness[valid_mask]))
        print('max weighted_skewness[valid_mask]', np.max(weighted_skewness[valid_mask]))

        print('min weighted_skewness[valid_mask]', np.min(weighted_kurtosis[valid_mask]))
        print('max weighted_skewness[valid_mask]', np.max(weighted_kurtosis[valid_mask]))

        #sum_weighted_kurt = np.sum(self.range_volumes * ((bin_centers_expanded - weighted_mean)/weighted_std) ** 4, axis=0)
        #weighted_kurtosis = sum_weighted_kurt/(np.sum(self.range_volumes, axis=0)) - 3
        ##weighted_kurtosis = sum_weighted_kurt / (np.sum(self.range_volumes, axis=0) + 10 ** -15) - 3

        ##weighted_kurtosis = np.average(((bin_centers_expanded - weighted_mean) / weighted_std) ** 4 - 3, axis=0, weights=self.range_volumes)
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


    #def statistic_volumes(self): ##### REVISAR
    #    ''''si para relion hay que añadir el valor de 10**-15 en el denominador del calculo de las
    #    ponderaciones, pero para xmipp no es necesario porque se realizan correctamente todos los
    #    calculos, lo ideal es crear una funcion que tenga ese parametro, por ejemplo, epsilon,
    #    de tal manera que cuando se haga la reconstr por relion valga 10**-15, pero cuando sea por
    #    medio de xmipp entonces valga 0 '''''


    #    self.range_volumes = []
    #    for i in range(1, self.numBins.get() + 1):
    #        volume = self._getExtraPath("rangeVol_%s.mrc" % i)
    #        #vol = NumpyImgHandler.loadMrc(volume)
    #        self.range_volumes.append(NumpyImgHandler.loadMrc(volume).copy())

    #    bin_centers = np.array([(self.rango[i] + self.rango[i + 1]) / 2.0 for i in range(len(self.rango) - 1)],
    #                           dtype=np.float32)
    #    print(f'bin_centers {bin_centers}')
    #    print(f'bin_centers SHAPE {bin_centers.shape}')

    #    bin_centers_expanded = bin_centers[:, np.newaxis, np.newaxis, np.newaxis]
    #    print(f'bin_centers_expanded SHAPE {bin_centers_expanded.shape}')
    #    print('TIPO DE DATOS DE bin_centers_expanded', type(bin_centers_expanded))
    #    print('TIPO DE DATOS DE range_volumes', type(self.range_volumes))


    #    # ------------------------ WEIGHTED MEAN ----------------------------------------
    #    sum_mean = np.sum(self.range_volumes * bin_centers_expanded, axis=0)
    #    print(f'weighted_sum {sum_mean}')

    #    weighted_mean = sum_mean/(np.sum(self.range_volumes, axis=0)) #+ 10**-15
    #    print(f'valor de media ponderada {weighted_mean}')
    #    print(f'TAMAÑO de media ponderada {weighted_mean.shape}')

    #    # ------------------------ MOST PROBABLE BIN ----------------------------------------
    #    most_probable_freq = np.max(self.range_volumes, axis=0)
    #    print(f'Valor más probable por voxel:\n {most_probable_freq}'
    #          f'\n {most_probable_freq.shape}')

    #    max_index = np.argmax(self.range_volumes, axis=0)
    #    print(f'VALOR DE LOS INDICES: {max_index}')

    #    bin_values = np.squeeze(bin_centers_expanded[max_index])
    #    print(f'DIMENSIONES DE VALORES_BIN {bin_values.shape}')
    #    print(f'Valor real correspondiente al máximo de frecuencia:{bin_values}')

    #    # ------------------------ WEIGHTED VARIANCE AND STD ----------------------------------------
    #    sum_weighted_variance = np.sum(self.range_volumes * (bin_centers_expanded - weighted_mean) ** 2, axis=0)
    #    print(f'suma de varianza ponderada \n{sum_weighted_variance}')
    #    weighted_variance = sum_weighted_variance/(np.sum(self.range_volumes, axis=0))# + 10**-15)
    #    print(f'Varianza ponderada SEPARADA: \n{weighted_variance}')
    #    print(f'TAMAÑO de varianza ponderada {weighted_variance.shape}')

    #    #weighted_variance = np.average((bin_centers_expanded - weighted_mean) ** 2, axis=0, weights=self.range_volumes)
    #    #print(f'Varianza ponderada: \n{weighted_variance}')

    #    weighted_std = np.sqrt(weighted_variance)
    #    print(f'desviacion tipica \n{weighted_std}')

    #    # ------------------------ WEIGHTED SKEWNESS ----------------------------------------
    #    sum_weighted_skew = np.sum(self.range_volumes * ((bin_centers_expanded - weighted_mean)/weighted_std) ** 3, axis=0)
    #    weighted_skewness = sum_weighted_skew/(np.sum(self.range_volumes, axis=0)) # + 10**-15)

    #    #weighted_skewness = np.average(((bin_centers_expanded - weighted_mean) / weighted_std) ** 3, axis=0, weights=self.range_volumes)
    #    print(f'Skewness ponderado \n{weighted_skewness}')

    #    # ------------------------   WEIGHTED KURTOSIS ----------------------------------------
    #    sum_weighted_kurt = np.sum(self.range_volumes * ((bin_centers_expanded - weighted_mean)/weighted_std) ** 4, axis=0)
    #    weighted_kurtosis = sum_weighted_kurt/(np.sum(self.range_volumes, axis=0)) - 3
    #    #weighted_kurtosis = sum_weighted_kurt / (np.sum(self.range_volumes, axis=0) + 10 ** -15) - 3

    #    #weighted_kurtosis = np.average(((bin_centers_expanded - weighted_mean) / weighted_std) ** 4 - 3, axis=0, weights=self.range_volumes)
    #    print(f'curtosis ponderada \n{weighted_kurtosis}')


    #    output_weighted_mean = os.path.join(self._getExtraPath(), 'weighted_mean.mrc')
    #    output_max_freq = os.path.join(self._getExtraPath(), 'most_probable_freq.mrc')
    #    output_max_bin = os.path.join(self._getExtraPath(), 'most_probable_bin.mrc')
    #    output_path_std = os.path.join(self._getExtraPath(), 'weighted_std.mrc')
    #    output_path_skewness = os.path.join(self._getExtraPath(), 'weighted_skewness.mrc')
    #    output_path_kurtosis = os.path.join(self._getExtraPath(), 'weighted_kurtosis.mrc')


    #    mrcfile.write(output_weighted_mean, weighted_mean.astype(np.float32), voxel_size=self.voxel_size)
    #    mrcfile.write(output_max_freq, most_probable_freq.astype(np.float32), voxel_size=self.voxel_size)
    #    mrcfile.write(output_max_bin, bin_values.astype(np.float32), voxel_size=self.voxel_size)
    #    mrcfile.write(output_path_std, weighted_std.astype(np.float32), voxel_size=self.voxel_size)
    #    mrcfile.write(output_path_skewness, weighted_skewness.astype(np.float32), voxel_size=self.voxel_size)
    #    mrcfile.write(output_path_kurtosis, weighted_kurtosis.astype(np.float32), voxel_size=self.voxel_size)



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


    #def _processParticles_old_ver(self, m_index=1):

    #    prot_classes = self.inputProt.get()

    #    inputParticles = prot_classes.getImages()

    #    outputParticles = self._createSetOfParticles()
    #    outputParticles.copyInfo(inputParticles)

    #    for cls in prot_classes:
    #        ids = list(cls.getIdSet())
    #        #print('ids', sorted(ids))

    #        if not ids:
    #            continue

    #        # Deterministic seed for reproducibility
    #        # 42: arbitrary constant
    #        # cls.getObjId(): ensures different classes get different seeds
    #        # m_index: ensures different bootstrap iterations get different samples from the same
    #        #          class; without it, every iteration would select the same particles, defeating
    #        #          the purpose of bootstrapping
    #        random.seed(42 + cls.getObjId() + m_index)

    #        selected_ids = random.sample(ids, min(self.numParticlesPerClass.get(), len(ids)))
    #        #print('selected_ids', selected_ids)
    #        #print("------------------\n")

    #        for objId in selected_ids:
    #            particle = inputParticles[objId]
    #            outputParticles.append(particle.clone())

    #    #self._defineOutputs(outputParticles=outputParticles)
    #    outputParticles.write()
    #    outputParticles.close()


    #    if self.reconstruction == RELION_RECONSTRUCTION:
    #        # Save new .star file
    #        output_star = self._getExtraPath(f'sample_{m_index}.star') #cambiar al temporal
    #        writeSetOfParticles(
    #            outputParticles,
    #            output_star,
    #            outputDir=self._getExtraPath(),
    #            alignType=ALIGN_PROJ
    #        )

    #        print("STAR written:", output_star)

    #    else:
    #        # Save new .xmd
    #        output_xmd = self._getTmpPath(f"sample_{m_index}.xmd") #_getExtraPath
    #        writeSetOfParticlesXmipp(outputParticles, output_xmd)


    #def _processParticles_completo(self, m_index=1):
         # Esta el solpamaiento y el bootstrapping junto

    #    prot_classes = self.inputProt.get()
    #    inputParticles = prot_classes.getImages()

    #    print("Input particles:")
    #    print("  Size:", inputParticles.getSize())
    #    print("  Dimensions:", inputParticles.getDimensions())

    #    outputParticles = self._createSetOfParticles()
    #    outputParticles.copyInfo(inputParticles)

    #    class_info = []
    #    for cls in prot_classes:
    #        # returns the IDs of the particles belonging to that class
    #        particle_ids = set(cls.getIdSet())

    #        if not particle_ids:
    #            continue

    #        class_info.append({
    #            "class_id": cls.getObjId(),  # identifies the Class2D object
    #            "ids": particle_ids,
    #            "size": len(particle_ids)
    #        })

    #    # 2. FIND PARTICLES THAT APPEAR IN MULTIPLE CLASSES
    #    # =========================================================
    #    # Map each particle ID to the classes it belongs to
    #    particle_classes = {}
    #    for info in class_info:
    #        class_id = info["class_id"]

    #        # Assign the current class ID to each particle in the class
    #        for particle_id in info["ids"]:
    #            if particle_id not in particle_classes:
    #                particle_classes[particle_id] = []

    #            particle_classes[particle_id].append(class_id)

    #    # 3. FIND OVERLAPPING PARTICLES
    #    # =========================================================
    #    # Identify particles assigned to more than one class
    #    overlapping_particles = {
    #        particle_id: class_ids
    #        for particle_id, class_ids in particle_classes.items() if len(class_ids) > 1
    #    }
    #    print("Particle/class overlap check")
    #    print("==============================================")

    #    print(f"Total unique particles before resolving overlap: {len(particle_classes)}")
    #    print(f"Particles present in multiple classes: {len(overlapping_particles)}")

    #    # If a particle belongs to several classes, keep it in the MINORITY class, i.e. the class
    #    # containing fewer particles
    #    # In case of a tie, the class with the smallest class ID wins. This makes the decision deterministic
    #    allowed_ids_by_class = {
    #        info["class_id"]: set(info["ids"])
    #        for info in class_info
    #    }

    #    if overlapping_particles:
    #        print("Resolving overlapping particles...")
    #        print("----------------------------------------------")

    #        class_sizes = {
    #            info["class_id"]: info["size"]
    #            for info in class_info
    #        }

    #        for particle_id, class_ids in overlapping_particles.items():
    #            # Sort by number of particles in the class and class ID. The smallest class wins
    #            winning_class = min(
    #                class_ids,
    #                key=lambda class_id: (
    #                    class_sizes[class_id],
    #                    class_id)
    #            )

    #            print(f"Particle {particle_id}: "
    #                  f"classes {class_ids} -> "
    #                  f"keeping in minority class {winning_class} "
    #                  f"(size={class_sizes[winning_class]})")

    #            # Remove the particle from every class except the winning class
    #            for class_id in class_ids:
    #                if class_id != winning_class:
    #                    allowed_ids_by_class[class_id].discard(particle_id)

    #        print("----------------------------------------------")
    #        print(f"Resolved {len(overlapping_particles)} overlapping particles")
    #        print("----------------------------------------------")

    #    else:
    #        print("No particle overlap detected")


    #    # IMPORTANT:
    #    # We sample from allowed_ids_by_class, NOT directly from cls.getIdSet().
    #    # Therefore particles that were found in several classes are only available in their minority class.
    #    used_ids = set()
    #    duplicates_skipped = 0

    #    for class_id, ids_set in allowed_ids_by_class.items():
    #        ids = list(ids_set)

    #        if not ids:
    #            continue

    #        # Deterministic seed for reproducibility
    #        # 42: arbitrary constant
    #        # class_id: ensures different classes get different seeds
    #        # m_index: ensures different bootstrap iterations get different samples from the same
    #        #          class; without it, every iteration would select the same particles, defeating
    #        #          the purpose of bootstrapping
    #        random.seed(42 + class_id + m_index)

    #        selected_ids = random.sample(ids, min(self.numParticlesPerClass.get(), len(ids)))
    #        #print('selected_ids', selected_ids)
    #        #print("------------------\n")

    #        for particle_id in selected_ids:
    #            # Even after resolving class overlaps, make sure the same particle can NEVER be added twice to
    #            # the reconstruction.
    #            if particle_id in used_ids:
    #                duplicates_skipped += 1

    #                print(f"WARNING: Particle {particle_id} was already selected in bootstrap {m_index}. "
    #                      f"Skipping duplicate")
    #                continue

    #            used_ids.add(particle_id)
    #            particle = inputParticles[particle_id]
    #            outputParticles.append(particle.clone())

    #    # AÑADIDO AHORA
    #    print("Output particles:")
    #    print("  Size:", outputParticles.getSize())
    #    print("  Dimensions:", outputParticles.getDimensions())


    #    num_selected = len(used_ids)
    #    num_output = outputParticles.getSize()

    #    print("==============================================")
    #    print(f"BOOTSTRAP {m_index} CHECK")
    #    print("==============================================")

    #    print(f"Particles selected: {num_selected}")
    #    print(f"Duplicates skipped during bootstrap: {duplicates_skipped}")
    #    print(f"Particles in output: {num_output}")

    #    # This should always be true because used_ids is a set and duplicates are filtered before append()
    #    if num_selected != num_output:
    #        raise RuntimeError(
    #            f"Internal consistency error in bootstrap "
    #            f"{m_index}: "
    #            f"{num_selected} selected particles but "
    #            f"{num_output} particles in output."
    #        )

    #    print("==============================================")

    #    outputParticles.write()
    #    print("Output particles 2:")
    #    print("  Size:", outputParticles.getSize())
    #    print("  Dimensions:", outputParticles.getDimensions())
    #    outputParticles.close()

    #    if self.reconstruction == RELION_RECONSTRUCTION:
    #        output_star = self._getExtraPath(f'sample_{m_index}.star')  #cambiar al temporal
    #        writeSetOfParticles(
    #            outputParticles,
    #            output_star,
    #            outputDir=self._getExtraPath(),
    #            alignType=ALIGN_PROJ
    #        )

    #        print("STAR written:", output_star)

    #    else:
    #        # Save new .xmd
    #        output_xmd = self._getTmpPath(f"sample_{m_index}.xmd")  #_getExtraPath
    #        writeSetOfParticlesXmipp(outputParticles, output_xmd)
    #        print("XMD written:", output_xmd)



