# **************************************************************************
# *
# * Authors:     Swathi Adinarayanan (swathi.adinarayanan@mail.mcgill.ca)
# *              Javier Vargas (javier.vargasbalbuena@mcgill.ca)
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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************f
from os.path import join, exists
import math
import numpy as np

import pwem.emlib.metadata as md
import pyworkflow.protocol.params as pwparams
import pyworkflow.protocol.constants as pwcons
import pyworkflow.utils.path as pwpath
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfParticles, ALIGN_PROJ
from pwem.protocols import ProtAnalysis3D, ALIGN_NONE

import cryomethods.convert.convertXmp as convXmp
import cryomethods.constants as cmcons
from cryomethods.convert import writeSetOfParticles
#

class ProtDirectionalPruning(ProtAnalysis3D):
    """    
    Analyze 2D classes as assigned to the different directions. Be more
    creative and do better explanation about your method.
    """
    _label = 'directional_pruning'

    def __init__(self, *args, **kwargs):
        ProtAnalysis3D.__init__(self, *args, **kwargs)

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', pwparams.PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      important=True,
                      label="Input particles",
                      help='Select the input projection images with an '
                           'angular assignment.')
        form.addParam('inputVolume', pwparams.PointerParam,
                      pointerClass='Volume', label="Input volume",
                      important=True,
                      help='Select the input volume.')
        form.addParam('symmetryGroup', pwparams.StringParam, default='c1',
                      label="Symmetry group",
                      help='See [[Xmipp Symmetry][http://www2.mrc-lmb.cam.ac.uk'
                           '/Xmipp/index.php/Conventions_%26_'
                           'File_formats#Symmetry]] page for a description of '
                           'the symmetry format accepted by Xmipp')

        form.addSection(label='Pruning')
        form.addParam('classMethod', pwparams.EnumParam, default=cmcons.CL2D,
                      label='Choose a method to classify',
                      choices=['CL2D', 'Relion 2D'],
                      display=pwparams.EnumParam.DISPLAY_COMBO)
        form.addParam('targetResolution', pwparams.FloatParam, default=10,
                       label='Target Resolution (A)',
                       help='In order to save time, you could rescale both '
                            'particles and maps to a pixel size = resol/2. '
                            'If set to 0, no rescale will be applied to '
                            'the initial references.')

        group = form.addGroup('Solid Angles settings')
        group.addParam('angularSampling', pwparams.FloatParam, default=5,
                      label='Angular sampling',
                      help="Angular step size, in degrees")
        group.addParam('angularDistance', pwparams.FloatParam, default=10,
                      label='Angular distance',
                      help="In degrees. An image belongs to a group if its "
                           "distance is smaller than this value")
        group.addParam('directionalClasses', pwparams.IntParam, default=2,
                      label='Number of directional classes',
                      expertLevel=pwcons.LEVEL_ADVANCED)
        group.addParam('thresholdValue', pwparams.FloatParam, default=0.25,
                      label='DO BETTER EXPLANATION!!!!!',
                      help='Enter a value less than 1(in decimals)')
        group.addParam('noOfParticles', pwparams.IntParam,default=25,
                      label='Number of Particles',
                      help='Minimum number of particles required to do 2D'
                           'Classification')

        form.addSection(label='Methods Settings')
        form.addParam('cl2dIterations', pwparams.IntParam, default=10,
                      label='Number of CL2D iterations',
                      condition="classMethod==0" ,
                      expertLevel=pwcons.LEVEL_ADVANCED)
        form.addParam('maxIters', pwparams.IntParam, default=100,
                      expertLevel=pwcons.LEVEL_ADVANCED,
                      condition='classMethod==1',
                      label='Maximum number of iterations',
                      help='If the convergence has not been reached after '
                           'this number of iterations, the process will be '
                           'stopped.')
        form.addParam('maxShift', pwparams.FloatParam, default=15,
                      label='Maximum shift',
                      expertLevel=pwcons.LEVEL_ADVANCED,
                      condition='classMethod!=2',
                      help="Provide maximum shift, In pixels")

        group = form.addGroup('CTF', condition='classMethod==2')
        group.addParam('continueMsg', pwparams.LabelParam, default=True,
                      condition='classMethod==2',
                      label='CTF parameters are not available in continue mode')
        group.addParam('doCTF', pwparams.BooleanParam, default=True,
                      label='Do CTF-correction?', condition='classMethod==2',
                      help='If set to Yes, CTFs will be corrected inside the '
                           'MAP refinement. The resulting algorithm '
                           'intrinsically implements the optimal linear, or '
                           'Wiener filter. Note that input particles should '
                           'contains CTF parameters.')
        group.addParam('hasReferenceCTFCorrected', pwparams.BooleanParam,
                       default=False,
                       condition='classMethod==2',
                       label='Has reference been CTF-corrected?',
                       help='Set this option to Yes if the reference map '
                            'represents CTF-unaffected density, e.g. it was '
                            'created using Wiener filtering inside RELION or '
                            'from a PDB. If set to No, then in the first '
                            'iteration, the Fourier transforms of the '
                            'reference projections are not multiplied by the '
                            'CTFs.')
        group.addParam('haveDataBeenPhaseFlipped', pwparams.LabelParam,
                      condition='classMethod==2',
                      label='Have data been phase-flipped?      '
                            '(Don\'t answer, see help)',
                      help='The phase-flip status is recorded and managed by '
                           'Scipion. \n In other words, when you import or '
                           'extract particles, \nScipion will record whether '
                           'or not phase flipping has been done.\n\n'
                           'Note that CTF-phase flipping is NOT a necessary '
                           'pre-processing step \nfor MAP-refinement in '
                           'RELION, as this can be done inside the internal\n'
                           'CTF-correction. However, if the phases have been '
                           'flipped, the program will handle it.')
        group.addParam('ignoreCTFUntilFirstPeak', pwparams.BooleanParam,
                       default=False,
                       expertLevel=pwcons.LEVEL_ADVANCED,
                       label='Ignore CTFs until first peak?',
                       condition='classMethod==2',
                       help='If set to Yes, then CTF-amplitude correction will '
                            'only be performed from the first peak '
                            'of each CTF onward. This can be useful if the CTF '
                            'model is inadequate at the lowest resolution. '
                            'Still, in general using higher amplitude contrast '
                            'on the CTFs (e.g. 10-20%) often yields better '
                            'results. Therefore, this option is not generally '
                            'recommended.')
        group.addParam('doCtfManualGroups', pwparams.BooleanParam, default=False,
                      label='Do manual grouping ctfs?',
                      condition='classMethod==2',
                      help='Set this to Yes the CTFs will grouping manually.')
        group.addParam('defocusRange', pwparams.FloatParam, default=500,
                      label='defocus range for group creation (in Angstroms)',
                      condition='classMethod==2 and doCtfManualGroups',
                      help='Particles will be grouped by defocus.'
                           'This parameter is the bin for an histogram.'
                           'All particles assigned to a bin form a group')
        group.addParam('numParticles', pwparams.FloatParam, default=200,
                      label='minimum size for defocus group',
                      condition='classMethod==2 and doCtfManualGroups',
                      help='If defocus group is smaller than this value, '
                           'it will be expanded until number of particles '
                           'per defocus group is reached')

        group = form.addGroup('Optimization', condition='classMethod==2')
        group.addParam('numberOfIterations', pwparams.IntParam, default=25,
                      label='Number of iterations', condition='classMethod==2',
                      help='Number of iterations to be performed. Note '
                           'that the current implementation does NOT '
                           'comprise a convergence criterium. Therefore, '
                           'the calculations will need to be stopped '
                           'by the user if further iterations do not yield '
                           'improvements in resolution or classes. '
                           'If continue option is True, you going to do '
                           'this number of new iterations (e.g. if '
                           '*Continue from iteration* is set 3 and this '
                           'param is set 25, the final iteration of the '
                           'protocol will be the 28th.')
        group.addParam('regularisationParamT', pwparams.IntParam,
                      default=2, label='Regularisation parameter T',
                      condition='classMethod==2',
                      help='Bayes law strictly determines the relative '
                           'weight between the contribution of the '
                           'experimental data and the prior. '
                           'However, in practice one may need to adjust '
                           'this weight to put slightly more weight on the '
                           'experimental data to allow optimal results. '
                           'Values greater than 1 for this regularisation '
                           'parameter (T in the JMB2011 paper) put more '
                           'weight on the experimental data. Values around '
                           '2-4 have been observed to be useful for 3D '
                           'refinements, values of 1-2 for 2D refinements. '
                           'Too small values yield too-low resolution '
                           'structures; too high values result in '
                           'over-estimated resolutions and overfitting.')
        group.addParam('copyAlignment', pwparams.BooleanParam, default=True,
                      label='Consider previous alignment?',
                      condition='classMethod==2',
                      help='If set to Yes, then alignment information from'
                           ' input particles will be considered.')
        group.addParam('alignmentAsPriors', pwparams.BooleanParam,
                       default=False,
                       condition='classMethod==2',
                       expertLevel=pwcons.LEVEL_ADVANCED,
                       label='Consider alignment as priors?',
                       help='If set to Yes, then alignment information from '
                           'input particles will be considered as PRIORS. This '
                           'option is mandatory if you want to do local '
                           'searches')
        group.addParam('fillRandomSubset', pwparams.BooleanParam, default=False,
                      condition='classMethod==2',
                      expertLevel=pwcons.LEVEL_ADVANCED,
                      label='Consider random subset value?',
                      help='If set to Yes, then random subset value '
                           'of input particles will be put into the'
                           'star file that is generated.')
        group.addParam('maskDiameterA', pwparams.IntParam, default=-1,
                      condition='classMethod==2',
                      label='Particle mask diameter (A)',
                      help='The experimental images will be masked with a '
                           'soft circular mask with this <diameter>. '
                           'Make sure this diameter is not set too small '
                           'because that may mask away part of the signal! If '
                           'set to a value larger than the image size no '
                           'masking will be performed.\n\n'
                           'The same diameter will also be used for a '
                           'spherical mask of the reference structures if no '
                           'user-provided mask is specified.')
        group.addParam('referenceClassification' , pwparams.BooleanParam,
                      default=True, condition='classMethod==2',
                       label='Perform reference based classification?')

        group = form.addGroup('Sampling', condition='classMethod==2')
        group.addParam('doImageAlignment', pwparams.BooleanParam, default=True,
                      label='Perform Image Alignment?',
                      condition='classMethod==2',
                      )
        group.addParam('inplaneAngularSamplingDeg', pwparams.FloatParam, default=5,
                      label='In-plane angular sampling (deg)',
                      condition='classMethod==2 and doImageAlignment',

                      help='The sampling rate for the in-plane rotation '
                           'angle (psi) in degrees.\n'
                           'Using fine values will slow down the program. '
                           'Recommended value for\n'
                           'most 2D refinements: 5 degrees. \n\n'
                           'If auto-sampling is used, this will be the '
                           'value for the first \niteration(s) only, and '
                           'the sampling rate will be increased \n'
                           'automatically after that.')
        group.addParam('offsetSearchRangePix', pwparams.FloatParam, default=5,
                      condition='classMethod==2 and doImageAlignment',
                      label='Offset search range (pix)',
                      help='Probabilities will be calculated only for '
                           'translations in a circle with this radius (in '
                           'pixels). The center of this circle changes at '
                           'every iteration and is placed at the optimal '
                           'translation for each image in the previous '
                           'iteration.')
        group.addParam('offsetSearchStepPix', pwparams.FloatParam, default=1.0,
                      condition='classMethod==2 and doImageAlignment',
                      label='Offset search step (pix)',
                      help='Translations will be sampled with this step-size '
                           '(in pixels). Translational sampling is also done '
                           'using the adaptive approach. Therefore, if '
                           'adaptive=1, the translations will first be '
                           'evaluated on a 2x coarser grid.')
        form.addSection(label='Compute')
        form.addParam('noRelion', pwparams.HiddenBooleanParam,
                      condition='classMethod!=2',
                      label='This section is empty cause is only useful in '
                            'case of using 2D classification Relion method')
        form.addParam('allParticlesRam', pwparams.BooleanParam, default=False,
                      label='Pre-read all particles into RAM?',
                      condition='classMethod==2',
                      help='If set to Yes, all particle images will be '
                           'read into computer memory, which will greatly '
                           'speed up calculations on systems with slow '
                           'disk access. However, one should of course be '
                           'careful with the amount of RAM available. '
                           'Because particles are read in '
                           'float-precision, it will take \n'
                           '( N * (box_size)^2 * 4 / (1024 * 1024 '
                           '* 1024) ) Giga-bytes to read N particles into '
                           'RAM. For 100 thousand 200x200 images, that '
                           'becomes 15Gb, or 60 Gb for the same number of '
                           '400x400 particles. Remember that running a '
                           'single MPI slave on each node that runs as '
                           'many threads as available cores will have '
                           'access to all available RAM.\n\n'
                           'If parallel disc I/O is set to No, then only '
                           'the master reads all particles into RAM and '
                           'sends those particles through the network to '
                           'the MPI slaves during the refinement '
                           'iterations.')
        form.addParam('scratchDir', pwparams.PathParam,
                      condition='classMethod==2 and not allParticlesRam',
                      label='Copy particles to scratch directory: ',
                      help='If a directory is provided here, then the job '
                           'will create a sub-directory in it called '
                           'relion_volatile. If that relion_volatile '
                           'directory already exists, it will be wiped. '
                           'Then, the program will copy all input '
                           'particles into a large stack inside the '
                           'relion_volatile subdirectory. Provided this '
                           'directory is on a fast local drive (e.g. an '
                           'SSD drive), processing in all the iterations '
                           'will be faster. If the job finishes '
                           'correctly, the relion_volatile directory will '
                           'be wiped. If the job crashes, you may want to '
                           'remove it yourself.')
        form.addParam('combineItersDisc', pwparams.BooleanParam, default=False,
                      label='Combine iterations through disc?',
                      condition='classMethod==2',
                      help='If set to Yes, at the end of every iteration '
                           'all MPI slaves will write out a large file '
                           'with their accumulated results. The MPI '
                           'master will read in all these files, combine '
                           'them all, and write out a new file with the '
                           'combined results. All MPI slaves will then '
                           'read in the combined results. This reduces '
                           'heavy load on the network, but increases load '
                           'on the disc I/O. This will affect the time it '
                           'takes between the progress-bar in the '
                           'expectation step reaching its end (the mouse '
                           'gets to the cheese) and the start of the '
                           'ensuing maximisation step. It will depend on '
                           'your system setup which is most efficient.')
        form.addParam('doGpu', pwparams.BooleanParam, default=True,
                      label='Use GPU acceleration?',
                      condition='classMethod==2',
                      help='If set to Yes, the job will try to use GPU '
                           'acceleration.')
        form.addParam('gpusToUse', pwparams.StringParam, default='',
                      label='Which GPUs to use:',
                      condition='classMethod==2 and doGpu',
                      help='This argument is not necessary. If left empty, '
                           'the job itself will try to allocate available '
                           'GPU resources. You can override the default '
                           'allocation by providing a list of which GPUs '
                           '(0,1,2,3, etc) to use. MPI-processes are '
                           'separated by ":", threads by ",". '
                           'For example: "0,0:1,1:0,0:1,1"')
        form.addParam('useParallelDisk', pwparams.BooleanParam, default=True,
                          label='Use parallel disc I/O?',
                          condition='classMethod==2',
                          help='If set to Yes, all MPI slaves will read '
                               'their own images from disc. Otherwise, only '
                               'the master will read images and send them '
                               'through the network to the slaves. Parallel '
                               'file systems like gluster of fhgfs are good '
                               'at parallel disc I/O. NFS may break with many '
                               'slaves reading in parallel.')
        form.addParam('pooledParticles', pwparams.IntParam, default=3,
                          label='Number of pooled particles:',
                          condition='classMethod==2',
                          help='Particles are processed in individual batches '
                               'by MPI slaves. During each batch, a stack of '
                               'particle images is only opened and closed '
                               'once to improve disk access times. All '
                               'particle images of a single batch are read '
                               'into memory together. The size of these '
                               'batches is at least one particle per thread '
                               'used. The nr_pooled_particles parameter '
                               'controls how many particles are read together '
                               'for each thread. If it is set to 3 and one '
                               'uses 8 threads, batches of 3x8=24 particles '
                               'will be read together. This may improve '
                               'performance on systems where disk access, and '
                               'particularly metadata handling of disk '
                               'access, is a problem. It has a modest cost of '
                               'increased RAM usage.')

        form.addParallelSection(threads=1, mpi=3)

    ###TODO LIST:
    #1) ML number of iterations
    #2) Input parameter to determine minimun number of particles to do or not
    # to do 2D classification: line 223, 25 number

    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        partId = self.inputParticles.get().getObjId()
        volId = self.inputVolume.get().getObjId()

        self._insertFunctionStep('convertInputStep', partId, volId,
                                 self.targetResolution.get())
        self._insertFunctionStep('constructGroupsStep', partId,
                                 self.angularSampling.get(),
                                 self.angularDistance.get(),
                                 self.symmetryGroup.get())
        self._insertFunctionStep('classifyGroupsStep')
        self._insertFunctionStep('cleanStep')
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions -------------------------------
    def convertInputStep(self, inputParticles, inputVolume, targetResolution):
        """ Write the input images as a Xmipp metadata file. 
        particlesId: is only need to detect changes in
        input particles and cause restart from here.
        """
        imgSet = self._getInputParticles()
        convXmp.writeSetOfParticles(imgSet, self._getInputXmd())
        newXdim =self._getNewDim()

        ih = ImageHandler()
        inputVol = self.inputVolume.get()
        fn = ih.fixXmippVolumeFileName(inputVol)
        img = ih.read(fn)
        img.scale(newXdim, newXdim, newXdim)
        img.write(self._getExtraPath("volume.vol"))

        args = '-i %(parts)s -o %(scaledStk)s --save_metadata_stack '
        args += '%(scaledXmd)s --dim %(xDim)d'

        params = {"parts" : self._getInputXmd(),
                  "scaledStk": self._getExtraPath('scaled_particles.stk'),
                  "scaledXmd": self._getExtraPath('scaled_particles.xmd'),
                  "xDim": newXdim
                  }
        self.runJob("xmipp_image_resize", args % params)

    def constructGroupsStep(self, partId, angularSampling,
                            angularDistance, symmetryGroup):
        # Generate projections from this reconstruction
        argsProj = {"-i": self._getExtraPath("volume.vol"),
                    "-o": self._getExtraPath("gallery.stk"),
                    "--sym": self.symmetryGroup.get(),
                    "--sampling_rate": self.angularSampling.get(),
                    "--angular_distance": self.angularDistance.get(),
                    "--experimental_images":
                        self._getExtraPath('scaled_particles.xmd'),
                    "--method": "fourier 1 0.25 bspline",
                    "--compute_neighbors": "",
                    "--max_tilt_angle": 180
                    }

        argsNeib = {"-o": self._getExtraPath("neighbours.xmd"),
                    "--i2": self._getExtraPath("gallery.doc"),
                   "--sym": self.symmetryGroup.get(),
                   "--dist": self.angularDistance.get(),
                   "--i1": self._getExtraPath('scaled_particles.xmd'),
                   "--check_mirrors": ""
                   }

        params = self._getParams(argsProj)
        self.runJob("xmipp_angular_project_library", params)

        paramsNeib = self._getParams(argsNeib)
        self.runJob("angular_neighbourhood", paramsNeib, numberOfMpi=1)

    def classifyGroupsStep(self):
        fnNeighbours = self._getExtraPath("neighbours.xmd")
        gallery = self._getExtraPath("gallery.stk")

        for block in md.getBlocksInMetaDataFile(fnNeighbours):

            #creating a folder to each direction
            imgNo = int(block.split("_")[1])
            fnDir = self._getExtraPath("direction_%05d" % imgNo)
            fnBlock = "%s@%s" % (block, fnNeighbours)

            if not exists(fnDir):
                pwpath.makePath(fnDir)

            if self.classMethod.get() == cmcons.CL2D:
                dirClasses = self.directionalClasses.get()
                nLevs = self._getCl2dLevels()
                fnOut = join(fnDir, "level_%02d/class_classes.stk" % nLevs)
                self._runClassifSteps(fnOut, fnBlock, fnDir, imgNo, gallery,
                                      callbackMethod=self._runCl2dStep)
            else:
                relPart = self._createSetOfParticles()
                relPart.copyInfo(self.inputParticles.get())
                fnOut = join(fnDir, "class_")
                self._runClassifSteps(fnOut, fnBlock, fnDir, imgNo, gallery,
                                      callbackMethod=self._runRelionStep)

    def _runClassifSteps(self, fnOut, fnBlock, fnDir,
                         imgNo, fnGallery, callbackMethod=None):
        nop = self.noOfParticles.get()
        if callbackMethod and not exists(fnOut):
            if md.getSize(fnBlock) > nop:
                # try:
                callbackMethod(fnOut, fnBlock, fnDir, imgNo, fnGallery)
                # except:
                #     print("The classification failed, probably because of a "
                #           "low number of images. However, this classification "
                #           "does not hinder the protocol to continue")

    def _runCl2dStep(self, fnOut, fnBlock, fnDir, imgNo, fnGallery):
        fnAlignRoot = join(fnDir, "classes")
        args = {'-i': fnBlock,
                '--odir': fnDir,
                '--ref0': str(imgNo) + '@' + fnGallery,
                '--iter': self.cl2dIterations.get(),
                '--nref': self.directionalClasses.get(),
                '--distance': 'correlation',
                '--classicalMultiref': '',
                '--maxShift': self.maxShift.get()
                }

        argsAlign = {'-i': fnOut,
                     '--oroot': fnAlignRoot,
                     '--ref': str(imgNo) + '@' + fnGallery,
                     '--iter': 1
                     }

        params = self._getParams(args)
        self.runJob("xmipp_classify_CL2D", params)

        parALign = self._getParams(argsAlign)
        self.runJob("xmipp_image_align", parALign, numberOfMpi=1)

        paramGeo = "-i %s_alignment.xmd --apply_transform" % fnAlignRoot
        self.runJob("xmipp_transform_geometry", paramGeo , numberOfMpi=1)

        fnOutput = self._getPath('output_particles.xmd')
        mdOutput = self._getMetadata(fnOutput)

        cLevels = self._getCl2dLevels()
        mdClassesFn = fnDir + '/level_%02d/class_classes.xmd' % cLevels

        CC = []
        mdCount = self._getMetadata(mdClassesFn)
        for row in md.iterRows(mdCount):
            CC.append(row.getValue(md.MDL_CLASS_COUNT))

        n = sum(CC)
        out = np.true_divide(CC, n)
        highest = out >= self.thresholdValue.get()
        xmippBlocks = md.getBlocksInMetaDataFile(mdClassesFn)[2:]

        for indx, block in enumerate(xmippBlocks):
            fnBlock = '%s@%s' % (block, mdClassesFn)
            mdClassCount = self._getMetadata(fnBlock)
            for row in md.iterRows(mdClassCount):
                objId = mdOutput.addObject()
                if not highest[indx]:
                    row.setValue(md.MDL_ENABLED, -1)
                row.writeToMd(mdOutput, objId)
        mdOutput.write(fnOutput)

    def _runRelionStep(self, fnOut, fnBlock, fnDir, imgNo, fnGallery):
        relPart = SetOfParticles(filename=":memory:")
        convXmp.readSetOfParticles(fnBlock, relPart)

        if self.copyAlignment.get():
            alignType = relPart.getAlignment()
            alignType != ALIGN_NONE
        else:
            alignType = ALIGN_NONE

        alignToPrior = getattr(self, 'alignmentAsPriors', False)
        fillRandomSubset = getattr(self, 'fillRandomSubset', False)
        fnRelion = self._getExtraPath('relion_%s.star' % imgNo)

        writeSetOfParticles(relPart, fnRelion, self._getExtraPath(),
                            alignType=alignType,
                            postprocessImageRow=self._postprocessParticleRow,
                            fillRandomSubset=fillRandomSubset)
        if alignToPrior:
            mdParts = md.MetaData(fnRelion)
            self._copyAlignAsPriors(mdParts, alignType)
            mdParts.write(fnRelion)
        if self.doCtfManualGroups:
            self._splitInCTFGroups(fnRelion)

        args = {}
        self._setNormalArgs(args)
        args['--i'] = fnRelion
        args['--o'] = fnOut
        if self.referenceClassification.get():
            fnRef = "%s@%s" % (imgNo, fnGallery)
            args['--ref'] = fnRef
        self._setComputeArgs(args)

        params = ' '.join(
            ['%s %s' % (k, str(v)) for k, v in args.items()])
        print('Vamos a correr relion', params)
        self.runJob(self._getRelionProgram(), params)

        clsDistList = []
        it = self.numberOfIterations.get()
        model = '_it%03d_' % it
        fnModel = (fnOut + model + 'model.star')

        block = md.getBlocksInMetaDataFile(fnModel)[1]
        fnBlock = "%s@%s" % (block, fnModel)

        mdBlocks = md.MetaData(fnBlock)
        fnData = (fnOut + model + 'data.star')

        for objId in mdBlocks:
            clsDist = mdBlocks.getValue(md.RLN_MLMODEL_PDF_CLASS, objId)
            clsDistList.append(clsDist)

        fnOutput = self._getPath('output_particles.xmd')
        mdOutput = self._getMetadata(fnOutput)

        mdParticles = md.MetaData(fnData)
        for row in md.iterRows(mdParticles):
            objId = mdOutput.addObject()
            clsNum = row.getValue('rlnClassNumber')
            clsDist = clsDistList[clsNum-1]

            if clsDist >= self.thresholdValue.get():
                row.setValue(md.MDL_ENABLED, 1)
            else:
                row.setValue(md.MDL_ENABLED, -1)
            row.writeToMd(mdOutput, objId)
        mdOutput.write(fnOutput)

    def cleanStep(self):
        pass

    def createOutputStep(self):
        inputImgs = self._getInputParticles()
        fnOutput = self._getPath('output_particles.xmd')

        imgSetOut = self._createSetOfParticles()
        imgSetOut.copyInfo(inputImgs)
        imgSetOut.setAlignmentProj()
        self._fillDataFromIter(imgSetOut, fnOutput)

        self._defineOutputs(outputParticles=imgSetOut)
        self._defineSourceRelation(self.inputParticles, imgSetOut)
        self._defineSourceRelation(self.inputVolume, imgSetOut)

    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        validateMsgs = []
        # if there are Volume references, it cannot be empty.
        if self.inputVolume.get() and not self.inputVolume.hasValue():
            validateMsgs.append('Please provide an input reference volume.')
        if self.inputParticles.get() and not self.inputParticles.hasValue():
            validateMsgs.append('Please provide input particles.')
        return validateMsgs

    def _summary(self):
        summary = []
        return summary

    #--------------------------- UTILS functions -------------------------------
    def _setNormalArgs(self, args):
        maskDiameter = self.maskDiameterA.get()
        newTs = self.targetResolution.get() * 0.4
        if maskDiameter <= 0:
          x = self._getInputParticles().getDim()[0]
          maskDiameter = self._getInputParticles().getSamplingRate() * x

        args.update({'--particle_diameter': maskDiameter,
                     '--angpix': newTs,
                     })

        args['--zero_mask'] = ''
        args['--K'] = self.directionalClasses.get()

        self._setCTFArgs(args)
        self._setBasicArgs(args)

    def _setComputeArgs(self, args):
        if not self.combineItersDisc:
            args['--dont_combine_weights_via_disc'] = ''

        if not self.useParallelDisk:
            args['--no_parallel_disc_io'] = ''

        if self.allParticlesRam:
            args['--preread_images'] = ''
        else:
             if self.scratchDir.get():
                args['--scratch_dir'] = self.scratchDir.get()

        args['--pool'] = self.pooledParticles.get()

        if self.doGpu:
            args['--gpu'] = self.gpusToUse.get()

        args['--j'] = self.numberOfThreads.get()

    def _setBasicArgs(self, args):
        """ Return a dictionary with basic arguments. """
        args.update({'--flatten_solvent': '',
                     '--dont_check_norm': '',
                     '--scale': '',
                     '--oversampling': 1
                     })

       # if self.IS_CLASSIFY:
        args['--tau2_fudge'] = self.regularisationParamT.get() #This param should be set by user
        args['--iter'] = self.numberOfIterations.get()

        self._setSamplingArgs(args)

    def _setCTFArgs(self, args):
        if self.doCTF.get():
           args['--ctf'] = ''

        if self._getInputParticles().isPhaseFlipped():
            args['--ctf_phase_flipped'] = ''

        if self.ignoreCTFUntilFirstPeak.get():
            args['--ctf_intact_first_peak'] = ''

    def _getRelionProgram(self, program='relion_refine'):
        #""" Get the program name depending on the MPI use or not. ""
        if self.numberOfMpi > 1:
             program += '_mpi'
        return program

    def _getInputParticles(self):
        return self.inputParticles.get()

    def _setSamplingArgs(self, args):
        """ Set sampling related params. """
        # Sampling stuff
        if self.doImageAlignment:
            args['--offset_range'] = self.offsetSearchRangePix.get()
            args['--offset_step']  = self.offsetSearchStepPix.get() * 2
            args['--psi_step'] = self.inplaneAngularSamplingDeg.get() * 2

    def _copyAlignAsPriors(self, mdParts, alignType):
        # set priors equal to orig. values
        mdParts.copyColumn(md.RLN_ORIENT_ORIGIN_X_PRIOR, md.RLN_ORIENT_ORIGIN_X)
        mdParts.copyColumn(md.RLN_ORIENT_ORIGIN_Y_PRIOR, md.RLN_ORIENT_ORIGIN_Y)
        mdParts.copyColumn(md.RLN_ORIENT_PSI_PRIOR, md.RLN_ORIENT_PSI)

        if alignType == ALIGN_PROJ:
            mdParts.copyColumn(md.RLN_ORIENT_ROT_PRIOR, md.RLN_ORIENT_ROT)
            mdParts.copyColumn(md.RLN_ORIENT_TILT_PRIOR, md.RLN_ORIENT_TILT)

    def _postprocessParticleRow(self, part, partRow):
        if part.hasAttribute('_rlnGroupName'):
            partRow.setValue(md.RLN_MLMODEL_GROUP_NAME,
                             '%s' % part.getAttributeValue('_rlnGroupName'))
        else:
            partRow.setValue(md.RLN_MLMODEL_GROUP_NAME,
                             '%s' % part.getMicId())

        ctf = part.getCTF()

        if ctf is not None and ctf.getPhaseShift():
            partRow.setValue(md.RLN_CTF_PHASESHIFT, ctf.getPhaseShift())

    def _splitInCTFGroups(self, fnRelion):
        """ Add a new column in the image star to separate the particles
        into ctf groups """

        splitInCTFGroups(fnRelion,
                         self.defocusRange.get(),
                         self.numParticles.get())

    def _fillDataFromIter(self, imgSetOut, mdData):
        imgSetOut.setAlignmentProj()
        imgSetOut.copyItems(self._getInputParticles(),
                            updateItemCallback= self._callBack,
                            itemDataIterator=md.iterRows(mdData,
                                                   sortByLabel=md.RLN_IMAGE_ID))

    def _callBack(self, newItem, row):
        print(row)
        if row.getValue(md.MDL_ENABLED) == -1:
            setattr(newItem, "_appendItem", False)

    def _getInputXmd(self, filename=''):
        if filename == '':
            filename = 'input_particles.xmd'
        return self._getPath(filename)

    def _getParams(self, args):
        params = ' '.join(['%s %s' % (k, str(v)) for k, v in args.items()])
        return params


    def _getNewDim(self):
        tgResol = self.getAttributeValue('targetResolution', 0)
        partSet = self._getInputParticles()
        size = partSet.getXDim()
        nyquist = 2 * partSet.getSamplingRate()

        if tgResol > nyquist:
            newSize = int(round(size * nyquist / tgResol))
            if newSize % 2 == 1:
                newSize += 1
            return newSize
        else:
            return size

    def _scaleImages(self,indx, img):
        fn = img.getFileName()
        index = img.getIndex()
        newFn = self._getTmpPath('particles_subset.mrcs')
        xdim = self._getNewDim()

        ih = ImageHandler()
        image = ih.read((index, fn))
        image.scale(xdim, xdim)

        image.write((indx, newFn))

        img.setFileName(newFn)
        img.setIndex(indx)
        img.setSamplingRate(self._getPixeSize())

    def _getPixeSize(self):
        partSet = self._getInputParticles()
        oldSize = partSet.getXDim()
        newSize  = self._getNewDim()
        pxSize = partSet.getSamplingRate() * oldSize / newSize
        return pxSize

    def _getCl2dLevels(self):
        dirClasses = self.directionalClasses.get()
        return int(math.ceil(math.log(dirClasses) / math.log(2)))

    def _getMetadata(self, file='filepath'):
        fList = file.split("@")
        return md.MetaData(file) if exists(fList[-1]) else md.MetaData()
