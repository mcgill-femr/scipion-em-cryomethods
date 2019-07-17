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
# **************************************************************************

from os.path import join, exists
from pyworkflow.object import Float, String
from pyworkflow.protocol.params import (PointerParam, FloatParam, IntParam,
                                        StringParam, LEVEL_ADVANCED,
                                        BooleanParam,LabelParam, PathParam, EnumParam)
from pyworkflow.em.data import Volume
from pyworkflow.em.protocol import ProtAnalysis3D
from pyworkflow.utils.path import moveFile, makePath, cleanPath, cleanPattern
import cryomethods.convertXmp as convXmp
from cryomethods.convert import writeSetOfParticles, readSetOfParticles,splitInCTFGroups
from pyworkflow.em.metadata.utils import getSize
#
import xmippLib
import math
import numpy as np
import pyworkflow.em as em
import pyworkflow.em.metadata as md

import pyworkflow.object as pwobj

class ProtDirectionalPruning(ProtAnalysis3D):
    """    
    Analyze 2D classes as assigned to the different directions
    """

    _label = 'directional_pruning'

    CL2D = 0
    ML2D = 1
    RL2D = 2




    def __init__(self, *args, **kwargs):
        ProtAnalysis3D.__init__(self, *args, **kwargs)

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input volume",
                      help='Select the input volume.')
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles',
                      label="Input particles",
                      pointerCondition='hasAlignmentProj',
                      help='Select the input projection images with an '
                           'angular assignment.')
        form.addParam('symmetryGroup', StringParam, default='c1',
                      label="Symmetry group",
                      help='See [[Xmipp Symmetry][http://www2.mrc-lmb.cam.ac.uk'
                           '/Xmipp/index.php/Conventions_%26_'
                           'File_formats#Symmetry]] page for a description of '
                           'the symmetry format accepted by Xmipp')
        form.addSection(label='Pruning')
        form.addParam('targetResolution', FloatParam, default=10,
                      label='Target resolution (A)', expertLevel=LEVEL_ADVANCED)
        form.addParam('angularSampling', FloatParam, default=5,
                      label='Angular sampling', expertLevel=LEVEL_ADVANCED,
                      help="In degrees")
        form.addParam('angularDistance', FloatParam, default=10,
                      label='Angular distance', expertLevel=LEVEL_ADVANCED,
                      help="In degrees. An image belongs to a group if its "
                           "distance is smaller than this value")
        form.addParam('maxShift', FloatParam, default=15,
                      label='Maximum shift',

                      expertLevel=LEVEL_ADVANCED,
                      help="In pixels")

        form.addParam('directionalClasses', IntParam, default=2,
                      label='Number of directional classes',
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('classMethod', EnumParam, default=0,
                      label='Choose a method to classify',
                      choices=['CL2D', 'ML2D', 'Relion 2D'],
                      display=EnumParam.DISPLAY_LIST,
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('cl2dIterations', IntParam, default=5,
                      label='Number of CL2D iterations',
                      condition="classMethod==0" ,
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('refineAngles', BooleanParam, default=True,
                      label='Refine angles',
                      condition="classMethod==0",
                      expertLevel=LEVEL_ADVANCED,
                      help="Refine the angles of the classes using a"
                           " continuous angular assignment")

        form.addParam('maxIters', IntParam, default=100,
                      expertLevel=LEVEL_ADVANCED,
                      label='Maximum number of iterations',
                      help='If the convergence has not been reached after '
                           'this number of iterations, the process will be '
                           'stopped.',
                      condition='classMethod==1')
        form.addParam('thresholdValue',FloatParam, default=0.5,
                      expertLevel=LEVEL_ADVANCED,
                      label='Minimum threshold Value',
                      help='Enter a value less than 1(in decimals)')
        form.addParam('noOfParticles',IntParam,default=25,
                      expertLevel=LEVEL_ADVANCED,
                      label='Number of Particles',
                      help='minimum number of particles required to do 2D'
                           'Classification')
        form.addParam('numberOfIterations', IntParam, default=25,
                      label='Number of iterations',
                      condition='classMethod==2',
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
        form.addSection(label='Optimisation')
        form.addParam('regularisationParamT', IntParam,
                      default=2,
                      label='Regularisation parameter T',
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
        form.addParam('copyAlignment', BooleanParam, default=True,
                      label='Consider previous alignment?',
                      condition='classMethod==2',

                      help='If set to Yes, then alignment information from'
                           ' input particles will be considered.')
        form.addParam('alignmentAsPriors', BooleanParam, default=False,
                      condition='classMethod==2',

                      expertLevel=LEVEL_ADVANCED,
                      label='Consider alignment as priors?',
                      help='If set to Yes, then alignment information from '
                           'input particles will be considered as PRIORS. This '
                           'option is mandatory if you want to do local '
                           'searches')
        form.addParam('fillRandomSubset', BooleanParam, default=False,
                      condition='classMethod==2',

                      expertLevel=LEVEL_ADVANCED,
                      label='Consider random subset value?',
                      help='If set to Yes, then random subset value '
                           'of input particles will be put into the'
                           'star file that is generated.')
        form.addParam('maskDiameterA', IntParam, default=-1,
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
        form.addParam('referenceClassification' , BooleanParam, default=True,
                       condition='classMethod==2',
                       label='Perform reference based classification?')
        form.addSection(label='Sampling')
        form.addParam('doImageAlignment', BooleanParam, default=True,
                      label='Perform Image Alignment?',
                      condition='classMethod==2',
                      )
        form.addParam('inplaneAngularSamplingDeg', FloatParam, default=5,
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
        form.addParam('offsetSearchRangePix', FloatParam, default=5,

                      condition='classMethod==2 and doImageAlignment',
                      label='Offset search range (pix)',
                      help='Probabilities will be calculated only for '
                           'translations in a circle with this radius (in '
                           'pixels). The center of this circle changes at '
                           'every iteration and is placed at the optimal '
                           'translation for each image in the previous '
                           'iteration.')
        form.addParam('offsetSearchStepPix', FloatParam, default=1.0,

                      condition='classMethod==2 and doImageAlignment',
                      label='Offset search step (pix)',
                      help='Translations will be sampled with this step-size '
                           '(in pixels). Translational sampling is also done '
                           'using the adaptive approach. Therefore, if '
                           'adaptive=1, the translations will first be '
                           'evaluated on a 2x coarser grid.')
        form.addSection(label='Compute')
        form.addParam('allParticlesRam', BooleanParam, default=False,
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
        form.addParam('scratchDir', PathParam,

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
        form.addParam('combineItersDisc', BooleanParam, default=False,
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
        form.addParam('doGpu', BooleanParam, default=True,
                      label='Use GPU acceleration?',
                      condition='classMethod==2',
                      help='If set to Yes, the job will try to use GPU '
                           'acceleration.')
        form.addParam('gpusToUse', StringParam, default='',
                      label='Which GPUs to use:',
                      condition='classMethod==2 and doGpu',
                      help='This argument is not necessary. If left empty, '
                           'the job itself will try to allocate available '
                           'GPU resources. You can override the default '
                           'allocation by providing a list of which GPUs '
                           '(0,1,2,3, etc) to use. MPI-processes are '
                           'separated by ":", threads by ",". '
                           'For example: "0,0:1,1:0,0:1,1"')
        form.addParam('useParallelDisk', BooleanParam, default=True,
                          label='Use parallel disc I/O?',
                          condition='classMethod==2',
                          help='If set to Yes, all MPI slaves will read '
                               'their own images from disc. Otherwise, only '
                               'the master will read images and send them '
                               'through the network to the slaves. Parallel '
                               'file systems like gluster of fhgfs are good '
                               'at parallel disc I/O. NFS may break with many '
                               'slaves reading in parallel.')
        form.addParam('pooledParticles', IntParam, default=3,
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
        form.addSection(label='CTF')
        form.addParam('continueMsg', LabelParam, default=True,

                      condition='classMethod==2',
                      label='CTF parameters are not available in continue mode')
        form.addParam('doCTF', BooleanParam, default=True,
                      label='Do CTF-correction?', condition='classMethod==2',
                      help='If set to Yes, CTFs will be corrected inside the '
                           'MAP refinement. The resulting algorithm '
                           'intrinsically implements the optimal linear, or '
                           'Wiener filter. Note that input particles should '
                           'contains CTF parameters.')
        form.addParam('hasReferenceCTFCorrected', BooleanParam, default=False,
                      condition='classMethod==2',
                      label='Has reference been CTF-corrected?',
                      help='Set this option to Yes if the reference map '
                           'represents CTF-unaffected density, e.g. it was '
                           'created using Wiener filtering inside RELION or '
                           'from a PDB. If set to No, then in the first '
                           'iteration, the Fourier transforms of the reference '
                           'projections are not multiplied by the CTFs.')

        form.addParam('haveDataBeenPhaseFlipped', LabelParam,

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
        form.addParam('ignoreCTFUntilFirstPeak', BooleanParam, default=False,
                      expertLevel=LEVEL_ADVANCED,
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
        form.addParam('doCtfManualGroups', BooleanParam, default=False,
                      label='Do manual grouping ctfs?',

                      condition='classMethod==2',
                      help='Set this to Yes the CTFs will grouping manually.')
        form.addParam('defocusRange', FloatParam, default=1000,
                      label='defocus range for group creation (in Angstroms)',

                      condition='classMethod==2 and doCtfManualGroups',
                      help='Particles will be grouped by defocus.'
                           'This parameter is the bin for an histogram.'
                           'All particles assigned to a bin form a group')
        form.addParam('numParticles', FloatParam, default=10,
                      label='minimum size for defocus group',

                      condition='classMethod==2 and doCtfManualGroups',
                      help='If defocus group is smaller than this value, '
                           'it will be expanded until number of particles '
                           'per defocus group is reached')




        form.addParallelSection(threads=1, mpi=3)




    ###TODO LIST:
    #1) ML number of iterations
    #2) Input parameter to determine minimun number of particles to do or not to do 2D classification: line 223, 25 number

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
        if self.refineAngles:
            self._insertFunctionStep('refineAnglesStep')
        self._insertFunctionStep('cleanStep')
        self._insertFunctionStep('createOutputStep',1)




    #--------------------------- STEPS functions -------------------------------
    def convertInputStep(self,inputParticles,inputVolume, targetResolution):
        """ Write the input images as a Xmipp metadata file. 
        particlesId: is only need to detect changes in
        input particles and cause restart from here.
        """
        print('SWATHI')
        print(self._getPath('input_particles.xmd'))
        imgSet = self.inputParticles.get()

        convXmp.writeSetOfParticles(imgSet, self._getPath(
            'input_particles.xmd'))

        Xdim = self.inputParticles.get().getDimensions()[0]
        Ts = self.inputParticles.get().getSamplingRate()
        newTs = self.targetResolution.get()*0.4
        newTs = max(Ts,newTs)
        newXdim = Xdim*Ts/newTs
        self.runJob("xmipp_image_resize",
                    "-i %s -o %s --save_metadata_stack %s --dim %d"% \
                    (self._getPath('input_particles.xmd'),
                     self._getExtraPath('scaled_particles.stk'),
                     self._getExtraPath('scaled_particles.xmd'),
                     newXdim))

        from pyworkflow.em.convert import ImageHandler
        img = ImageHandler()
        img.convert(self.inputVolume.get(), self._getExtraPath("volume.vol"))
        Xdim = self.inputVolume.get().getDim()[0]
        if Xdim!=newXdim:
            self.runJob("xmipp_image_resize","-i %s --dim %d"% \
                        (self._getExtraPath("volume.vol"),
                         newXdim), numberOfMpi=1)




    def constructGroupsStep(self, partId, angularSampling,
                            angularDistance, symmetryGroup):
        # Generate projections from this reconstruction
        params = {"inputVol" : self._getExtraPath("volume.vol"),
                  "galleryStk" : self._getExtraPath("gallery.stk"),
                  "galleryXmd" : self._getExtraPath("gallery.doc"),
                  "neighborhoods": self._getExtraPath("neighbours.xmd"),
                  "symmetry" : self.symmetryGroup.get(),
                  "angularSampling" : self.angularSampling.get(),
                  "angularDistance" : self.angularDistance.get(),
                  "expParticles" : self._getExtraPath('scaled_particles.xmd')
                  }

        args = '-i %(inputVol)s -o %(galleryStk)s ' \
               '--sampling_rate %(angularSampling)f --sym %(symmetry)s'
        args += ' --method fourier 1 0.25 bspline --compute_neighbors ' \
                '--angular_distance %(angularSampling)f'
        args += ' --experimental_images %(expParticles)s --max_tilt_angle 90'

        self.runJob("xmipp_angular_project_library", args % params)

        args = ('--i1 %(expParticles)s --i2 %(galleryXmd)s '
                '-o %(neighborhoods)s --dist %(angularDistance)f '
                '--sym %(symmetry)s --check_mirrors')
        self.runJob("xmipp_angular_neighbourhood", args % params, numberOfMpi=1)



    def classifyGroupsStep(self):
        mdOut = xmippLib.MetaData()
        mdClasses = xmippLib.MetaData()
        mdClassesParticles=xmippLib.MetaData()
        mdData=xmippLib.MetaData()
        mdClassesClass=xmippLib.MetaData()
        mdBlocks=xmippLib.MetaData()
        mdCount=xmippLib.MetaData()
        mdClassCount=xmippLib.MetaData()

        fnClassParticles = self._getPath('input_particles.xmd')
        fnPrunedParticles = self._getPath('output_particles_pruned.xmd')
        mdClassesParticles.read(fnClassParticles)



        fnNeighbours = self._getExtraPath("neighbours.xmd")
        fnGallery = self._getExtraPath("gallery.stk")
        nop=self.noOfParticles.get()
        for block in xmippLib.getBlocksInMetaDataFile( fnNeighbours):
            imgNo = block.split("_")[1]
            fnDir = self._getExtraPath("direction_%s" % imgNo)

            if not exists(fnDir):
                makePath(fnDir)


            if self.classMethod.get() == self.CL2D:
                Nlevels = int(math.ceil(math.log(self.directionalClasses.get())
                                        /math.log(2)))
                fnOut = join(fnDir, "level_%02d/class_classes.stk"%Nlevels)
                if not exists(fnOut):
                    fnBlock = "%s@%s"%(block,fnNeighbours)
                    if getSize(fnBlock) > nop:
                        try:
                            args = "-i %s --odir %s --ref0 %s@%s --iter %d " \
                                   "--nref %d --distance correlation " \
                                   "--classicalMultiref --maxShift %d" % \
                                   (fnBlock,fnDir,imgNo,fnGallery,
                                    self.cl2dIterations.get(),
                                    self.directionalClasses.get(),
                                    self.maxShift.get())
                            self.runJob("xmipp_classify_CL2D",args)
                            fnAlignRoot = join(fnDir,"classes")
                            self.runJob("xmipp_image_align",
                                        "-i %s --ref %s@%s --oroot %s --iter 1"%
                                        (fnOut,imgNo,fnGallery,fnAlignRoot),
                                        numberOfMpi=1)
                            self.runJob("xmipp_transform_geometry",
                                        "-i %s_alignment.xmd --apply_transform" %
                                        fnAlignRoot, numberOfMpi=1)


                            # Construct output metadata
                            if exists(fnOut):

                                fnClassCount= join(fnDir, "level_%02d//class_classes.xmd"%Nlevels)
                                print(fnClassCount)
                                mdCount.read(fnClassCount)
                                CC=[]
                                out=[]


                                for i in range(self.directionalClasses.get()):


                                    CC.append(
                                        mdCount.getValue(xmippLib.MDL_CLASS_COUNT,
                                                           i + 1))

                                    objId = mdOut.addObject()
                                    mdOut.setValue(xmippLib.MDL_REF, int(imgNo),
                                                   objId)
                                    mdOut.setValue(xmippLib.MDL_IMAGE,
                                                   "%d@%s" % (i + 1, fnOut),
                                                   objId)

                                n = sum(CC)
                                out = np.true_divide(CC ,n)
                                lowest = out < self.thresholdValue.get()
                                itemIdInput=[]
                                for objId in mdClassesParticles:

                                    itemIdInput.append(
                                        mdClassesParticles.getValue(
                                            xmippLib.MDL_ITEM_ID, objId))

                                for indx, block in enumerate(
                                        xmippLib.getBlocksInMetaDataFile(
                                                fnClassCount)[2:]):
                                    if lowest[indx]:

                                        fnBlock = '%s@%s' % (block, fnClassCount)
                                        mdClassCount.read(fnBlock)
                                        for objId in mdClassCount:

                                            itemIdPart = mdClassCount.getValue(
                                                xmippLib.MDL_ITEM_ID, objId)
                                            idx = itemIdInput.index(
                                                itemIdPart) + 1
                                            mdClassesParticles.setValue(
                                                xmippLib.MDL_ENABLED, -1, idx)


                        except:
                            print(
                                "The classification failed, "
                                "probably because of a low number of images.")
                            print(
                                "However, this classification does not "
                                "hinder the protocol to continue")

                            #fnDirectional = self._getPath("directionalClasses.xmd")
                            #mdOut.write(fnDirectional)

                            #self.runJob("xmipp_metadata_utilities", "-i %s --set join %s ref" %
                            #       (fnDirectional, self._getExtraPath("gallery.doc")),
                            #         numberOfMpi=1)

            elif self.classMethod.get() == self.ML2D:
                fnOut = join(fnDir,"class_")
                fnBlock = "%s@%s"% (block,fnNeighbours)
                if getSize(fnBlock) > nop:
                    try:
                        params="-i %s --oroot %s --nref %d --fast --mirror --iter %d" \
                                %(fnBlock,
                                 fnOut,
                                  self.directionalClasses.get(),
                                  self.maxIters.get())

                        self.runJob("xmipp_ml_align2d",params)

                        fnOrig = join(fnDir,"class_classes.stk")
                        fnAlign = join(fnDir,"class_align")

                        self.runJob("xmipp_image_align",
                                    "-i %s --ref %s@%s --oroot %s --iter 1" %
                                    (fnOrig, imgNo,fnGallery, fnAlign),
                                    numberOfMpi=1)

                        self.runJob("xmipp_transform_geometry",
                                    "-i %s_alignment.xmd --apply_transform" %
                                    fnAlign, numberOfMpi=1)

                        if exists(fnDir):
                            fnClassClasses = fnDir+'/class_classes.xmd'

                            mdClasses.read(fnClassClasses)
                            WeightsArray = []


                            for i in range(self.directionalClasses.get()):

                                WeightsArray.append(mdClasses.getValue(xmippLib.MDL_WEIGHT, i+1))
                                n= sum(WeightsArray)
                                out=np.divide(WeightsArray, n)
                                lowest= out < self.thresholdValue.get()

                                objId = mdOut.addObject()
                                mdOut.setValue(xmippLib.MDL_REF, int(imgNo),
                                                objId)
                                mdOut.setValue(xmippLib.MDL_IMAGE,
                                                "%d@%s" % (i + 1, fnOrig), objId)


                            itemIdInput=[]


                            for objId in mdClassesParticles:
                                itemIdInput.append(mdClassesParticles.getValue(xmippLib.MDL_ITEM_ID, objId))

                            for indx, block in enumerate(xmippLib.getBlocksInMetaDataFile(fnClassClasses)[2:]):
                                if lowest[indx]:
                                    fnBlock = '%s@%s'% (block, fnClassClasses)
                                    mdClassesClass.read(fnBlock)
                                    for objId in mdClassesClass:
                                        itemIdPart = mdClassesClass.getValue(xmippLib.MDL_ITEM_ID, objId)
                                        idx= itemIdInput.index(itemIdPart)+1
                                        mdClassesParticles.setValue(xmippLib.MDL_ENABLED, -1, idx)


                    except:
                        print("The classification failed,"
                                "probably because of a low number of images.")
                        print("However, this classification"
                               "does not hinder the protocol to continue")

                        #fnDirectional = self._getPath("directionalClasses.xmd")
                        #mdOut.write(fnDirectional)
                        #self.runJob("xmipp_metadata_utilities", "-i %s --set join %s ref" %
                        #   (fnDirectional, self._getExtraPath("gallery.doc")),
                        #     numberOfMpi=1)


            else:



                relPart=self._createSetOfParticles()
                relPart.copyInfo(self.inputParticles.get())

                fnRelion = self._getExtraPath('relion_%s.star'% imgNo)

                fnBlock = "%s@%s" % (block, fnNeighbours)
                fnRef = "%s@%s" % (imgNo, fnGallery)




                if getSize(fnBlock) > nop:
                    try:
                        convXmp.readSetOfParticles(fnBlock, relPart)


                        if self.copyAlignment.get():
                            alignType = relPart.getAlignment()
                            alignType != em.ALIGN_NONE
                        else:
                            alignType = em.ALIGN_NONE

                        alignToPrior =  getattr(self, 'alignmentAsPriors',
                                                                True)
                        fillRandomSubset =  getattr(self, 'fillRandomSubset', False)

                        writeSetOfParticles(relPart,fnRelion,self._getExtraPath(),
                                                     alignType=alignType,
                                                     postprocessImageRow=self._postprocessParticleRow,
                                                     fillRandomSubset=fillRandomSubset)

                        if alignToPrior:
                            mdParts = md.MetaData(fnRelion)
                            self._copyAlignAsPriors(mdParts, alignType)
                            mdParts.write(fnRelion)
                        if self.doCtfManualGroups:
                            self._splitInCTFGroups(fnRelion)

                        fnOut = join(fnDir, "class_")
                        print("SAAAA", fnOut)
                        args = {}
                        self._setNormalArgs(args)
                        args['--i'] = fnRelion
                        args['--o'] = fnOut
                        if self.referenceClassification.get():
                            args['--ref']= fnRef
                        self._setComputeArgs(args)

                        params = ' '.join(['%s %s' % (k, str(v)) for k, v in args.iteritems()])

                        self.runJob(self._getRelionProgram(), params)

                        Rcd=[]

                        it=self.numberOfIterations.get()
                        if it < 10:
                           model='_it00%d_'%it
                        else:
                            model = '_it0%d_' % it

                        fnModel= (fnOut +model+'model.star')

                        block=md.getBlocksInMetaDataFile(fnModel)[1]
                        fnBlock = "%s@%s" % (block, fnModel)

                        mdBlocks.read(fnBlock)
                        fnData = (fnOut + model + 'data.star')
                        fnClass= (fnOut + model + 'classes.mrcs')

                        for objId in mdBlocks:
                           ClsDist=mdBlocks.getValue(xmippLib.RLN_MLMODEL_PDF_CLASS,objId)

                           Rcd.append(ClsDist)
                        w=[]
                        for x in Rcd:
                            if  x < self.thresholdValue.get():

                                w.append(Rcd.index(x) + 1)
                        print(w)

                        ImageId = []
                        mdData.read(fnData)
                        itemIdInput = []

                        for objId in mdClassesParticles:
                            itemIdInput.append(
                                mdClassesParticles.getValue(xmippLib.MDL_ITEM_ID,
                                                            objId))


                        for x in w[:]:

                            for objId in mdData:
                                ClsNr = mdData.getValue(xmippLib.RLN_PARTICLE_CLASS, objId)

                                if x == ClsNr:

                                    ImageId =mdData.getValue(xmippLib.RLN_IMAGE_ID,objId)

                                    idx = itemIdInput.index(ImageId)+1

                                    mdClassesParticles.setValue(xmippLib.MDL_ENABLED, -1,
                                                                   idx)






                        if exists(fnOut):
                            for i in range(self.directionalClasses.get()):
                                objId = mdOut.addObject()
                                mdOut.setValue(xmippLib.MDL_REF, int(imgNo),
                                               objId)
                                mdOut.setValue(xmippLib.MDL_IMAGE,
                                               "%d@%s" % (i + 1, fnClass), objId)
                    except:
                        print("The classification failed,"
                              "probably because of a low number of images.")
                        print("However, this classification"
                              "does not hinder the protocol to continue")
                        #fnDirectional = self._getPath("directionalClasses.xmd")
                        #mdOut.write(fnDirectional)
                        #self.runJob("xmipp_metadata_utilities", "-i %s --set join %s ref" %
                        #           (fnDirectional, self._getExtraPath("gallery.doc")),
                        #          numberOfMpi=1)
        # print ("Size before remove disabled", mdClassesParticles.size())
        # mdClassesParticles.removeDisabled()
        # print ("Size afte remove disabled", mdClassesParticles.size())
        mdClassesParticles.write(fnPrunedParticles)











    def refineAnglesStep(self):
      if self.classMethod.get() == self.CL2D:
          pass
            #fnDirectional = self._getPath("directionalClasses.xmd")
            #newTs = self.targetResolution.get()*0.4
            #self.runJob("xmipp_angular_continuous_assign2","-i %s --ref %s "
            #                                          "--max_resolution %f "
            #                                         "--sampling %f "
            #                                        "--optimizeAngles "
            #                                       "--optimizeShift"% \
            #   (fnDirectional,self._getExtraPath("volume.vol"),
            #   self.targetResolution.get(),newTs))





    def cleanStep(self):
        pass

        #cleanPath(self._getExtraPath('scaled_particles.stk'))
        #cleanPath(self._getExtraPath('scaled_particles.xmd'))
        #cleanPath(self._getExtraPath('volume.vol'))
        #cleanPattern(self._getExtraPath("direction_*/level_00"))

    def createOutputStep(self, numeroFeo):
        fnDirectional= self._getPath("directionalClasses.xmd")
        fnPrunedParticles = self._getPath('output_particles_pruned.xmd')



        if exists(fnDirectional):
            imgSetOut = self._createSetOfParticles()
            imgSetOut.setSamplingRate(imgSetOut.getSamplingRate())
            imgSetOut.setAlignmentProj()
            convXmp.readSetOfParticles(fnDirectional,imgSetOut)
            print(fnDirectional)
            self._defineOutputs(outputParticles=imgSetOut)
            self._defineSourceRelation(self.inputParticles,imgSetOut)
            self._defineSourceRelation(self.inputVolume, imgSetOut)
        else:
            imgSetOut = self._createSetOfParticles()
            imgSetOut.copyInfo(self.inputParticles.get())

            imgSetOut.setSamplingRate(imgSetOut.getSamplingRate())

            #imgSetOut.setAlignmentProj()
            #readSetOfParticles(fnPrunedParticles, imgSetOut)
            self._fillDataFromIter(imgSetOut)

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

        if alignType == em.ALIGN_PROJ:
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

    def _fillDataFromIter(self, imgSetOut):
        imgSetOut.setAlignmentProj()
        fnPrunedParticles = self._getPath('output_particles_pruned.xmd')
        imgSetOut.copyItems(self._getInputParticles(),
                            updateItemCallback= self._callBack,
                            itemDataIterator=md.iterRows(fnPrunedParticles,
                                                         sortByLabel=md.RLN_IMAGE_ID))
    def _callBack(self, newItem, row):
        if row.getValue(xmippLib.MDL_ENABLED) == -1:
            setattr(newItem, "_appendItem", False)