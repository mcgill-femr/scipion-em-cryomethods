# **************************************************************************
# *
# * Authors:         Javier Vargas (jvargas@cnb.csic.es) (2016)
# *                  Swathi Adinarayanan(swathi.adinarayanan@mail.mcgill.ca)
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
from pyworkflow.protocol.params import (PointerParam, FloatParam, EnumParam,
                                        StringParam, BooleanParam, IntParam,
                                        LabelParam, PathParam,LEVEL_ADVANCED)
import pyworkflow.em.metadata as md
from cryomethods.protocols import ProtDirectionalPruning
from pyworkflow.utils.path import cleanPath,makePath,cleanPattern
from cryomethods.convert import writeSetOfParticles,splitInCTFGroups,rowToAlignment
from pyworkflow.em.metadata.utils import getSize
import xmippLib
import math
import numpy as np
from cryomethods.convert import loadMrc, saveMrc
from cryomethods import Plugin
from pyworkflow.em.convert import ImageHandler
import random
import pyworkflow.em as em
from os.path import join, exists
import cryomethods.convertXmp as convXmp
from glob import glob
from itertools import *
import collections

class ProtClass3DRansac(ProtDirectionalPruning):
    """
    Performs 3D classification of input particles with previous alignment
    """
    _label = 'directional_ransac'

    CL2D = 0
    ML2D = 1
    RL2D = 2
    KM = 0
    AP = 1

    def __init__(self, *args, **kwargs):
        ProtDirectionalPruning.__init__(self, *args, **kwargs)

    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):

        form.addSection(label='Input')
        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input volume",
                      help='Select the input volume.')
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles', pointerCondition='hasAlignment',
                      label="Input particles", important=True,
                      help='Select the input projection images.')
        form.addParam('backRadius', IntParam, default=-1,
                      label='Mask radius',
                      help='Pixels outside this circle are assumed to be noise')
        form.addParam('targetResolution', FloatParam, default=10, label='Target resolution (A)', expertLevel=LEVEL_ADVANCED,
                      help='Expected Resolution of the initial 3D classes obtained by the 2D classes. You should have a good' 
                      'reason to modify the 10 A value')

        form.addParam('symmetryGroup', StringParam, default='c1',
                      label="Symmetry group",
                      help='See [[Xmipp Symmetry][http://www2.mrc-lmb.cam.ac.uk/Xmipp/index.php/Conventions_%26_File_formats#Symmetry]] page '
                           'for a description of the symmetry format accepted by Xmipp')

        form.addSection(label='Directional Classes')

        form.addParam('angularSampling', FloatParam, default=5, label='Angular sampling', expertLevel=LEVEL_ADVANCED, help="In degrees")
        form.addParam('angularDistance', FloatParam, default=10, label='Angular distance', expertLevel=LEVEL_ADVANCED,
                      help="In degrees. An image belongs to a group if its distance is smaller than this value")
        form.addParam('noOfParticles', IntParam, default=25,
                      expertLevel=LEVEL_ADVANCED,
                      label='Number of Particles',
                      help='minimum number of particles required to do 2D'
                           'Classification')
        form.addParam('directionalClasses', IntParam, default=2,
                      label='Number of 2D classes in per directions',
                      expertLevel=LEVEL_ADVANCED)
        groupClass2D = form.addSection(label='2D Classification')
        groupClass2D.addParam('Class2D', EnumParam,
                              choices=['ML2D','CL2D','RL2D'], default=2,
                              label="2D classification method",
                              display=EnumParam.DISPLAY_COMBO,
                              help='2D classification algorithm used to be '
                                   'applied to the directional classes.')

        groupClass2D.addParam('CL2D_it', IntParam, default=20, condition='Class2D == 0',
                     label='number of iterations',
                     help='This is the radius (in pixels) of the spherical mask ')

        groupClass2D.addParam('CL2D_shift', IntParam, default=5, condition='Class2D == 0',
                     label='Maximum allowed shift',
                     help='Maximum allowed shift ')
        groupClass2D.addParam('maxIters', IntParam, default=100,
                      label='Maximum number of iterations',
                      help='If the convergence has not been reached after '
                           'this number of iterations, the process will be '
                           'stopped.',
                      condition='Class2D==1')
        form.addParam('numberOfIterations', IntParam, default=25,
                      label='Number of iterations',
                      condition='Class2D==2',
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
        form.addParam('randomVolume', IntParam, default=5,
                      label='Number of random volume',
                      help="Number of random volume to be genrated.This number"
                           " of volume will be consider for PCA")
        form.addSection(label='Optimisation')
        form.addParam('regularisationParamT', IntParam,
                      default=2,
                      label='Regularisation parameter T',
                      condition='Class2D==2',
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
                      condition='Class2D==2',

                      help='If set to Yes, then alignment information from'
                           ' input particles will be considered.')
        form.addParam('alignmentAsPriors', BooleanParam, default=False,
                      condition='Class2D==2',
                      label='Consider alignment as priors?',
                      help='If set to Yes, then alignment information from '
                           'input particles will be considered as PRIORS. This '
                           'option is mandatory if you want to do local '
                           'searches')
        form.addParam('fillRandomSubset', BooleanParam, default=False,
                      condition='Class2D==2',
                      label='Consider random subset value?',
                      help='If set to Yes, then random subset value '
                           'of input particles will be put into the'
                           'star file that is generated.')
        form.addParam('maskDiameterA', IntParam, default=-1,
                      condition='Class2D==2',
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
        form.addParam('referenceClassification', BooleanParam, default=True,
                      condition='Class2D==2',
                      label='Perform reference based classification?')
        form.addSection(label='Sampling')
        form.addParam('doImageAlignment', BooleanParam, default=True,
                      label='Perform Image Alignment?',
                      condition='Class2D==2',
                      )
        form.addParam('inplaneAngularSamplingDeg', FloatParam, default=5,
                      label='In-plane angular sampling (deg)',
                      condition='Class2D==2 and doImageAlignment',

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

                      condition='Class2D==2 and doImageAlignment',
                      label='Offset search range (pix)',
                      help='Probabilities will be calculated only for '
                           'translations in a circle with this radius (in '
                           'pixels). The center of this circle changes at '
                           'every iteration and is placed at the optimal '
                           'translation for each image in the previous '
                           'iteration.')
        form.addParam('offsetSearchStepPix', FloatParam, default=1.0,

                      condition='Class2D==2 and doImageAlignment',
                      label='Offset search step (pix)',
                      help='Translations will be sampled with this step-size '
                           '(in pixels). Translational sampling is also done '
                           'using the adaptive approach. Therefore, if '
                           'adaptive=1, the translations will first be '
                           'evaluated on a 2x coarser grid.')
        form.addSection(label='Compute')
        form.addParam('allParticlesRam', BooleanParam, default=False,
                      label='Pre-read all particles into RAM?',
                      condition='Class2D==2',
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

                      condition='Class2D==2 and not allParticlesRam',
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
                      condition='Class2D==2',
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
                      condition='Class2D==2',
                      help='If set to Yes, the job will try to use GPU '
                           'acceleration.')
        form.addParam('gpusToUse', StringParam, default='',
                      label='Which GPUs to use:',
                      condition='Class2D==2 and doGpu',
                      help='This argument is not necessary. If left empty, '
                           'the job itself will try to allocate available '
                           'GPU resources. You can override the default '
                           'allocation by providing a list of which GPUs '
                           '(0,1,2,3, etc) to use. MPI-processes are '
                           'separated by ":", threads by ",". '
                           'For example: "0,0:1,1:0,0:1,1"')
        form.addParam('useParallelDisk', BooleanParam, default=True,
                      label='Use parallel disc I/O?',
                      condition='Class2D==2',
                      help='If set to Yes, all MPI slaves will read '
                           'their own images from disc. Otherwise, only '
                           'the master will read images and send them '
                           'through the network to the slaves. Parallel '
                           'file systems like gluster of fhgfs are good '
                           'at parallel disc I/O. NFS may break with many '
                           'slaves reading in parallel.')
        form.addParam('pooledParticles', IntParam, default=3,
                      label='Number of pooled particles:',
                      condition='Class2D==2',
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

                      condition='Class2D==2',
                      label='CTF parameters are not available in continue mode')
        form.addParam('doCTF', BooleanParam, default=True,
                      label='Do CTF-correction?', condition='Class2D==2',
                      help='If set to Yes, CTFs will be corrected inside the '
                           'MAP refinement. The resulting algorithm '
                           'intrinsically implements the optimal linear, or '
                           'Wiener filter. Note that input particles should '
                           'contains CTF parameters.')
        form.addParam('hasReferenceCTFCorrected', BooleanParam, default=False,
                      condition='Class2D==2',
                      label='Has reference been CTF-corrected?',
                      help='Set this option to Yes if the reference map '
                           'represents CTF-unaffected density, e.g. it was '
                           'created using Wiener filtering inside RELION or '
                           'from a PDB. If set to No, then in the first '
                           'iteration, the Fourier transforms of the reference '
                           'projections are not multiplied by the CTFs.')

        form.addParam('haveDataBeenPhaseFlipped', LabelParam,

                      condition='Class2D==2',
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

                      condition='Class2D==2',
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

                      condition='Class2D==2',
                      help='Set this to Yes the CTFs will grouping manually.')
        form.addParam('defocusRange', FloatParam, default=1000,
                      label='defocus range for group creation (in Angstroms)',

                      condition='Class2D==2 and doCtfManualGroups',
                      help='Particles will be grouped by defocus.'
                           'This parameter is the bin for an histogram.'
                           'All particles assigned to a bin form a group')
        form.addParam('numParticles', FloatParam, default=10,
                      label='minimum size for defocus group',

                      condition='Class2D==2 and doCtfManualGroups',
                      help='If defocus group is smaller than this value, '
                           'it will be expanded until number of particles '
                           'per defocus group is reached')
        form.addSection(label='Clustering')
        groupClass2D.addParam('ClusteringMethod', EnumParam,
                              choices=['Kmeans','AffinityPropagation'], default=0,
                              label="clustering method",
                              display=EnumParam.DISPLAY_COMBO,
                              help='Select a method to cluster the data. \n ')
        form.addParam('clusterCentres', IntParam, default =5,
                      label = 'Number of Cluster centres',
                      condition = 'ClusteringMethod==KM',
                      help =' This value corresponds to the final number of 3D volumes '
                            'to be reconstructed')
        form.addParallelSection(threads=1, mpi=1)

    def _insertAllSteps(self):
        particlesId = self.inputParticles.get().getObjId()
        volId = self.inputVolume.get().getObjId()
        convertId = self._insertFunctionStep('convertInputStep', particlesId,
                                             volId, self.targetResolution.get())
        self._insertFunctionStep('constructGroupsStep', particlesId,
                                 self.angularSampling.get(),
                                 self.angularDistance.get(),
                                 self.symmetryGroup.get())
        self._insertFunctionStep('classify2DStep')
        self._insertFunctionStep('randomSelectionStep')
        self._insertFunctionStep('reconstruct3DStep')
        self._insertFunctionStep('pcaStep')
        self._insertFunctionStep('cleanStep')
        self._insertFunctionStep('createOutputStep')


        
    def convertInputStep(self, particlesId, volId, targetResolution):
        ProtDirectionalPruning.convertInputStep(self, particlesId,
                                                   volId,targetResolution)


    def constructGroupsStep(self, particlesId, angularSampling, angularDistance, symmetryGroup):
       ProtDirectionalPruning.constructGroupsStep(self, particlesId, angularSampling, angularDistance, symmetryGroup)
        
    def classify2DStep(self):
        mdClassesParticles = xmippLib.MetaData()
        fnClassParticles = self._getPath('input_particles.xmd')
        mdClassesParticles.read(fnClassParticles)
        fnNeighbours = self._getExtraPath("neighbours.xmd")
        fnGallery = self._getExtraPath("gallery.stk")
        nop = self.noOfParticles.get()
        fnDirectional = self._getPath("directionalClasses.xmd")
        mdOut = xmippLib.MetaData()
        mdRef = xmippLib.MetaData(self._getExtraPath("gallery.doc"))
        for block in xmippLib.getBlocksInMetaDataFile(fnNeighbours):
            imgNo = block.split("_")[1]
            galleryImgNo = int(block.split("_")[1])

            fnDir = self._getExtraPath("direction_%s" % imgNo)
            rot = mdRef.getValue(xmippLib.MDL_ANGLE_ROT,galleryImgNo)
            tilt = mdRef.getValue(xmippLib.MDL_ANGLE_TILT,galleryImgNo )
            psi = 0.0
            if not exists(fnDir):
                makePath(fnDir)
            if self.Class2D.get() == self.CL2D:
                Nlevels = int(math.ceil(math.log(self.directionalClasses.get())
                                        / math.log(2)))
                fnOut = join(fnDir, "level_%02d/class_classes.stk" % Nlevels)
                if not exists(fnOut):
                    fnBlock = "%s@%s" % (block, fnNeighbours)
                    if getSize(fnBlock) > nop:
                        args = "-i %s --odir %s --ref0 %s@%s --iter %d " \
                                   "--nref %d --distance correlation " \
                                   "--classicalMultiref --maxShift %d" % \
                                   (fnBlock, fnDir, imgNo, fnGallery,
                                    self.CL2D_it.get(),
                                    self.directionalClasses.get(),
                                    self.CL2D_shift.get())
                        self.runJob("xmipp_classify_CL2D", args)
                        fnAlignRoot = join(fnDir, "classes")
                        fnOut = join(fnDir, "level_%02d/class_classes.stk" % (
                                self.directionalClasses.get() - 1))
                        for n in range(self.directionalClasses.get()):
                            objId = mdOut.addObject()
                            mdOut.setValue(xmippLib.MDL_REF,int(imgNo), objId)
                            mdOut.setValue(xmippLib.MDL_IMAGE,
                                           "%d@%s" % (n + 1, fnOut), objId)
                            mdOut.setValue(xmippLib.MDL_IMAGE_IDX, long(n + 1),
                                           objId)
                            mdOut.setValue(xmippLib.MDL_ANGLE_ROT, rot, objId)
                            mdOut.setValue(xmippLib.MDL_ANGLE_TILT, tilt, objId)
                            mdOut.setValue(xmippLib.MDL_ANGLE_PSI, psi, objId)
                            mdOut.setValue(xmippLib.MDL_SHIFT_X, 0.0, objId)
                            mdOut.setValue(xmippLib.MDL_SHIFT_Y, 0.0, objId)
                            mdOut.write("%s@%s" % (block, fnDirectional),
                                        xmippLib.MD_APPEND)
                        mdOut.clear()

            elif self.Class2D.get() == self.ML2D:
                fnOut = join(fnDir, "class_")
                fnBlock = "%s@%s" % (block, fnNeighbours)
                if getSize(fnBlock) > nop:
                        params = "-i %s --oroot %s --nref %d --fast --mirror --iter %d" \
                                 % (fnBlock,
                                    fnOut,
                                    self.directionalClasses.get(),
                                    self.maxIters.get())

                        self.runJob("xmipp_ml_align2d", params)
                        fnOut = self._getExtraPath(
                            "direction_%s/class_classes.stk" % imgNo)
                        for n in range(self.directionalClasses.get()):
                            objId = mdOut.addObject()
                            mdOut.setValue(xmippLib.MDL_REF,int(imgNo), objId)
                            mdOut.setValue(xmippLib.MDL_IMAGE,
                                           "%d@%s" % (n + 1, fnOut), objId)
                            mdOut.setValue(xmippLib.MDL_IMAGE_IDX, long(n + 1),
                                           objId)
                            mdOut.setValue(xmippLib.MDL_ANGLE_ROT, rot, objId)
                            mdOut.setValue(xmippLib.MDL_ANGLE_TILT, tilt, objId)
                            mdOut.setValue(xmippLib.MDL_ANGLE_PSI, psi, objId)
                            mdOut.setValue(xmippLib.MDL_SHIFT_X, 0.0, objId)
                            mdOut.setValue(xmippLib.MDL_SHIFT_Y, 0.0, objId)
                            mdOut.write("%s@%s" % (block, fnDirectional),
                                        xmippLib.MD_APPEND)
                        mdOut.clear()

            else:
                    relPart = self._createSetOfParticles()
                    relPart.copyInfo(self.inputParticles.get())
                    fnRelion = self._getExtraPath('relion_%s.star' % imgNo)
                    fnBlock = "%s@%s" % (block, fnNeighbours)
                    fnRef = "%s@%s" % (imgNo, fnGallery)
                    if getSize(fnBlock) > nop:
                        convXmp.readSetOfParticles(fnBlock, relPart)
                        if self.copyAlignment.get():
                            alignType = relPart.getAlignment()
                            alignType != em.ALIGN_NONE
                        else:
                            alignType = em.ALIGN_NONE
                        alignToPrior = getattr(self, 'alignmentAsPriors',
                                               True)
                        fillRandomSubset = getattr(self, 'fillRandomSubset',
                                                   False)
                        writeSetOfParticles(relPart, fnRelion,
                                            self._getExtraPath(),
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
                        args = {}
                        self._setNormalArgs(args)
                        args['--i'] = fnRelion
                        args['--o'] = fnOut
                        if self.referenceClassification.get():
                            args['--ref'] = fnRef
                        self._setComputeArgs(args)
                        params = ' '.join(['%s %s' % (k, str(v)) for k, v in
                                           args.iteritems()])
                        self.runJob(self._getRelionProgram(), params)
                        it = self.numberOfIterations.get()
                        if it < 10:
                            model = '_it00%d_' % it
                        else:
                            model = '_it0%d_' % it
                        fnModel = (fnOut + model + 'model.star')
                        Newblock = md.getBlocksInMetaDataFile(fnModel)[1]
                        fnNewBlock = "%s@%s" % (Newblock, fnModel)
                        mdBlocks = xmippLib.MetaData()
                        mdBlocks.read(fnNewBlock)
                        fnClass = (fnOut + model + 'classes.mrcs')
                        for n in range(self.directionalClasses):
                            objId = mdOut.addObject()
                            mdOut.setValue(xmippLib.MDL_REF, int(imgNo), objId)
                            mdOut.setValue(xmippLib.MDL_IMAGE,
                                           "%d@%s" % (n + 1, fnClass), objId)
                            mdOut.setValue(xmippLib.MDL_IMAGE_IDX, long(n + 1),
                                           objId)
                            mdOut.setValue(xmippLib.MDL_ANGLE_ROT, rot, objId)
                            mdOut.setValue(xmippLib.MDL_ANGLE_TILT, tilt, objId)
                            mdOut.setValue(xmippLib.MDL_ANGLE_PSI, psi, objId)
                            mdOut.setValue(xmippLib.MDL_SHIFT_X, 0.0, objId)
                            mdOut.setValue(xmippLib.MDL_SHIFT_Y, 0.0, objId)
                            mdOut.setValue(xmippLib.MDL_ENABLED, 1, objId)
                            mdOut.write("%s@%s" % (block, fnDirectional),
                                        xmippLib.MD_APPEND)
                        mdOut.clear()
                        for num in range(it):
                            if num != it:
                                if num < 10:
                                    fmt = '_it00%d_' % num
                                else:
                                    fmt = '_it0%d_' % num
                                fnMod = (fnOut + fmt + 'model.star')
                                fnCla =  (fnOut +fmt + 'classes.mrcs')
                                fnDat = (fnOut + fmt +'data.star')
                                fnOpt =(fnOut + fmt + 'optimiser.star')
                                fnSam = (fnOut + fmt + 'sampling.star')
                                cleanPattern(fnMod)
                                cleanPattern(fnCla)
                                cleanPattern(fnDat)
                                cleanPattern(fnOpt)
                                cleanPattern(fnSam)
            cleanPattern(self._getExtraPath('relion_*.star'))

    def randomSelectionStep(self):
        mdRandom=xmippLib.MetaData()
        mdClass=xmippLib.MetaData()
        mdRef = xmippLib.MetaData(self._getExtraPath("gallery.doc"))
        fnDirectional = self._getPath("directionalClasses.xmd")
        for i in range (self.randomVolume):
            stack=i+1
            fnRandomAverages = self._getExtraPath('randomAverages_%s' %stack)
            #nop = self.noOfParticles.get()
            for indx, block in enumerate(
                    xmippLib.getBlocksInMetaDataFile(fnDirectional)[:]):
                fnClasses = "%s@%s" % (block, fnDirectional)
                mdClass.read(fnClasses)
                rc = random.randint(1,  self.directionalClasses.get())
                imgNo = block.split("_")[1]
                galleryImgNo = int(block.split("_")[1])
                rot = mdRef.getValue(xmippLib.MDL_ANGLE_ROT, galleryImgNo)
                tilt = mdRef.getValue(xmippLib.MDL_ANGLE_TILT, galleryImgNo)
                psi = 0.0
                objId = mdRandom.addObject()
                mdRandom.setValue(xmippLib.MDL_IMAGE,
                            mdClass.getValue(xmippLib.MDL_IMAGE, rc),
                            objId)
                mdRandom.setValue(xmippLib.MDL_REF, int(imgNo), objId)
                mdRandom.setValue(xmippLib.MDL_ANGLE_ROT, rot, objId)
                mdRandom.setValue(xmippLib.MDL_ANGLE_TILT, tilt, objId)
                mdRandom.setValue(xmippLib.MDL_ANGLE_PSI, psi, objId)
                mdRandom.setValue(xmippLib.MDL_SHIFT_X, 0.0, objId)
                mdRandom.setValue(xmippLib.MDL_SHIFT_Y, 0.0, objId)
            mdRandom.write(fnRandomAverages+'.xmd')
            mdRandom.clear()
        cleanPath(fnDirectional)


    def reconstruct3DStep(self):
        self.Xdim = self.inputParticles.get().getDimensions()[0]
        ts = self.inputParticles.get().getSamplingRate()
        maxFreq=self.targetResolution.get()
        normFreq = 0.25 * (maxFreq / ts)
        K = 0.25 * (maxFreq / ts)
        if K < 1:
            K = 1
        self.Xdim2 = self.Xdim / K
        if self.Xdim2 < 32:
            self.Xdim2 = 32
            K = self.Xdim / self.Xdim2
        freq = ts / maxFreq
        ts = K * ts
        Mc = (self.backRadius.get()) * (self.Xdim2/2)
        Sym = self.symmetryGroup.get()
        for i in range(self.randomVolume):
            stack=i+1
            fnRandomAverages = self._getExtraPath('randomAverages_%s' %stack)
            self.runJob("xmipp_reconstruct_fourier","-i %s.xmd -o %s.vol --sym %s --max_resolution %f" %(fnRandomAverages,fnRandomAverages,Sym,normFreq))
            self.runJob("xmipp_transform_filter",   "-i %s.vol -o %s.vol --fourier low_pass %f --bad_pixels outliers 0.5" %(fnRandomAverages,fnRandomAverages,freq))
            self.runJob("xmipp_transform_mask","-i %s.vol  -o %s.vol --mask circular %f" %(fnRandomAverages,fnRandomAverages,Mc))

    def pcaStep(self):
        ##"".vol to .mrc conversion""##
        listVol = []
        Plugin.setEnviron()
        for i in range (self.randomVolume):
            stack = i + 1
            fnRandomAverages = self._getExtraPath('randomAverages_%s' %stack)
            inputVol = fnRandomAverages +'.vol'
            img = ImageHandler()
            mrcFn = self._getExtraPath("volume_%s.mrc" %stack)
            img.convert(inputVol, mrcFn)
            listVol.append(mrcFn)


        # ""AVERAGE VOLUME GENERATION""#
        avgVol = self._getPath('map_average.mrc')
        for vol in listVol:
            npVol = loadMrc(vol, writable=False)
            if vol == listVol[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)
            npAvgVol += npVol
        npAvgVol = npAvgVol/ len(listVol)
        saveMrc(npAvgVol.astype(dType), avgVol)

        ##""PCA ESTIMATION- generates covariance matrix""##
        npVol = loadMrc(listVol[0], False)
        dim = npVol.shape[0]
        lenght = dim ** 3
        covMatrix = []
        for vol1 in listVol:
            npVol1 = loadMrc(vol1, False)
            restNpVol1 = npVol1 - npAvgVol
            listRestNpVol1 = restNpVol1.reshape(lenght)
            row = []
            for vol2 in listVol:
                npVol2 = loadMrc(vol2, writable=False)
                restNpVol2  = npVol2 - npAvgVol
                listRestNpVol2 = restNpVol2.reshape(lenght)
                coef = np.cov(listRestNpVol1,listRestNpVol2)[0][1]
                row.append(coef)
            covMatrix.append(row)

        ##""DO PCA - generated principal components""##
        u, s, vh = np.linalg.svd(covMatrix)
        cuttOffMatrix = sum(s) * 0.60
        sCut = 0
        for i in s:
            if cuttOffMatrix > 0:
                cuttOffMatrix = cuttOffMatrix - i
                sCut += 1
            else:
                break

        ###'Generates eigenvalues and vectors"###
        eigValsFile = 'eigenvalues.txt'
        self._createMFile(s, eigValsFile)
        eigVecsFile = 'eigenvectors.txt'
        self._createMFile(vh, eigVecsFile)
        vhDel = np.transpose(np.delete(vh, np.s_[sCut:vh.shape[1]], axis=0))
        self._createMFile(vhDel, 'matrix_vhDel.txt')

        # Generates base volumes
        counter = 0
        for eigenRow in vhDel.T:
            volBase = np.zeros((dim, dim, dim))
            for (volFn, eigenCoef) in izip(listVol, eigenRow):
                npVol = loadMrc(volFn, False)
                restNpVol = npVol - npAvgVol
                volBase += eigenCoef * restNpVol
            nameVol = 'volume_base_%02d.mrc' % (counter)
            saveMrc(volBase.astype(dType), self._getExtraPath(nameVol))
            counter += 1

        # Generates the matrix projection
        matProj = []
        baseMrc = self._getExtraPath("volume_base_??.mrc")
        baseMrcFile = sorted(glob(baseMrc))
        for vol in listVol:
            npVol = loadMrc(vol, False)
            restNpVol = npVol - npAvgVol
            volRow = restNpVol.reshape(lenght)
            rowCoef = []
            for baseFn in baseMrcFile:
                npBase = loadMrc(baseFn, writable=False)
                npBaseRow = npBase.reshape(lenght)
                npBaseCol = npBaseRow.transpose()
                projCoef = np.dot(volRow, npBaseCol)
                rowCoef.append(projCoef)
            matProj.append(rowCoef)


        ###Clustering Methods ####
        if self.ClusteringMethod.get() == 0:
          from sklearn.cluster import KMeans
          nC=self.clusterCentres.get()
          kmeans = KMeans(n_clusters=nC).fit(matProj)
          cc = kmeans.cluster_centers_

        else:
             from sklearn.cluster import AffinityPropagation
             ap = AffinityPropagation(damping=0.9).fit(matProj)
             print("cluster_centers", ap.cluster_centers_)
             cc = ap.cluster_centers_

        ####Coordinates to volume ######
        fnVolume = self._getExtraPath('recons_vols')
        fnOutVol = self._getPath('output_volumes.vol')

        orignCount = 0
        if not exists(fnVolume):
            makePath(fnVolume)
        for projRow in cc:
            vol = np.zeros((dim, dim, dim))
            for baseVol, proj in izip(baseMrcFile, projRow):
                volNpo = loadMrc(baseVol, False)
                vol += volNpo * proj
            finalVol = vol + npAvgVol
            nameVol = 'volume_reconstructed_%03d.mrc' % (orignCount)
            saveMrc(finalVol.astype(dType), self._getExtraPath('recons_vols', nameVol))
            orignCount += 1
            ####for output step####
            fnRecon = join(fnVolume + '/' + nameVol)
            mdOutVol = xmippLib.MetaData(fnRecon)
            mdOutVol.write(fnOutVol)
        cleanPattern(self._getExtraPath('volume_base_*.mrc'))
        cleanPattern(self._getExtraPath('volume_*.mrc'))

    def cleanStep(self):
        cleanPath(self._getExtraPath('scaled_particles.stk'))
        cleanPath(self._getExtraPath('scaled_particles.xmd'))
        cleanPath(self._getExtraPath('volume.vol'))

    def createOutputStep(self):

        ## create a SetOfVolumes and define its relations
        volumes = self._createSetOfVolumes()
        volumes.copyInfo(self.inputParticles.get())
        volumes.setSamplingRate(volumes.getSamplingRate())
        self. _fillDataFromIter(volumes)
        self._defineOutputs(outputVolumes=volumes)
        self._defineSourceRelation(self.inputParticles, volumes)
    #--------------------------- INFO functions -------------------------------------------- 
    def _validate(self):
        pass
    
    def _summary(self):
        pass
    
    def _methods(self):
        messages = []
        return messages
    
    def _citations(self):
        return ['Vargas2014a']
    
    #--------------------------- UTILS functions -------------------------------------------- 
    #def _updateLocation(self, item, row):

     #   index, filename = xmippToLocation(row.getValue(md.MDL_IMAGE))
      #  item.setLocation(index, filename)
    def _setNormalArgs(self, args):
        maskDiameter = self.maskDiameterA.get()
        newTs = self.targetResolution.get() * 0.4
        if maskDiameter <= 0:
          x = self._getInputParticles().getDim()[0]
          maskDiameter = self._getInputParticles().getSamplingRate() * x
        args.update({'--particle_diameter': maskDiameter,
                     '--angpix': newTs,
                     })
        args['--K'] = self.directionalClasses.get()
        args['--zero_mask'] = ''


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

    def _fillDataFromIter(self, volumes):
        fnOutVol = self._getPath('output_volumes.vol')
        convXmp.readSetOfVolumes(fnOutVol,volumes)

    def _postprocessVolume(self, vol):
        self._counter += 1
        vol.setObjComment('Output volume %02d' % self._counter)

    def clone(self):
        """ Override the clone defined in Object
        to avoid copying _mapperPath property
        """
        pass

    def _callBack(self, newItem, row):
        if row.getValue(xmippLib.MDL_ENABLED) == -1:
            setattr(newItem, "_appendItem", False)

    def _createMFile(self, matrix, name='matrix.txt'):

        f = open(name, 'w')
        for list in matrix:
            s = "%s\n" % list
            f.write(s)
        f.close()


    def _mrcToNp(self, volList):
        listNpVol = []
        for vol in volList:
            volNp = loadMrc(vol, False)
            dim = volNp.shape[0]
            lenght = dim**3
            volList = volNp.reshape(lenght)
            listNpVol.append(volList)
        return listNpVol, listNpVol[0].dtype




    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #plt.hexbin(x_proj, y_proj, gridsize=20, mincnt=1, bins='log')
    #plt.xlabel('x_pca', fontsize=16)
    #plt.ylabel('y_pca', fontsize=16)
    #plt.colorbar()
    #plt.savefig('interpolated_controlPCA_splic.png', dpi=100)
    #plt.close(fig)
        ##""Construct PCA histogram""##
        #x_proj = [item[0] for item in matProj]
        #y_proj = [item[1] for item in matProj]

        ## save coordinates:
        #mat_file = 'matProj_splic.txt'
        #self._createMFile(matProj, mat_file)
        #x_file = 'x_proj_splic.txt'
        #self._createMFile(x_proj, x_file)
        #y_file = 'y_proj_splic.txt'
        #self._createMFile(y_proj, y_file)
