# **************************************************************************
# *
# * Authors:         Javier Vargas (jvargas@cnb.csic.es) (2016)
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


from pyworkflow.object import Float, String
from pyworkflow.protocol.params import (PointerParam, FloatParam, STEPS_PARALLEL,EnumParam,
                                        StringParam, BooleanParam, IntParam,LabelParam, PathParam,LEVEL_ADVANCED)
from pyworkflow.em.data import Volume, Image
from pyworkflow.em import Viewer
import pyworkflow.em.metadata as md
from cryomethods.protocols import ProtDirectionalPruning

from cryomethods.protocols import ProtocolBase

from pyworkflow.em.protocol import ProtClassify3D
from pyworkflow.em.protocol import ProtAnalysis3D
from cryomethods.convert import writeSetOfParticles,splitInCTFGroups
from pyworkflow.em.metadata.utils import getSize
import xmippLib
import math
import random
import pyworkflow.em as em
from os.path import join, exists
from os import remove

from pyworkflow.em.packages.xmipp3.constants import (ML2D, CL2D)

import cryomethods.convertXmp as convXmp


class ProtClass3DRansac(ProtClassify3D,ProtDirectionalPruning ,ProtAnalysis3D, ProtocolBase):

    """    
    Performs 3D classification of input particles with previous alignment
    """
    _label = 'directional_ransac'

    CL2D = 0
    ML2D = 1
    RL2D = 2

    
    def __init__(self, *args, **kwargs):
        ProtClassify3D.__init__(self, *args, **kwargs)
        ProtAnalysis3D.__init__(self, *args, **kwargs)
        ProtDirectionalPruning.__init__(self, *args, **kwargs)
        ProtocolBase.__init__(self, **args)
        
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
        form.addParam('numClasses', IntParam, default=2, label='Number of 3D classes')
        form.addParam('symmetryGroup', StringParam, default='c1',
                      label="Symmetry group", 
                      help='See [[Xmipp Symmetry][http://www2.mrc-lmb.cam.ac.uk/Xmipp/index.php/Conventions_%26_File_formats#Symmetry]] page '
                           'for a description of the symmetry format accepted by Xmipp') 

        form.addSection(label='Directional Classes')
        form.addParam('directionalSamples', IntParam, default=5, label='Number of directional samples',
                      help="Number of random samples of the angular directions to obtain 3D reconstructions")
        form.addParam('directionalTrials', IntParam, default=100, label='Number of directional trials', expertLevel=LEVEL_ADVANCED, 
                      help="Number of random combinations of the angular directions to select good orientations in which perform 2D classification")
        form.addParam('angularSampling', FloatParam, default=5, label='Angular sampling', expertLevel=LEVEL_ADVANCED, help="In degrees")
        form.addParam('angularDistance', FloatParam, default=10, label='Angular distance', expertLevel=LEVEL_ADVANCED,
                      help="In degrees. An image belongs to a group if its distance is smaller than this value")
        
        groupClass2D = form.addSection(label='2D Classification')
        groupClass2D.addParam('Class2D', EnumParam, choices=['ML2D','CL2D','RL2D'], default= RL2D,
                     label="2D classification method", display=EnumParam.DISPLAY_COMBO,
                     help='2D classification algorithm used to be applied to the directional classes. \n ')
        
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
                      condition='classMethod==1')
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
                      label='Consider alignment as priors?',
                      help='If set to Yes, then alignment information from '
                           'input particles will be considered as PRIORS. This '
                           'option is mandatory if you want to do local '
                           'searches')
        form.addParam('fillRandomSubset', BooleanParam, default=False,
                      condition='classMethod==2',
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
        form.addParam('referenceClassification', BooleanParam, default=True,
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

        
        form.addParallelSection(threads=1, mpi=1)

    def _insertAllSteps(self):
        
        convertId = self._insertFunctionStep('convertInputStep',
                                             self.inputParticles.get().getObjId(), self.inputVolume.get().getObjId(), 
                                             self.targetResolution.get())
        
        self._insertFunctionStep('constructGroupsStep', self.inputParticles.get().getObjId(),
                                 self.angularSampling.get(), self.angularDistance.get(), self.symmetryGroup.get())
        
        self._insertFunctionStep('selectDirections', self.symmetryGroup.get())

        self._insertFunctionStep('classify2DStep')
        
        self._insertFunctionStep('reconstruct3DStep')

        #deps = [] # store volumes steps id to use as dependencies for last step
        
        #consGS = self._insertFunctionStep('constructGroupsStep', self.inputParticles.get().getObjId(),
                                 
        #commonParams    = self._getCommonParams()
        #deps.append(convertId)
        
    def convertInputStep(self, particlesId, volId, targetResolution):
        #XmippProtDirectionalClasses.convertInputStep(self, particlesId, volId, targetResolution)
        """ 
        Write the input images as a Xmipp metadata file. 
        particlesId: is only need to detect changes in
        input particles and cause restart from here.
        """
        convXmp.writeSetOfParticles(self.inputParticles.get(), self._getPath('input_particles.xmd'))
        Xdim = self.inputParticles.get().getDimensions()[0]
        Ts = self.inputParticles.get().getSamplingRate()
        newTs = self.targetResolution.get()*0.4
        newTs = max(Ts,newTs)
        newXdim = Xdim*Ts/newTs
        

        params =  '  -i %s' % self._getPath('input_particles.xmd')
        params +=  '  -o %s' % self._getExtraPath('scaled_particles.stk')
        params +=  '  --save_metadata_stack %s' % self._getExtraPath('scaled_particles.xmd')
        params +=  '  --dim %d' % newXdim
        
        self.runJob('xmipp_image_resize',params)
        from pyworkflow.em.convert import ImageHandler
        img = ImageHandler()
        img.convert(self.inputVolume.get(), self._getExtraPath("volume.vol"))
        Xdim = self.inputVolume.get().getDim()[0]
        if Xdim!=newXdim:
            self.runJob("xmipp_image_resize","-i %s --dim %d"%\
                        (self._getExtraPath("volume.vol"),
                        newXdim), numberOfMpi=1)

    def constructGroupsStep(self, particlesId, angularSampling, angularDistance, symmetryGroup):
       ProtDirectionalPruning.constructGroupsStep(self, particlesId, angularSampling, angularDistance, symmetryGroup)
        
    def selectDirections(self,symmetryGroup):
        fnNeighbours = self._getExtraPath("neighbours.xmd")
        fnGallery=self._getExtraPath("gallery.doc")
        listOfBlocks = xmippLib.getBlocksInMetaDataFile(fnNeighbours)
        
        Xdim = self.inputParticles.get().getDimensions()[0]
#JV: es Ts o NewTs en normFreq        
        Ts = self.inputParticles.get().getSamplingRate()
        newTs = self.targetResolution.get()*0.4
        newTs = max(Ts,newTs)
        self.newRadius=(self.backRadius.get())*(Ts/newTs)
        normFreq = 0.25*(self.targetResolution.get()/Ts)
        
        volRef = xmippLib.Image(self._getExtraPath("volume.vol"))
        volRef.convert2DataType(xmippLib.DT_DOUBLE)
        
        #MDL_DIRECTION
        mdDirections = xmippLib.MetaData()
        mdRef = xmippLib.MetaData(fnGallery)
        
        for d in range(self.directionalTrials):
            
            list = range(self.directionalSamples)
            objIdDirs = mdDirections.addObject()
            md = xmippLib.MetaData()
            
            for i in range(self.directionalSamples):
                randBlock =random.randint(0, len(listOfBlocks))
                block = listOfBlocks[randBlock]
                fnBlock="%s@%s"%(block,fnNeighbours)
                fnDir = self._getExtraPath("direction_%s"%i)
                list[i] = float(randBlock)
                
                ''' the gallery give is a good reference'''
                galleryImgNo = int(block.split("_")[1])
                rot  = mdRef.getValue(xmippLib.MDL_ANGLE_ROT,galleryImgNo)
                tilt = mdRef.getValue(xmippLib.MDL_ANGLE_TILT,galleryImgNo)
                psi = 0.0  # we are aligning to the gallery so psi is equals to 0
                
                self.runJob("xmipp_image_align","-i %s  --oroot %s --iter 5 --ref %s --dontAlign"
                            %(fnBlock,fnDir,mdRef.getValue(xmippLib.MDL_IMAGE,galleryImgNo)),numberOfMpi=1)
                #
                self.runJob("xmipp_transform_mask","-i %s  -o %s --mask circular -%f"
                            %(self._getExtraPath("direction_%s_ref.xmp"%i),self._getExtraPath("direction_%s_ref.xmp"%i),self.newRadius)
                              ,numberOfMpi=1)
                
                objId = md.addObject()
                md.setValue(xmippLib.MDL_IMAGE,self._getExtraPath("direction_%s_ref.xmp"%i),objId)
                md.setValue(xmippLib.MDL_ANGLE_ROT,rot,objId)
                md.setValue(xmippLib.MDL_ANGLE_TILT,tilt,objId)
                md.setValue(xmippLib.MDL_ANGLE_PSI,psi,objId)
                md.setValue(xmippLib.MDL_SHIFT_X,0.0,objId)
                md.setValue(xmippLib.MDL_SHIFT_Y,0.0,objId)
    
            fnRecons = self._getExtraPath("guess")
            md.write(fnRecons+'.xmd')
            self.runJob("xmipp_reconstruct_fourier","-i %s.xmd -o %s.vol --sym %s --max_resolution %f" %(fnRecons,fnRecons,self.symmetryGroup.get(),normFreq))
            self.runJob("xmipp_transform_filter",   "-i %s.vol -o %s.vol --fourier low_pass %f --bad_pixels outliers 0.5" %(fnRecons,fnRecons,normFreq))
            self.runJob("xmipp_transform_mask","-i %s.vol  -o %s.vol --mask circular -%f" %(fnRecons,fnRecons,self.newRadius))
            md.clear()
            
            vol = xmippLib.Image(self._getExtraPath('guess.vol'))
            vol.convert2DataType(xmippLib.DT_DOUBLE)
            corr = vol.correlate(volRef)
            
            mdDirections.setValue(xmippLib.MDL_DIRECTION,list, objIdDirs)
            mdDirections.setValue(xmippLib.MDL_WEIGHT,corr,objIdDirs)

        for i in range(self.directionalSamples):
            remove(self._getExtraPath("direction_%s_ref.xmp"%i))
            remove(self._getExtraPath("direction_%s_alignment.xmd"%i))
        remove(self._getExtraPath('guess.vol'))
        remove(self._getExtraPath('guess.xmd'))

        mdDirections.sort(xmippLib.MDL_WEIGHT)
        mdDirections.write(self._getExtraPath("directions.xmd"))

    def classify2DStep(self):

        mdOut = xmippLib.MetaData()
        mdDirections = xmippLib.MetaData(self._getExtraPath("directions.xmd"))
        index = mdDirections.size()
        list = mdDirections.getValue(xmippLib.MDL_DIRECTION,index)
        fnNeighbours = self._getExtraPath("neighbours.xmd")
        fnGallery=self._getExtraPath("gallery.stk")
        listOfBlocks = xmippLib.getBlocksInMetaDataFile(fnNeighbours)
        fnDirectional=self._getPath("directionalClasses.xmd")
        mdRef = xmippLib.MetaData(self._getExtraPath("gallery.doc"))

        for i in range(self.directionalSamples):
            
            selectedBlockNumber = int(list[i])
            #This is because in one case the number starts in 0 and in other in 1
            block = listOfBlocks[selectedBlockNumber-1]
            fnBlock="%s@%s"%(block,fnNeighbours)
            fnDir = self._getExtraPath("direction_%s"%i)
            galleryImgNo = int(block.split("_")[1])
            rot  = mdRef.getValue(xmippLib.MDL_ANGLE_ROT,galleryImgNo)
            tilt = mdRef.getValue(xmippLib.MDL_ANGLE_TILT,galleryImgNo)
            psi = 0.0

            Nlevels = self.numClasses.get()
            fnBlock = "%s@%s" % (block, fnNeighbours)
            
            if getSize(fnBlock) >= 2:
                nClasses = Nlevels if (getSize(fnBlock)/(Nlevels*10)) >1 else int(getSize(fnBlock)/10)
                if (nClasses > Nlevels): Nlevels
                if (nClasses < 1): nClasses=1
                                       
                if ( self.Class2D.get() == CL2D):
                    args = "-i %s --odir %s --iter %d --nref %d --nref0 %d --distance correlation --classicalMultiref --maxShift %d --dontAlign" % \
                            (fnBlock, fnDir, self.CL2D_it, nClasses, 1, self.CL2D_shift)
                    self.runJob("xmipp_classify_CL2D", args, numberOfMpi=2)
                    fnAlignRoot = join(fnDir, "classes")
                    fnOut = join(fnDir,"level_%02d/class_classes.stk"%(nClasses-1))
                    self.runJob("xmipp_image_align", "-i %s --ref %s@%s --oroot %s --iter 5 --dontAlign" % \
                                (fnOut, selectedBlockNumber, fnGallery, fnAlignRoot), numberOfMpi=1)
                    self.runJob("xmipp_transform_geometry", "-i %s_alignment.xmd --apply_transform" % fnAlignRoot, numberOfMpi=1)

                elif ( self.Class2D.get() == ML2D):
                    args = "-i %s --oroot %s --ref %s@%s --nref %d --mirror" % \
                            (fnBlock, (fnDir+'_'), selectedBlockNumber, fnGallery, nClasses)
                    self.runJob("xmipp_ml_align2d", args, numberOfMpi=2)
                    fnOut = self._getExtraPath("direction_%s_classes.stk"%i)
                else:

                    relPart = self._createSetOfParticles()
                    relPart.copyInfo(self.inputParticles.get())

                    fnRelion = self._getExtraPath('relion_%s.star' % selectedBlockNumber)

                    fnBlock = "%s@%s" % (block, fnNeighbours)
                    fnRef = "%s@%s" % (selectedBlockNumber, fnGallery)

                    if getSize(fnBlock) >= 2:

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
                            print("SAAAA", fnOut)
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


                for n in range(nClasses):
                    objId = mdOut.addObject()
                    mdOut.setValue(xmippLib.MDL_REF, int(selectedBlockNumber), objId)
                    mdOut.setValue(xmippLib.MDL_IMAGE, "%d@%s" % (n + 1, fnOut), objId)
                    mdOut.setValue(xmippLib.MDL_IMAGE_IDX, long(n+1), objId)
                    mdOut.setValue(xmippLib.MDL_ANGLE_ROT,rot,objId)
                    mdOut.setValue(xmippLib.MDL_ANGLE_TILT,tilt,objId)
                    mdOut.setValue(xmippLib.MDL_ANGLE_PSI,psi,objId)
                    mdOut.setValue(xmippLib.MDL_SHIFT_X,0.0,objId)
                    mdOut.setValue(xmippLib.MDL_SHIFT_Y,0.0,objId)
                    
                    mdOut.write("%s@%s" % (block, fnDirectional), xmippLib.MD_APPEND)
                mdOut.clear()

    def reconstruct3DStep(self):
        
        fnDirectionalClasses = self._getPath("directionalClasses.xmd")
        listOfBlocks = xmippLib.getBlocksInMetaDataFile(fnDirectionalClasses)
        numClass = len(listOfBlocks)
        numParticlePerClass = self.numClasses.get()
        
        #Generate all possible combinations and reconstruct everything
        from numpy import mgrid, rollaxis, reshape
        args = "numComb=mgrid[0:%s, "%(str(numParticlePerClass))
        for i in range(numClass-2):
            args += "0:%s, "%(str(numParticlePerClass))
        args += "0:%s] "%(str(numParticlePerClass))
        exec args
        
        args = "numComb = rollaxis(numComb, 0, %s)"%(str(numClass+1))
        exec args
        
        args = "numComb=numComb.reshape(%s * "%(str(self.numClasses.get()))
        for i in range(numClass-2):
            args += "%s * "%(str(numParticlePerClass))
        args += "%s, %s) "%(str(numParticlePerClass), str(numClass))
        exec args

        
        raise Exception('spam', 'eggs')

        #  exec "from numpy import mgrid \na=mgrid[1:%s] \nprint(a)"%(str(n))
        #  exec "a=mgrid[1:%s] \nprint(a)"%(str(n))
        
        # a = np.mgrid[0:3, 0:3, 0:3, 0:3]
        # a = np.rollaxis(a, 0, 5)
        #
        #  a = np.rollaxis(a, 0, 5)
        #  a = a.reshape((3 * 3 * 3* 3, 4))
        

        md = xmippLib.MetaData()
        fnRecons = self._getExtraPath("recons")
        Ts = self.inputParticles.get().getSamplingRate()
        normFreq = 0.25*(self.targetResolution.get()/Ts)

        for i in range(len(listOfBlocks)):
            block = listOfBlocks[i]
            fnBlock="%s@%s"%(block,fnDirectionalClasses)
            mdDirectionalClasses = xmippLib.MetaData(fnBlock)

            objId = md.addObject()
            md.setValue(xmippLib.MDL_IMAGE,mdDirectionalClasses.getValue(xmippLib.MDL_IMAGE,1),objId)
            md.setValue(xmippLib.MDL_ANGLE_ROT,mdDirectionalClasses.getValue(xmippLib.MDL_ANGLE_ROT,1),objId)
            md.setValue(xmippLib.MDL_ANGLE_TILT,mdDirectionalClasses.getValue(xmippLib.MDL_ANGLE_TILT,1),objId)
            md.setValue(xmippLib.MDL_ANGLE_PSI,mdDirectionalClasses.getValue(xmippLib.MDL_ANGLE_PSI,1),objId)
            md.setValue(xmippLib.MDL_SHIFT_X,0.0,objId)
            md.setValue(xmippLib.MDL_SHIFT_Y,0.0,objId)
            
        md.write(fnRecons+'.xmd', xmippLib.MD_APPEND)
        self.runJob("xmipp_reconstruct_fourier","-i %s.xmd -o %s.vol --sym %s --max_resolution %f" %(fnRecons,fnRecons,self.symmetryGroup.get(),normFreq))
        self.runJob("xmipp_transform_filter",   "-i %s.vol -o %s.vol --fourier low_pass %f --bad_pixels outliers 0.5" %(fnRecons,fnRecons,normFreq))
        self.runJob("xmipp_transform_mask","-i %s.vol  -o %s.vol --mask circular -%f" %(fnRecons,fnRecons,self.newRadius))
        md.clear()


                
    def createOutputStep(self):
        pass
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
    def _updateLocation(self, item, row):
        index, filename = xmippToLocation(row.getValue(md.MDL_IMAGE))
        item.setLocation(index, filename)
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
