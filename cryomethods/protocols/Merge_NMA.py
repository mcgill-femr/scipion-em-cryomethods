# **************************************************************************
# *
# * Authors:  Satinder Kaur (satinder.kaur@mail.mcgill.ca), May 2019
# *
# *
# *
# * Department of Anatomy and Cell Biology, McGill University, Montreal
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
# *  e-mail address 'satinder.kaur@mail.mcgill.ca'
# *
# **************************************************************************

import math
import itertools


import xmippLib
import random
from pyworkflow.em.metadata import getSize
from pyworkflow.utils import *
from pyworkflow.em import *
from os.path import join
from pyworkflow.utils import redStr

from xmipp3.protocols.nma.convert import getNMAEnviron

from pyworkflow.utils.path import moveFile, createLink, cleanPattern, makePath
from xmipp3.convert import getImageLocation
from pyworkflow.protocol.params import (PointerParam, EnumParam, IntParam,
                                        FloatParam,
                                        LEVEL_ADVANCED, LabelParam)
from xmipp3.protocols.nma.protocol_nma_base import NMA_CUTOFF_REL
from pyworkflow.em.convert.atom_struct import cifToPdb
import pyworkflow.protocol.constants as const
from xmipp3.constants import NMA_HOME
from xmipp3 import Plugin
from os.path import exists, basename
from pyworkflow.em.protocol import ProtImportParticles
import re
import numpy as np
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
import pyworkflow.em as em
import pyworkflow.protocol.params as params
from pyworkflow.utils.path import cleanPath, replaceBaseExt

from cryomethods import Plugin
from cryomethods.constants import (METHOD, ANGULAR_SAMPLING_LIST,
                                   MASK_FILL_ZERO)
import pyworkflow.em.metadata as md
from cryomethods.convert import writeSetOfParticles
from cryomethods.convert import (writeSetOfParticles, rowToAlignment,
                                 relionToLocation, loadMrc, saveMrc,
                                 alignVolumes, applyTransforms)
from numpy.core import transpose





NMA_MASK_NONE = 0
NMA_MASK_THRE = 1
NMA_MASK_FILE = 2
NMA_CUTOFF_ABS = 0
NMA_CUTOFF_REL = 1
nVoli = 1
IMPORT_FROM_ID = 0
IMPORT_OBJ = 1
IMPORT_FROM_FILES = 2
VOL_ZERO = 0
VOL_ONE = 1
VOL_TWO = 2



class ProtLandscapeNMA(em.EMProtocol):
    _label = 'Landscape NMA'

    IS_2D = False
    IS_VOLSELECTOR = False
    IS_AUTOCLASSIFY = False
    OUTPUT_TYPE = em.SetOfVolumes
    FILE_KEYS = ['data', 'optimiser', 'sampling']
    PREFIXES = ['']


    #-------------------particleattrStep---------------------------------

    def _createFilenameTemplates(self):
        """ Centralize how files are called for iterations and references. """
        self.levDir = self._getExtraPath('run_%(run)02d/')
        self.extraIter = self._getExtraPath('run_%(ruNum)02d/relion_it%(iter)03d_')
        self.extraLast = self._getExtraPath('parSel2/relion_it%(iter)03d_')
        myDict = {
            'input_star': self._getPath('input_particles_%(run)02d.star'),
            'data_scipion': self.extraIter + 'data_scipion.sqlite',
            'volumes_scipion': self.extraIter + 'volumes.sqlite',
            'data': self.extraIter + 'data.star',
            'model': self.extraIter + 'model.star',
            'optimiser': self.extraIter + 'optimiser.star',
            'angularDist_xmipp': self.extraIter + 'angularDist_xmipp.xmd',
            'all_avgPmax_xmipp': self._getTmpPath(
                'iterations_avgPmax_xmipp.xmd'),
            'all_changes_xmipp': self._getTmpPath(
                'iterations_changes_xmipp.xmd'),
            'selected_volumes': self._getTmpPath('selected_volumes_xmipp.xmd'),
            'movie_particles': self._getPath('movie_particles.star'),
            'dataFinal': self._getExtraPath("relion_data.star"),
            'modelFinal': self.extraIter + 'model.star',
            'finalvolume': self._getExtraPath("relion_class%(ref3d)03d.mrc:mrc"),
            'preprocess_parts': self._getPath("preprocess_particles.mrcs"),
            'preprocess_parts_star': self._getPath("preprocess_particles.star"),
            'avgMap': self.levDir + 'map_average.mrc',
            'finalAvgMap': self._getExtraPath('map_average.mrc')
        }
        # add to keys, data.star, optimiser.star and sampling.star
        for key in self.FILE_KEYS:
            myDict[key] = self.extraIter + '%s.star' % key
            key_xmipp = key + '_xmipp'
            myDict[key_xmipp] = self.extraIter + '%s.xmd' % key
        # add other keys that depends on prefixes
        for p in self.PREFIXES:
            myDict['%smodel' % p] = self.extraIter + '%smodel.star' % p
            myDict['%svolume' % p] = self.extraIter + p + \
                                     'class%(ref3d)03d.mrc:mrc'
        self._updateFilenamesDict(myDict)

    def _createTemplates(self, run=None):
        run = self._rLev if run is None else run
        """ Setup the regex on how to find iterations. """
        self._iterTemplate = self._getFileName('data', ruNum=run,
                                               iter=0).replace('000', '???')
        # Iterations will be identify by _itXXX_ where XXX is the iteration
        # number and is restricted to only 3 digits.
        self._iterRegex = re.compile('_it(\d{3,3})_')
        self._classRegex = re.compile('_class(\d{3,3}).')


    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form, expertLev=em.LEVEL_ADVANCED, cond='True'):
        form.addSection(label='Input')
        form.addParam('inputVolume', PointerParam, label="Input Volume",
                      important=True, pointerClass='Volume')

        form.addParam('inputParticles', params.PointerParam,
                      pointerClass='SetOfParticles',
                      important=True,
                      label="Input particles",
                      help='Select the input images from the project.')

        form.addParam('subsetSize', params.IntParam, default=1000,
                       label='Subset size',
                       help='Number of individual particles that will be '
                            'use to obtain the best initial volume')

        form.addParam('targetResol', params.FloatParam, default=10,
                       label='Target Resolution (A)',
                       help='In order to save time, you could rescale both '
                            'particles and maps to a pisel size = resol/2. '
                            'If set to 0, no rescale will be applied to '
                            'the initial references.')
        form.addParam('maskMode', EnumParam,
                      choices=['none', 'threshold', 'file'],
                      default=NMA_MASK_NONE,
                      label='Mask mode', display=EnumParam.DISPLAY_COMBO,
                      help='')
        form.addParam('maskThreshold', FloatParam, default=0.01,
                      condition='maskMode==%d' % NMA_MASK_THRE,
                      label='Threshold value',
                      help='Gray values below this threshold are set to 0')
        form.addParam('volumeMask', PointerParam, pointerClass='VolumeMask',
                      label='Mask volume',
                      condition='maskMode==%d' % NMA_MASK_FILE,
                      )
        form.addParam('pseudoAtomRadius', FloatParam, default=1,
                      label='Pseudoatom radius (vox)',
                      help='Pseudoatoms are defined as Gaussians whose \n'
                           'standard deviation is this value in voxels')
        form.addParam('pseudoAtomTarget', FloatParam, default=5,
                      expertLevel=LEVEL_ADVANCED,
                      label='Volume approximation error(%)',
                      help='This value is a percentage (between 0.001 and '
                           '100) \n specifying how fine you want to '
                           'approximate the EM \n volume by the pseudoatomic '
                           'structure. Lower values \n imply lower '
                           'approximation error, and consequently, \n'
                           'more pseudoatoms.')

        #----------------------------------NMA------------------------------
        form.addSection(label='Normal Mode Analysis')
        form.addParam('numberOfModes', IntParam, default=20,
                      label='Number of modes',
                      help='The maximum number of modes allowed by the method for \n'
                           'atomic normal mode analysis is 6 times the number of  \n'
                           'RTB blocks and for pseudoatomic normal mode analysis 3\n'
                           'times the number of pseudoatoms. However, the protocol\n'
                           'allows only up to 200 modes as 20-100 modes are usually\n'
                           'enough. The number of modes given here should be below \n'
                           'the minimum between these two numbers.')
        form.addParam('cutoffMode', EnumParam, choices=['absolute', 'relative'],
                      default=NMA_CUTOFF_REL,
                      label='Cut-off mode',
                      help='Absolute distance allows specifying the maximum distance (in Angstroms) for which it\n'
                           'is considered that two atoms are connected. '
                           'Relative distance allows to specify this distance\n'
                           'as a percentile of all the distances between an atom and its nearest neighbors.')
        form.addParam('rc', FloatParam, default=8,
                      label="Cut-off distance (A)",
                      condition='cutoffMode==%d' % NMA_CUTOFF_ABS,
                      help='Atoms or pseudoatoms beyond this distance will not interact.')
        form.addParam('rcPercentage', FloatParam, default=95,
                      label="Cut-off percentage",
                      condition='cutoffMode==%d' % NMA_CUTOFF_REL,
                      help='The interaction cutoff distance is calculated as the distance\n'
                           'below which is this percentage of interatomic or interpseudoatomic\n'
                           'distances. \n'
                           'Atoms or pseudoatoms beyond this distance will not interact.')
        form.addParam('collectivityThreshold', FloatParam, default=0.15,
                      label='Threshold on collectivity',
                      help='Collectivity degree is related to the number of atoms or \n'
                           'pseudoatoms that are affected by the mode, and it is normalized\n'
                           'between 0 and 1. Modes below this threshold are deselected in  \n'
                           'the modes metadata file. Set to 0 for no deselection. You can  \n'
                           'always modify the selection manually after the modes metadata  \n'
                           'file is created. The modes metadata file can be used with      \n'
                           'Flexible fitting protocol. Modes 1-6 are always deselected as  \n'
                           'they are related to rigid-body movements.')
        form.addParam('rtbBlockSize', IntParam, default=10,
                      expertLevel=LEVEL_ADVANCED,
                      label='Number of residues per RTB block',
                      help='This is the RTB block size for the RTB NMA method. \n'
                           'When calculating the normal modes, aminoacids are '
                           'grouped\n'
                           'into blocks of this size that are moved '
                           'translationally\n'
                           'and rotationally together.')
        form.addParam('rtbForceConstant', FloatParam, default=10,
                      expertLevel=LEVEL_ADVANCED,
                      label='Interaction force constant',
                      help='This is the RTB block size for the RTB NMA method. \n'
                           'When calculating the normal modes, aminoacids are '
                           'grouped\n'
                           'into blocks of this size that are moved '
                           'translationally\n'
                           'and rotationally together.')
        form.addSection(label='Animation')
        form.addParam('amplitude', FloatParam, default=70,
                      label="Amplitude")
        form.addParam('nframes', IntParam, default=10,
                      expertLevel=LEVEL_ADVANCED,
                      label='Number of frames')
        form.addParam('downsample', FloatParam, default=1,
                      expertLevel=LEVEL_ADVANCED,
                      # condition=isEm
                      label='Downsample pseudoatomic structure',
                      help='Downsample factor 2 means removing one half of the '
                           'atoms or pseudoatoms.')
        form.addParam('pseudoAtomThreshold', FloatParam, default=0,
                      expertLevel=LEVEL_ADVANCED,
                      # cond
                      label='Pseudoatom mass threshold',
                      help='Remove pseudoatoms whose mass is below this '
                           'threshold. '
                           'This value should be between 0 and 1.\n'
                           'A threshold of 0 implies no atom removal.')
        group = form.addGroup('Single mode')
        group.addParam('modeNumber', IntParam, default=7,
                       label='Mode number')
        group.addParam('displayVmd', LabelParam,
                       label='Display mode animation with VMD?')
        group.addParam('displayDistanceProfile', LabelParam, default=False,
                       label="Plot mode distance profile?")
        #---------------------------convert pdb----------------------------
        form.addParam('inputPdbData', params.EnumParam,
                      choices=['id', 'object', 'file'],
                      label="Retrieve PDB from", default=IMPORT_FROM_ID,
                      display=params.EnumParam.DISPLAY_HLIST,
                      help='Retrieve PDB data from server, use a pdb Object, or a local file')
        form.addParam('pdbId', params.StringParam,
                      condition='inputPdbData == 0',
                      label="Pdb Id ", allowsNull=True,
                      help='Type a pdb Id (four alphanumeric characters).')
        form.addParam('pdbObj', params.PointerParam, pointerClass='AtomStruct',
                      label="Input pdb ",
                      condition='inputPdbData == 1', allowsNull=True,
                      help='Specify a pdb object.')
        form.addParam('pdbFile', params.FileParam,
                      label="File path",
                      condition='inputPdbData == 2',
                      allowsNull=True,
                      help='Specify a path to desired PDB structure.')
        form.addParam('sampling', params.FloatParam, default=1.0,
                      label="Sampling rate (A/px)",
                      help='Sampling rate (Angstroms/pixel)')
        form.addParam('setSize', params.BooleanParam, label='Set final size?',
                      default=False)
        form.addParam('size', params.IntParam, condition='setSize',
                      allowsNull=True,
                      label="Final size (px)",
                      help='Final size in pixels. If no value is provided, protocol will estimate it.')
        form.addParam('centerPdb', params.BooleanParam, default=True,
                      expertLevel=const.LEVEL_ADVANCED,
                      label="Center PDB",
                      help='Center PDB with the center of mass')
        #------------------------------volume attractor-----------------

        self._defineConstants()
        form.addSection(label='Particle attractor')


        form.addParam('maskDiameterA', params.IntParam, default=-1,
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
        form.addParam('maskZero', params.EnumParam, default=0,
                      choices=['Yes, fill with zeros',
                               'No, fill with random noise'],
                      label='Mask particles with zeros?',
                      help='If set to <Yes>, then in the individual particles, '
                           'the area outside a circle with the radius '
                           'of the particle will be set to zeros prior to '
                           'taking the Fourier transform. '
                           'This will remove noise and therefore increase '
                           'sensitivity in the alignment and classification. '
                           'However, it will also introduce correlations '
                           'between the Fourier components that are not '
                           'modelled. When set to <No>, then the solvent area '
                           'is filled with random noise, which prevents '
                           'introducing correlations.High-resolution '
                           'refinements (e.g. in 3D auto-refine) tend to work '
                           'better when filling the solvent area with random '
                           'noise, some classifications go better when using '
                           'zeros.')


        group = form.addGroup('Volume Selector')
        group.addParam('finalVols', params.EnumParam, default=  VOL_ZERO,
                      choices=['all of them',
                               'last one',
                               'equal space selection'],
                      label='Volumes to select for classification',
                      display=EnumParam.DISPLAY_COMBO,
                      help='If set to <all of them>, then select all volumes'
                           'converted from pdb maps created by NMA.'
                           'When set to <last one>, then select last volume'
                           'converted from pdb maps created by NMA. When set to'
                           ' <equal space selection>, then select 1st, 3rd and 5th'
                           ' volume converted from pdb maps created by NMA.')
        group.addParam('lastVol', params.IntParam, default=1,
                                    condition='finalVols==%d' % VOL_ONE,
                                    label='last Volume only',
                                    help='')
        group.addParam('equalVol', params.IntParam, default=1,
                      condition='finalVols==%d' % VOL_TWO,
                      label='equal space volume selection',
                      help='')

        form.addParam('numOfVols', params.IntParam,
                      default=5, label='Number of Volumes',
                      help='Select Volumes to work with.')
        form.addParam('referenceMask', params.PointerParam,
                      pointerClass='VolumeMask', expertLevel=em.LEVEL_ADVANCED,
                      label='Reference mask (optional)', allowsNull=True,
                      help='A volume mask containing a (soft) mask with '
                           'the same dimensions as the reference(s), '
                           'and values between 0 and 1, with 1 being 100% '
                           'protein and 0 being 100% solvent. The '
                           'reconstructed reference map will be multiplied '
                           'by this mask. If no mask is given, a soft '
                           'spherical mask based on the <radius> of the '
                           'mask for the experimental images will be '
                           'applied.\n\n'
                           'In some cases, for example for non-empty '
                           'icosahedral viruses, it is also useful to use '
                           'a second mask. Check _Advaced_ level and '
                           'select another volume mask')
        form.addParam('solventMask', params.PointerParam,
                      pointerClass='VolumeMask',
                      expertLevel=em.LEVEL_ADVANCED, allowsNull=True,
                      label='Second reference mask (optional)',
                      help='For all white (value 1) pixels in this second '
                           'mask the corresponding pixels in the '
                           'reconstructed map are set to the average value '
                           'of these pixels. Thereby, for example, the '
                           'higher density inside the virion may be set to '
                           'a constant. Note that this second mask should '
                           'have one-values inside the virion and '
                           'zero-values in the capsid and the solvent '
                           'areas.')
        form.addParam('solventFscMask', params.BooleanParam, default=False,
                      expertLevel=em.LEVEL_ADVANCED,
                      label='Use solvent-flattened FSCs?',
                      help='If set to Yes, then instead of using '
                           'unmasked maps to calculate the gold-standard '
                           'FSCs during refinement, masked half-maps '
                           'are used and a post-processing-like '
                           'correction of the FSC curves (with '
                           'phase-randomisation) is performed every '
                           'iteration. This only works when a reference '
                           'mask is provided on the I/O tab. This may '
                           'yield higher-resolution maps, especially '
                           'when the mask contains only a relatively '
                           'small volume inside the box.')
        form.addParam('isMapAbsoluteGreyScale', params.BooleanParam,
                      default=False,
                      label="Is initial 3D map on absolute greyscale?",
                      help='The probabilities are based on squared '
                           'differences, so that the absolute grey scale is '
                           'important. \n'
                           'Probabilities are calculated based on a Gaussian '
                           'noise model, which contains a squared difference '
                           'term between the reference and the experimental '
                           'image. This has a consequence that the reference '
                           'needs to be on the same absolute intensity '
                           'grey-scale as the experimental images. RELION and '
                           'XMIPP reconstruct maps at their absolute '
                           'intensity grey-scale. Other packages may perform '
                           'internal normalisations of the reference density, '
                           'which will result in incorrect grey-scales. '
                           'Therefore: if the map was reconstructed in RELION '
                           'or in XMIPP, set this option to Yes, otherwise '
                           'set it to No. If set to No, RELION will use a ('
                           'grey-scale invariant) cross-correlation criterion '
                           'in the first iteration, and prior to the second '
                           'iteration the map will be filtered again using '
                           'the initial low-pass filter. This procedure is '
                           'relatively quick and typically does not '
                           'negatively affect the outcome of the subsequent '
                           'MAP refinement. Therefore, if in doubt it is '
                           'recommended to set this option to No.')
        form.addParam('symmetryGroup', params.StringParam, default='c1',
                      label="Symmetry",
                      help='If the molecule is asymmetric, set Symmetry '
                           'group to C1. Note their are multiple '
                           'possibilities for icosahedral symmetry:\n'
                           '* I1: No-Crowther 222 (standard in Heymann,'
                           'Chagoyen  & Belnap, JSB, 151 (2005) 196-207)\n'
                           '* I2: Crowther 222                          \n'
                           '* I3: 52-setting (as used in SPIDER?)       \n'
                           '* I4: A different 52 setting                \n'
                           'The command *relion_refine --sym D2 '
                           '--print_symmetry_ops* prints a list of all '
                           'symmetry operators for symmetry group D2. RELION '
                           'uses MIPP\'s libraries for symmetry operations. '
                           'Therefore, look at the XMIPP Wiki for more '
                           'details:\n'
                           'http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp'
                           '/WebHome?topic=Symmetry')
        form.addParam('initialLowPassFilterA', params.FloatParam,
                      default=25 if self.IS_VOLSELECTOR else 40,
                      label='Initial low-pass filter (A)',
                      help='It is recommended to strongly low-pass filter '
                           'your initial reference map. If it has not yet '
                           'been low-pass filtered, it may be done '
                           'internally using this option. If set to 0, '
                           'no low-pass filter will be applied to the '
                           'initial reference(s).')

        form.addSection('CTF')
        form.addParam('doCTF', params.BooleanParam, default=True,
                      expertLevel=expertLev,
                      label='Do CTF-correction?',
                      help='If set to Yes, CTFs will be corrected inside the '
                           'MAP refinement. The resulting algorithm '
                           'intrinsically implements the optimal linear, or '
                           'Wiener filter. Note that input particles should '
                           'contains CTF parameters.')
        form.addParam('hasReferenceCTFCorrected', params.BooleanParam,
                      default=False, expertLevel=expertLev,
                      label='Has reference been CTF-corrected?',
                      help='Set this option to Yes if the reference map '
                           'represents CTF-unaffected density, e.g. it was '
                           'created using Wiener filtering inside RELION or '
                           'from a PDB. If set to No, then in the first '
                           'iteration, the Fourier transforms of the reference '
                           'projections are not multiplied by the CTFs.')
        form.addParam('haveDataBeenPhaseFlipped', params.LabelParam,
                      expertLevel=expertLev,
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
        form.addParam('ignoreCTFUntilFirstPeak', params.BooleanParam,
                      default=False, expertLevel=em.LEVEL_ADVANCED,
                      label='Ignore CTFs until first peak?',
                      help='If set to Yes, then CTF-amplitude correction will '
                           'only be performed from the first peak '
                           'of each CTF onward. This can be useful if the CTF '
                           'model is inadequate at the lowest resolution. '
                           'Still, in general using higher amplitude contrast '
                           'on the CTFs (e.g. 10-20%) often yields better '
                           'results. Therefore, this option is not generally '
                           'recommended.')
        form.addParam('doCtfManualGroups', params.BooleanParam,
                      default=False,
                      label='Do manual grouping ctfs?',
                      expertLevel=expertLev,
                      help='Set this to Yes the CTFs will grouping manually.')
        form.addParam('defocusRange', params.FloatParam, default=500,
                      label='defocus range for group creation (in Angstroms)',
                      condition='doCtfManualGroups', expertLevel=expertLev,
                      help='Particles will be grouped by defocus.'
                           'This parameter is the bin for an histogram.'
                           'All particles assigned to a bin form a group')
        form.addParam('numParticles', params.FloatParam, default=200,
                      label='minimum size for defocus group',
                      condition='doCtfManualGroups', expertLevel=expertLev,
                      help='If defocus group is smaller than this value, '
                           'it will be expanded until number of particles '
                           'per defocus group is reached')

        form.addSection(label='Optimisation')

        form.addParam('regularisationParamT', params.IntParam, default=4,
                      expertLevel=expertLev,
                      label='Regularisation parameter T',
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
        form.addParam('numberOfIterations', params.IntParam, default=2,
                      expertLevel=expertLev,
                      label='Number of iterations',
                      help='Number of iterations to be performed. Note '
                           'that the current implementation does NOT '
                           'comprise a convergence criterium. Therefore, '
                           'the calculations will need to be stopped '
                           'by the user if further iterations do not yield '
                           'improvements in resolution or classes.')

        form.addParam('limitResolEStep', params.FloatParam, default=-1,
                      expertLevel=em.LEVEL_ADVANCED,
                      label='Limit resolution E-step to (A)',
                      help='If set to a positive number, then the '
                           'expectation step (i.e. the alignment) will be '
                           'done only including the Fourier components up '
                           'to this resolution (in Angstroms). This is '
                           'useful to prevent overfitting, as the '
                           'classification runs in RELION are not to be '
                           'guaranteed to be 100% overfitting-free (unlike '
                           'the _3D auto-refine_ with its gold-standard '
                           'FSC). In particular for very difficult data '
                           'sets, e.g. of very small or featureless '
                           'particles, this has been shown to give much '
                           'better class averages. In such cases, values '
                           'in the range of 7-12 Angstroms have proven '
                           'useful.')

        form.addSection('Sampling')
        form.addParam('angularSamplingDeg', params.EnumParam, default=1,
                      choices=ANGULAR_SAMPLING_LIST,
                      expertLevel=expertLev, condition=cond,
                      label='Angular sampling interval (deg)',
                      help='There are only a few discrete angular samplings'
                           ' possible because we use the HealPix library to'
                           ' generate the sampling of the first two Euler '
                           'angles on the sphere. The samplings are '
                           'approximate numbers and vary slightly over '
                           'the sphere.')
        form.addParam('offsetSearchRangePix', params.FloatParam,
                      default=5, expertLevel=expertLev, condition=cond,
                      label='Offset search range (pix)',
                      help='Probabilities will be calculated only for '
                           'translations in a circle with this radius (in '
                           'pixels). The center of this circle changes at '
                           'every iteration and is placed at the optimal '
                           'translation for each image in the previous '
                           'iteration.')
        form.addParam('offsetSearchStepPix', params.FloatParam,
                      default=1.0, expertLevel=expertLev, condition=cond,
                      label='Offset search step (pix)',
                      help='Translations will be sampled with this step-size '
                           '(in pixels). Translational sampling is also done '
                           'using the adaptive approach. Therefore, if '
                           'adaptive=1, the translations will first be '
                           'evaluated on a 2x coarser grid.')

        form.addParam('localAngularSearch', params.BooleanParam,
                      default=False, expertLevel=expertLev,
                      condition=cond,
                      label='Perform local angular search?',
                      help='If set to Yes, then rather than performing '
                           'exhaustive angular searches, local searches '
                           'within the range given below will be performed.'
                           ' A prior Gaussian distribution centered at the '
                           'optimal orientation in the previous iteration '
                           'and with a stddev of 1/3 of the range given '
                           'below will be enforced.')
        form.addParam('localAngularSearchRange', params.FloatParam,
                      default=5.0, expertLevel=expertLev,
                      condition=cond,
                      label='Local angular search range',
                      help='Local angular searches will be performed '
                           'within +/- the given amount (in degrees) from '
                           'the optimal orientation in the previous '
                           'iteration. A Gaussian prior (also see previous '
                           'option) will be applied, so that orientations '
                           'closer to the optimal orientation in the '
                           'previous iteration will get higher weights '
                           'than those further away.')

        form.addSection('Compute')
        form.addParam('useParallelDisk', params.BooleanParam, default=True,
                      label='Use parallel disc I/O?',
                      help='If set to Yes, all MPI slaves will read '
                           'their own images from disc. Otherwise, only '
                           'the master will read images and send them '
                           'through the network to the slaves. Parallel '
                           'file systems like gluster of fhgfs are good '
                           'at parallel disc I/O. NFS may break with many '
                           'slaves reading in parallel.')
        form.addParam('pooledParticles', params.IntParam, default=3,
                      label='Number of pooled particles:',
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

        form.addParam('skipPadding', em.BooleanParam, default=False,
                      label='Skip padding',
                      help='If set to Yes, the calculations will not use '
                           'padding in Fourier space for better '
                           'interpolation in the references. Otherwise, '
                           'references are padded 2x before Fourier '
                           'transforms are calculated. Skipping padding '
                           '(i.e. use --pad 1) gives nearly as good '
                           'results as using --pad 2, but some artifacts '
                           'may appear in the corners from signal that is '
                           'folded back.')

        form.addParam('allParticlesRam', params.BooleanParam, default=False,
                      label='Pre-read all particles into RAM?',
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
        form.addParam('scratchDir', params.PathParam,
                      condition='not allParticlesRam',
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
        form.addParam('combineItersDisc', params.BooleanParam,
                      default=False,
                      label='Combine iterations through disc?',
                      help='If set to Yes, at the end of every iteration '
                           'all MPI slaves will write out a large file '
                           'with their accumulated results. The MPI '
                           'master will read in all these files, combine '
                           'them all, and write out a new file with the '
                           'combined results. All MPI salves will then '
                           'read in the combined results. This reduces '
                           'heavy load on the network, but increases load '
                           'on the disc I/O. This will affect the time it '
                           'takes between the progress-bar in the '
                           'expectation step reaching its end (the mouse '
                           'gets to the cheese) and the start of the '
                           'ensuing maximisation step. It will depend on '
                           'your system setup which is most efficient.')
        form.addParam('doGpu', params.BooleanParam, default=True,
                      label='Use GPU acceleration?',
                      help='If set to Yes, the job will try to use GPU '
                           'acceleration.')
        form.addParam('gpusToUse', params.StringParam, default='',
                      label='Which GPUs to use:', condition='doGpu',
                      help='This argument is not necessary. If left empty, '
                           'the job itself will try to allocate available '
                           'GPU resources. You can override the default '
                           'allocation by providing a list of which GPUs '
                           '(0,1,2,3, etc) to use. MPI-processes are '
                           'separated by ":", threads by ",". '
                           'For example: "0,0:1,1:0,0:1,1"')
        form.addParam('oversampling', params.IntParam, default=1,
                      label="Over-sampling",
                      help="Adaptive oversampling order to speed-up "
                           "calculations (0=no oversampling, 1=2x, 2=4x, etc)")
        form.addParam('extraParams', params.StringParam,
                      default='',
                      label='Additional parameters',
                      help="In this box command-line arguments may be "
                           "provided that are not generated by the GUI. This "
                           "may be useful for testing developmental options "
                           "and/or expert use of the program, e.g:\n"
                           "--dont_combine_weights_via_disc\n"
                           "--verb 1\n"
                           "--pad 2")
        group.addParam('numberOfClasses', params.IntParam, default=3,
                       label='Number of classes:',
                       help='The number of classes (K) for a multi-reference '
                            'refinement. These classes will be made in an '
                            'unsupervised manner from a single reference by '
                            'division of the data into random subsets during '
                            'the first iteration.')

        form.addParallelSection(threads=1, mpi=4)

    def _defineConstants(self):
        self.IS_3D = not self.IS_2D

    def _printWarnings(self, *lines):
        """ Print some warning lines to 'warnings.xmd',
        the function should be called inside the working dir."""
        fWarn = open("warnings.xmd", 'wa')
        for l in lines:
            print >> fWarn, l
        fWarn.close()



    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):

        self._insertFunctionStep('convertVolumeStep')
        self._insertFunctionStep('computeNMAStep')
        self._insertFunctionStep('convertPdbStep')
        self._insertFunctionStep('particleAttrStep')
        self._insertFunctionStep('estimatePCAStep')

        # self._insertFunctionStep('createOutputStep')


    #--------------------------- STEP functions --------------------------------
    def convertVolumeStep(self):
        inputVol = self.inputVolume.get()
        print (inputVol, "inputVol")
        fnMask = None
        fnIn = getImageLocation(inputVol)
        # fn_one= self._getExtraPath()
        # print (fnIn, "fnIn")
        # self.runJob("xmipp_image_convert",
        #         "-i %s -o %s/output_vol.mrc:mrc -t vol"
        #         % (fnIn, fn_one),
        #         numberOfMpi=1, numberOfThreads=1)
        # fnOut= self._getExtraPath("output_vol.mrc:mrc")
        # print (fnOut, "fnOut")
        # outFile = self._getPath(replaceBaseExt(basename(fnIn), 'mrc'))
        #
        # self.info("Output file: " + outFile)




        if self.maskMode == NMA_MASK_THRE:
            fnMask = self._getExtraPath('mask.vol')
            maskParams = '-i %s -o %s --select below %f --substitute binarize' \
                         % (fnIn, fnMask, self.maskThreshold.get())
            self.runJob('xmipp_transform_threshold', maskParams,
                        numberOfMpi=1, numberOfThreads=1)
        elif self.maskMode == NMA_MASK_FILE:
            fnMask = getImageLocation(self.volumeMask.get())

        print ("fnmask1")
        pseudoatoms = 'pseudoatoms'
        outputFn = self._getPath(pseudoatoms)
        print (outputFn, "outputFn")
        sampling = inputVol.getSamplingRate()
        sigma = sampling * self.pseudoAtomRadius.get()
        targetErr = self.pseudoAtomTarget.get()
        nthreads = self.numberOfThreads.get() * self.numberOfMpi.get()
        params = "-i %(fnIn)s -o %(outputFn)s --sigma %(sigma)f --thr " \
                 "%(nthreads)d "
        params += "--targetError %(targetErr)f --sampling_rate %(sampling)f " \
                  "-v 2 --intensityColumn Bfactor"

        print ("fnmask1")

        if fnMask:
            params += " --mask binary_file %(fnMask)s"
        self.runJob("xmipp_volume_to_pseudoatoms", params % locals(),
                        numberOfMpi=1, numberOfThreads=1)
        for suffix in ["_approximation.vol", "_distance.hist"]:
            moveFile(self._getPath(pseudoatoms + suffix),
                     self._getExtraPath(pseudoatoms + suffix))
        self.runJob("xmipp_image_convert",
                    "-i %s_approximation.vol -o %s_approximation.mrc -t vol"
                    % (self._getExtraPath(pseudoatoms),
                       self._getExtraPath(pseudoatoms)),
                        numberOfMpi=1, numberOfThreads=1)
        self.runJob("xmipp_image_header",
                    "-i %s_approximation.mrc --sampling_rate %f" %
                    (self._getExtraPath(pseudoatoms), sampling),
                        numberOfMpi=1, numberOfThreads=1)
        cleanPattern(self._getPath(pseudoatoms + '_*'))

    def computeNMAStep(self):
        # self.structureEM = self._getPath('pseudoatoms.pdb')
        n = self.numberOfModes.get()


        # Link the input
        pseudoFn = 'pseudoatoms.pdb'
        distanceFn = 'atoms_distance.hist'
        inputFn = self._getPath(pseudoFn)
        localFn = self._getPath(replaceBaseExt(basename(inputFn), 'pdb'))
        """ Copy the input pdb file and also create a link 'atoms.pdb'
         """
        cifToPdb(inputFn, localFn)

        if not os.path.exists(inputFn):
            createLink(localFn, inputFn)


        # Construct string for relative-absolute cutoff
        # This is used to detect when to reexecute a step or not
        cutoffStr = ''

        if self.cutoffMode == NMA_CUTOFF_REL:
            cutoffStr = 'Relative %f' % self.rcPercentage.get()
        else:
            cutoffStr = 'Absolute %f' % self.rc.get()

        print("cutoffStr", cutoffStr)
        # Compute modes
        if self.cutoffMode == NMA_CUTOFF_REL:
            print ("NMA_CUTOFF_REL", NMA_CUTOFF_REL)
            params = '-i %s --operation distance_histogram %s' \
                     % (localFn, self._getExtraPath(distanceFn))
            self.runJob("xmipp_pdb_analysis", params,
                        numberOfMpi=1, numberOfThreads=1)
            print "i ran xmipp pdb analysis"

        # fnBase = localFn.replace(".pdb", "")
        fnDistanceHist = self._getExtraPath(distanceFn)
        (baseDir, fnBase) = os.path.split(fnDistanceHist)
        rc = self._getRc(fnDistanceHist)
        self._enterWorkingDir()
        self.runJob('nma_record_info.py',
                    "%d %s %d" % (self.numberOfModes.get(), pseudoFn, rc),
                    env=getNMAEnviron(),
                        numberOfMpi=1, numberOfThreads=1)

        self.runJob("nma_pdbmat.pl", "pdbmat.dat", env=getNMAEnviron(),
                        numberOfMpi=1, numberOfThreads=1)
        self.runJob("nma_diag_arpack", "", env=getNMAEnviron(),
                        numberOfMpi=1, numberOfThreads=1)
        if not exists("fort.11"):
            self._printWarnings(redStr(
                'Modes cannot be computed. Check the number of '
                'modes you asked to compute and/or consider increasing '
                'cut-off distance. The maximum number of modes allowed by '
                'the method for pseudoatomic normal mode analysis is 3 times '
                'the number of pseudoatoms but the protocol allows only up to '
                '200 modes as 20-100 modes are usually enough.  '
                'If the number of modes is below the minimum between 200 and 3 '
                'times the number of pseudoatoms, consider increasing cut-off distance.'))
        cleanPath("diag_arpack.in", "pdbmat.dat")

        # self._leaveWorkingDir()
        n = self._countAtoms(pseudoFn)
        self.runJob("nma_reformat_vector_foranimate.pl", "%d fort.11" % n,
                    env=getNMAEnviron(),
                        numberOfMpi=1, numberOfThreads=1)
        self.runJob("cat", "vec.1* > vec_ani.txt",
                        numberOfMpi=1, numberOfThreads=1)
        self.runJob("rm", "-f vec.1*",
                        numberOfMpi=1, numberOfThreads=1)
        self.runJob("nma_reformat_vector.pl", "%d fort.11" % n,
                    env=getNMAEnviron(),
                        numberOfMpi=1, numberOfThreads=1)
        fnModesDir = "modes"
        makePath(fnModesDir)
        self.runJob("mv", "-f vec.* %s" % fnModesDir,
                        numberOfMpi=1, numberOfThreads=1)
        self.runJob("nma_prepare_for_animate.py", "", env=getNMAEnviron(),
                        numberOfMpi=1, numberOfThreads=1)
        self.runJob("rm", "-f vec_ani.txt fort.11 matrice.sdijf",
                        numberOfMpi=1, numberOfThreads=1)
        moveFile('vec_ani.pkl', 'extra/vec_ani.pkl')



        #self.PseudoAtomThreshold = 0.0

        fnVec = glob("modes/vec.*")

        if len(fnVec) < self.numberOfModes.get():
            msg = "There are only %d modes instead of %d. "
            msg += "Check the number of modes you asked to compute and/or consider increasing cut-off distance."
            msg += "The maximum number of modes allowed by the method for atomic normal mode analysis is 6 times"
            msg += "the number of RTB blocks and for pseudoatomic normal mode analysis 3 times the number of pseudoatoms. "
            msg += "However, the protocol allows only up to 200 modes as 20-100 modes are usually enough. If the number of"
            msg += "modes is below the minimum between these two numbers, consider increasing cut-off distance."
            self._printWarnings(redStr(msg % (len(fnVec), self.numberOfModes.get())))
            print redStr('Warning: There are only %d modes instead of %d.' % (
            len(fnVec), self.numberOfModes.get()))
            print redStr(
                "Check the number of modes you asked to compute and/or consider increasing cut-off distance.")
            print redStr(
                "The maximum number of modes allowed by the method for atomic normal mode analysis is 6 times")
            print redStr(
                "the number of RTB blocks and for pseudoatomic normal mode analysis 3 times the number of pseudoatoms.")
            print redStr(
                "However, the protocol allows only up to 200 modes as 20-100 modes are usually enough. If the number of")
            print redStr(
                "modes is below the minimum between these two numbers, consider increasing cut-off distance.")

        fnDiag = "diagrtb.eigenfacs"


        self.runJob("nma_reformatForElNemo.sh", "%d" % len(fnVec),
                    env=getNMAEnviron(),
                        numberOfMpi=1, numberOfThreads=1)
        fnDiag = "diag_arpack.eigenfacs"

        self.runJob("echo", "%s | nma_check_modes" % fnDiag,
                    env=getNMAEnviron(),
                        numberOfMpi=1, numberOfThreads=1)
        cleanPath(fnDiag)


        fh = open("Chkmod.res")
        mdOut = xmippLib.MetaData()
        collectivityList = []
        print (fnVec, "fnVec")
        for n in range(len(fnVec)):
            print (n, "n")
            line = fh.readline()
            print (line, "line")
            collectivity = float(line.split()[1])
            collectivityList.append(collectivity)
            print (collectivity, "collectivity")
            print (collectivityList, "collectivityList")

            objId = mdOut.addObject()
            modefile = self._getPath("modes", "vec.%d" % (n + 1))
            print (modefile, "modefile")
            mdOut.setValue(xmippLib.MDL_NMA_MODEFILE, modefile, objId)
            print (xmippLib.MDL_NMA_MODEFILE, "xmippLib.MDL_NMA_MODEFILE")
            mdOut.setValue(xmippLib.MDL_ORDER, long(n + 1), objId)
            print (xmippLib.MDL_ORDER, "xmippLib.MDL_ORDER")

            if n >= 6:
                mdOut.setValue(xmippLib.MDL_ENABLED, 1, objId)
                print (xmippLib.MDL_ENABLED, "xmippLib.MDL_ENABLED")
            else:
                mdOut.setValue(xmippLib.MDL_ENABLED, -1, objId)
                print (xmippLib.MDL_ENABLED, "xmippLib.MDL_ENABLED")
            mdOut.setValue(xmippLib.MDL_NMA_COLLECTIVITY, collectivity, objId)
            print (mdOut, "mdOut")
            if collectivity < self.collectivityThreshold.get():
                mdOut.setValue(xmippLib.MDL_ENABLED, -1, objId)
        fh.close()
        idxSorted = [i[0] for i in
                     sorted(enumerate(collectivityList), key=lambda x: x[1],
                            reverse=True)]

        print (collectivityList, "collectivityList")
        print (collectivity, "collectivity")

        score = []
        for j in range(len(fnVec)):
            score.append(0)

        modeNum = []
        l = 0
        for k in range(len(fnVec)):
            modeNum.append(k)
            l += 1
        print(score, "score")
        print(idxSorted, "idxSorted")
        print(modeNum, 'modeNum')
        # score = [0]*numberOfModes
        for i in range(len(fnVec)):
            score[idxSorted[i]] = idxSorted[i] + modeNum[i] + 2
        print(score, 'score')
        i = 0
        print (mdOut, "mdOut")
        for objId in mdOut:
            print (objId, "objId")
            # print (mdOut, "mdOut")
            score_i = float(score[i]) / (2.0 * l)
            print (score[i], "score[i]")
            print (l, "l")
            mdOut.setValue(xmippLib.MDL_NMA_SCORE, score_i, objId)
            print (xmippLib.MDL_NMA_SCORE, "xmippLib.MDL_NMA_SCORE")
            i += 1
            print(score_i, "score_i")

        suffix = ''
        mdOut.write("modes%s.xmd" % suffix)
        cleanPath("Chkmod.res")

#
        # print("JV")
        # import sys
        # sys.exit()
#

        makePath('extra/animations')

        fn = "pseudoatoms.pdb"
        print ("animation step eneter")
        self.runJob("nma_animate_pseudoatoms.py",
                    "%s extra/vec_ani.pkl 7 %d "
                    "%f extra/animations/"
                    "animated_mode %d %d %f" % \
                    (fn, self.numberOfModes.get(), self.amplitude.get(),
                     self.nframes.get(), self.downsample.get(),
                     self.pseudoAtomThreshold.get()), env=getNMAEnviron(),
                        numberOfMpi=1, numberOfThreads=1)
        print "madre mia animaaaaaaaaaaaa"

        for mode in range(7, self.numberOfModes.get() + 1):
            fnAnimation = join("extra", "animations", "animated_mode_%03d"
                               % mode)
            fhCmd = open(fnAnimation + ".vmd", 'w')
            fhCmd.write("mol new %s.pdb\n" % self._getPath(fnAnimation))
            fhCmd.write("animate style Loop\n")
            fhCmd.write("display projection Orthographic\n")

            fhCmd.write("mol modcolor 0 0 Beta\n")
            fhCmd.write("mol modstyle 0 0 Beads %f 8.000000\n"
                        % (self.pseudoAtomThreshold.get()))

            fhCmd.write("animate speed 0.5\n")
            fhCmd.write("animate forward\n")
            fhCmd.close();


        fnOutDir = self._getExtraPath("distanceProfiles")
        maxShift = []
        maxShiftMode = []

        for n in range(7, self.numberOfModes.get() + 1):
            fnVec = self._getPath("modes", "vec.%d" % n)
            if exists(fnVec):
                fhIn = open(fnVec)
                md = xmippLib.MetaData()
                atomCounter = 0
                for line in fhIn:
                    x, y, z = map(float, line.split())
                    d = math.sqrt(x * x + y * y + z * z)
                    if n == 7:
                        maxShift.append(d)
                        maxShiftMode.append(7)
                    else:
                        if d > maxShift[atomCounter]:
                            maxShift[atomCounter] = d
                            maxShiftMode[atomCounter] = n
                    atomCounter += 1
                    md.setValue(xmippLib.MDL_NMA_ATOMSHIFT, d, md.addObject())
                md.write(join(fnOutDir, "vec%d.xmd" % n))
                fhIn.close()
        md = xmippLib.MetaData()
        for i, _ in enumerate(maxShift):
            fnVec = self._getPath("modes", "vec.%d" % (maxShiftMode[i] + 1))
            if exists(fnVec):
                objId = md.addObject()
                md.setValue(xmippLib.MDL_NMA_ATOMSHIFT, maxShift[i], objId)
                md.setValue(xmippLib.MDL_NMA_MODEFILE, fnVec, objId)
        md.write(self._getExtraPath('maxAtomShifts.xmd'))

        # import shutil
        # from shutil import copyfile
        # makePath('extra/split_animations')
        # splitPdb = self._getExtraPath("split_animations")
        # pdbFns = self._getExtraPath("animations", "*.pdb")
        # # print(splitPdb, type(splitPdb), self._currentDir, "splitPdbvvvvvvvvvvvv")
        # fnList = glob(pdbFns)


        # for i, pdbFns in enumerate(fnList):
        #     print (pdbFns, "pdbfnsssssssssss")
        #     with open(pdbFns) as infile:
        #         outfile = open(str(i) + pdbFns, 'w')
        #         for line in infile:
        #             if 'TER' in line:
        #                 outfile = open(str(i) + pdbFns, 'w')
        #             outfile.write(line)









         # with open(splitPdb, 'w') as f:
         #    for s in fnList:
         #        f.write((s + u'\n').encode('unicode-escape'))
        #
        # with open(splitPdb, 'r') as f:
        #     splitPdb = [line.decode('unicode-escape').rstrip(u'\n') for line in
        #                f]
        # print (splitPdb)
        # for fn in (fnList, self.numberOfModes.get() + 1):
        #     pdb_split = join("extra", "split_animations", "split_animations_%03d"
        #                        % fn)
        #     pdb_sfile = open(pdb_split + ".pdb", 'w')
        #     pdb_sfile.write('%s\n' % fn)
        #     print ("splittttttttt")
        # MyFile = open(splitPdb, 'w')
        # for element in fnList:
        #     MyFile.write(element)
        #     MyFile.write('\n')
        # MyFile.close()




        self._leaveWorkingDir()

    def _validate(self):
        errors = []
        nmaBin = Plugin.getVar(NMA_HOME)
        nma_programs = ['nma_check_modes',
                        'nma_diag_arpack',
                        'nma_diagrtb',
                        'nma_elnemo_pdbmat']
        # Check Xmipp was compiled with NMA flag to True and
        # some of the nma programs are under the Xmipp/bin/ folder
        for prog in nma_programs:
            if not exists(join(nmaBin, prog)):
                errors.append("Some NMA programs are missing in the NMA folder.")
                errors.append("Check that Scipion was installed with NMA: 'scipion install nma'")
                break
        from pyworkflow.utils.which import which
        if which("csh") == "":
            errors.append("Cannot find csh in the PATH")

        return errors

    def _countAtoms(self, fnPDB):
        fh = open(fnPDB, 'r')
        n = 0
        for line in fh:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                n += 1
        fh.close()
        return n

    def _getRc(self, fnDistanceHist):
        if self.cutoffMode == NMA_CUTOFF_REL:
            rc = self._computeCutoff(fnDistanceHist, self.rcPercentage.get())
        else:
            rc = self.rc.get()
        return rc

    def _computeCutoff(self, fnHist, rcPercentage):
        mdHist = xmippLib.MetaData(fnHist)
        distances = mdHist.getColumnValues(xmippLib.MDL_X)
        distanceCount = mdHist.getColumnValues(xmippLib.MDL_COUNT)
        # compute total number of distances
        nCounts = 0
        for count in distanceCount:
            nCounts += count
        # Compute threshold
        NcountThreshold = nCounts * rcPercentage / 100.0
        nCounts = 0
        for i in range(len(distanceCount)):
            nCounts += distanceCount[i]
            if nCounts > NcountThreshold:
                rc = distances[i]
                break
        msg = "Cut-off distance = %s A" % rc
        print(msg)
        self._enterWorkingDir()
        self._printWarnings(msg)
        self._leaveWorkingDir()
        return rc

    # --------------------------------convert pdb------------------------------
    def convertPdbStep(self):

        pdbFns = self._getExtraPath("animations", "*.pdb")
        print(pdbFns, type(pdbFns), self._currentDir)
        fnListl = glob(pdbFns)
        print (fnListl, "fnlisttttttttttttttt")
        fnList = []
        for pdbFns in fnListl:
            print (pdbFns, "pdbfnsssssssssss")
            with open(pdbFns) as infile:
                i = 0
                filename = pdbFns[:-4] + "_" + str(i) + '.pdb'
                outfile = open(filename, 'w')
                # fnList.append(filename)
                for line in infile:
                    if 'TER' in line:
                        i += 1
                        if i == self.nframes.get():
                            break
                        filename = pdbFns[:-4] + "_" + str(i) + '.pdb'
                        outfile = open(filename, 'w')
                        fnList.append(filename)
                    elif 'ENDMDL' not in line:
                        outfile.write(line)
                fnList = fnList[:-1]

        # fnListl = sorted(fnList)
        print (fnList, "fnlist")
        pseudoFn = 'pseudoatoms.pdb'
        inputFn = self._getPath(pseudoFn)
        print(inputFn, type(inputFn), self._currentDir)
        fnList.append(inputFn)
        inputVol = self.inputVolume.get()
        sampling = inputVol.getSamplingRate()
        size = inputVol.getDim()[0]

        for fn in fnList:
            outFile = removeExt(self._getExtraPath(replaceBaseExt(fn, "vol")))
            fixed_Gaussian= sampling* self.pseudoAtomRadius.get()
            args = '-i %s --sampling %f --fixed_Gaussian %f -o %s' % (fn, sampling, fixed_Gaussian, outFile)

            if self.centerPdb:
                args += ' --centerPDB'

            args += ' --size %d' % size

            self.info("Input file: " + fn)
            self.info("Output file: " + outFile)

            program = "xmipp_volume_from_pdb"
            self.runJob(program, args,
                        numberOfMpi=1, numberOfThreads=1)


    #--------------------------paticle attractor step-----------------------
    def particleAttrStep(self):
        totalVolumes = self._getExtraPath("*.vol")
        fnList = glob(totalVolumes)
        sizeList = len(fnList)
        pseudoFn = 'pseudoatoms.vol'

        inputFn = self._getExtraPath(pseudoFn)
        print (inputFn, 'inputFn')
        for i in fnList:
            # print (i, sizeList)
            if (inputFn==i):
                index = fnList.index(i)
                fnList.pop(index)
                break

        selectedVols = self.numOfVols.get()



        print (selectedVols, "selectedVols")

        # b = np.log((1 - (float(selectedVols) / float(sizeList))))
        # numOfRuns = int(-3 / b)
        numOfRuns= 3
        for run in range(numOfRuns):
            self._createFilenameTemplates()
            self._createTemplates(run)
            self._rLev = run
            self._imgFnList = []
            imgSet = self.inputParticles.get()
            imgStar = self._getFileName('input_star', run=run)
            os.makedirs(self._getExtraPath('run_%02d' % run))
            subset = em.SetOfParticles(filename=":memory:")
            print (run, "runn")
            newIndex = 1
            for img in imgSet.iterItems(orderBy='RANDOM()', direction='ASC'):
                self._scaleImages(newIndex, img)
                newIndex += 1
                subset.append(img)
                subsetSize = self.subsetSize.get() * self.numOfVols.get()
                # print (subsetSize, "subsetSizeeeee")
                minSize = min(subsetSize, imgSet.getSize())
                # print (imgSet.getSize(), "imgSet.getSize()")
                # print (minSize, "minSizeeeeeeeeeee")
                if subsetSize > 0 and subset.getSize() == minSize:
                    break
            writeSetOfParticles(subset, imgStar, self._getExtraPath(),
                                alignType=em.ALIGN_NONE,
                                postprocessImageRow=self._postprocessParticleRow)
            self._convertRef()

            args = {}
            self._setNormalArgs(args)
            self._setComputeArgs(args)
            params = self._getParams(args)
            params += ' --j %d' % self.numberOfThreads.get()
            self.runJob(self._getProgram(), params)





    # def createOutputStep(self):
    #     # create a SetOfVolumes and define its relations
    #     volumes = self._createSetOfVolumes()
    #     self._fillVolSetFromIter(volumes, self._lastIter())
    #     self._defineOutputs(outputVolumes=volumes)
    #     totalVolumes = self._getExtraPath("*.vol")
    #     fnList = glob(totalVolumes)
    #     sizeList = len(fnList)
    #     self._defineSourceRelation(fnList, volumes)
    #     print (volumes, "volumes")


    #
    def _getAverageVol(self, listVol=[]):

        self._createFilenameTemplates()
        m = []
        totalVolumes = self._getExtraPath("*.vol")
        fnList = glob(totalVolumes)
        sizeList = len(fnList)
        selectedVols = self.numOfVols.get()
        b = np.log((1 - (float(selectedVols) / float(sizeList))))
        # numOfRuns = int(-3 / b)
        numOfRuns= 3
        iter = self.numberOfIterations.get()
        listModelStar = []
        p = ['']
        ref3d = self.numOfVols.get()
        for run in range(numOfRuns):
            for m in range (1, ref3d+1):
                mf = (self._getExtraPath('run_%02d' % run,
                                                 'relion_it%03d_' % iter+
                                                 'class%03d.mrc' % m))
                listModelStar.append(mf)

        print (listModelStar, "list")
        listVol = listModelStar
        print('creating average map: ', listVol)
        avgVol = self._getFileName('avgMap', run=run)
        print (avgVol, "avgVolll")
        #
        # print('alignining each volume vs. reference')
        for vol in listVol:
            print (vol, "vol")

            npVol = loadMrc(vol,  writable=False)
            #saveMrc(vol,self._getExtraPath('kk.mrc'))
            #sys.exit()

            print (npVol, "npVol")
            if vol == listVol[0]:
                dType = npVol.dtype
                #print (dType, "dType")
                npAvgVol = np.zeros(npVol.shape)
                #print (npAvgVol, "npAvgVol")
            npAvgVol += npVol


        print (npAvgVol, "npAvgVol1")
        npAvgVol = np.divide(npAvgVol, len(listVol))
        #print (npAvgVol, "npAvgVol2")
        print('saving average volume')

        saveMrc(npAvgVol.astype(dType), avgVol)
        # preCorr = []
        # print ("PART2")
        # for vol in listVol:
        #     npVol = loadMrc(vol, writable=False)
        #     print (npVol, "npVol2")
        #
        #     subst = npVol - npAvgVol
        #     print (npAvgVol, "npAvgVol")
        #     print (subst, "subst")
        #     preCorr.append(subst)
        #
        # for i in preCorr:
        #     x = i-1
        #     y = i+1
        #     # X = np.stack((x, y), axis=0)
        #     print(np.cov(x, y))
        #
        #     # covMatrix = np.cov(preCorr)
        #
        #     # npAvgVol += npVol


    def estimatePCAStep(self):
        Plugin.setEnviron()
        totalVolumes = self._getExtraPath("*.vol")
        fnList = glob(totalVolumes)
        sizeList = len(fnList)
        selectedVols = self.numOfVols.get()
        # b = np.log((1 - (float(selectedVols) / float(sizeList))))
        # numOfRuns = int(-3 / b)
        numOfRuns= 3
        iter = self.numberOfIterations.get()
        listModelStar = []
        p = ['']
        ref3d = self.numOfVols.get()
        print (ref3d, "ref3d")
        for run in range(numOfRuns):
            for m in range(1, ref3d+1):
                    mf = (self._getExtraPath('run_%02d' % run,
                                             'relion_it%03d_' % iter+
                                             'class%03d.mrc' % m))
                    listModelStar.append(mf)


        listVol = listModelStar
        print (len(listVol), "listvol")
        self._getAverageVol(listVol)
        avgVol = self._getFileName('avgMap', run=run)
        npAvgVol = loadMrc(avgVol, False)
        dType = npAvgVol.dtype

        volNp = loadMrc(listVol.__getitem__(0), False)
        dim = volNp.shape[0]
        lenght = dim ** 3
        listNpVol = []
        cov_matrix= []
        checkList = []
        for vol in listVol:
            volNp = loadMrc(vol, False)
            # Now, not using diff volume to estimate PCA
            # diffVol = volNp - npAvgVol
            volList = volNp.reshape(lenght)
            # listNpVol.append(volList)
            # volList = volList - npAvgVol

            print (npAvgVol, "npAvgVol")

            row = []
            b = volList - npAvgVol.reshape(lenght)
            print (b, 'b')
            for j in listVol:
                npVol = loadMrc(j, writable=False)
                volList = npVol.reshape(lenght)
                volList_two = volList - npAvgVol.reshape(lenght)
                print (volList, "vollist")
                temp_a= np.corrcoef(volList_two, b).item(1)
                print (temp_a, "temp_a")
                row.append(temp_a)
                # b= volList_two
                # print (corr, "corr")
            cov_matrix.append(row)


        print (cov_matrix, "cov_matrix");
            #preCorr.append(sub)

        u, s, vh = np.linalg.svd(cov_matrix)
        cuttOffMatrix = sum(s) * 0.95
        sCut = 0

        print('cuttOffMatrix: ', cuttOffMatrix)
        print (s, "s")
        for i in s:
            print('cuttOffMatrix: ', cuttOffMatrix)
            if (cuttOffMatrix > 0).any():
                print("Pass, i = %s " % i)
                cuttOffMatrix = cuttOffMatrix - i
                sCut += 1
            else:
                break
        print('sCut: ', sCut)

        eigValsFile = 'eigenvalues.txt'
        self._createMFile(s, eigValsFile)

        eigVecsFile = 'eigenvectors.txt'
        self._createMFile(vh, eigVecsFile)

        vhDel = np.transpose(np.delete(vh, np.s_[sCut:vh.shape[1]], axis=0))
        self._createMFile(vhDel, 'matrix_vhDel.txt')

        print(' this is the matrix "vhDel": ', vhDel)
        mat_one= []
        for vol in listVol:
            volNp = loadMrc(vol, False)
            volList = volNp.reshape(lenght)
            print (volList, "volList")
            row_one= []
            for j in listVol:
                npVol = loadMrc(j, writable=False)
                volList_three = npVol.reshape(lenght)
                j_trans = volList_three.transpose()
                matrix_two= np.dot(volList, j_trans)
                row_one.append(matrix_two)
            mat_one.append(row_one)

        matProj = np.dot(mat_one, vhDel)
        # print (newBaseAxis, "newbase")
        # matProj = np.transpose(np.dot(newBaseAxis, mat_one))
        print (matProj, "matProj")
#----------------------
        mdOut = md.MetaData()
        dictMd = {}
        for run in range(numOfRuns):
            it = self.numberOfIterations.get()
            modelFile = self._getFileName('model', ruNum=run, iter=it)
            mdIn = md.MetaData('model_classes@%s' % modelFile)
            for row in md.iterRows(mdIn, md.RLN_MLMODEL_REF_IMAGE):
                mV = row.getValue(md.RLN_MLMODEL_REF_IMAGE)
                lV = row.getValue('rlnClassDistribution')
                dictMd[lV] = mV

        x_proj = [item[0] for item in matProj]
        y_proj = [item[1] for item in matProj]
        # plt.hist2d(x_proj, y_proj, bins=5, cmap='Blues')
        # plt.show()
        plt.figure(figsize=(12, 4));plt.subplot(150)
        plt.hexbin(x_proj, y_proj)
        plt.colorbar();
        plt.tight_layout()
        plt.show()





        # print (newBaseAxis, "newBaseAxis")
        # print (matProj, "matProj")
        #
        # projFile = 'projection_matrix.txt'
        # self._createMFile(matProj, projFile)
        # return matProj



        # checkList_one= []
        # for vol in listVol:
        #     volNp = loadMrc(vol, False)
        #     volList = volNp.reshape(lenght)
        #     # newBaseAxis = []
        #     b = volList - npAvgVol.reshape(lenght)
        #     print (b, 'b')
        #     # vhDel = vhDel.reshape(lenght)
        #     for j in listVol:
        #         if not j in checkList_one:
        #             npVol = loadMrc(j, writable=False)
        #             volList = npVol.reshape(lenght)
        #             volList_two = volList - npAvgVol.reshape(lenght)
        #             print (volList_two, "volList_two")
            #         print (newBaseAxis, "newBaseAxis")
            #         b = volList_two
            # checkList_one.append(vol)
        # print (newBaseAxis, "newBaseAxis")

    def _getLevelPath(self, run):
        return self._getExtraPath('run_%02d' % self._rLev)

    def _createMFile(self, matrix, name='matrix.txt'):
        f = open(name, 'w')
        for list in matrix:
            s = "%s\n" % list
            f.write(s)
        f.close()

        # m = np.asarray(j)
        # fweights = np.asarray(np.arange(m) * 2, dtype=float)
        # print (fweights, "m")
        # print (i, 'I')
        # print ("all clear")
        # p= m.astype(int)
        # f = np.arange(m) * 2
        # p = np.vectorize(np.int(f))
        # print (p, "f")

        # a = np.arange(p) ** 2
        # print (f, "f")
        # print (a, "a")

        #         bias = False
        #         ddof = None
        #         fweights = None
        #         f = fweights
        #         aweights = None
        #         a = aweights
        #         w = f * a
        #         v1 = np.sum(w)
        #         v2 = np.sum(w * a)
        #         cov_mat = np.dot(i, j.T) *v1 / (v1 ** 2 - ddof * v2)

        # ddof=0
        # f= cov_mat.shape[1] - ddof
        # a= f - ddof
        # w = f * a
        # v1 = np.sum(w)
        # v2 = np.sum(w * a)
        # # w_sum = np.average(cov_mat, axis=1, weights=None, returned=True)
        # print(v1, "f")
        # print (v2, "a")
        # cov = cov_mat * v1 / (v1**2 - ddof * v2)

        # covMatrix = np.cov(preCorr)

        # npAvgVol += npVol

        # covMatrix = np.cov(listNpVol)
        # u, s, vh = np.linalg.svd(covMatrix)
        # cuttOffMatrix = sum(s) * 0.95
        # sCut = 0
        #
        # print('cuttOffMatrix & s: ', cuttOffMatrix, s)



    #--------------------------- INFO functions ---------------------------
    def _methods(self):
        summary = []


        return summary

    def _summary(self):
        """ Even if the full set of parameters is available, this function provides
        summary information about an specific run.
        """
        summary = []

        return summary

    def _citations(self):
        return ['Nogales2013']
    #
    #


    #--------------------------- UTLIS functions --------------------------------------------
    def _getPdbFileName(self):
        if self.inputPdbData == IMPORT_FROM_ID:
            return self._getExtraPath('%s.cif' % self.pdbId.get())
        elif self.inputPdbData == IMPORT_OBJ:
            return self.pdbObj.get().getFileName()
        else:
            return self.pdbFile.get()




    #----------------------------volume attr---------------------------
    def _validateNormal(self):
        """ Should be overwritten in subclasses to
        return summary message for NORMAL EXECUTION.
        """
        return []

    def _summaryNormal(self):
        """ Should be overwritten in subclasses to
        return summary message for NORMAL EXECUTION.
        """
        return []

    def _methods(self):
        """ Should be overwritten in each protocol.
        """
        return []

    # -------------------------- UTILS functions ------------------------------
    def _setSamplingArgs(self, args):
        """ Set sampling related params. """
        args['--healpix_order'] = self.angularSamplingDeg.get()
        args['--offset_range'] = self.offsetSearchRangePix.get()
        args['--offset_step'] = (self.offsetSearchStepPix.get() *
                                 self._getSamplingFactor())

    def _getResetDeps(self):
        return "%s, %s, %s" % (self.inputParticles.get().getObjId(),
                               self.inputVolume.get().getObjId(),
                               self.targetResol.get())

    def _getClassId(self, volFile):
        result = None
        s = self._classRegex.search(volFile)
        if s:
            result = int(s.group(1)) # group 1 is 2 digits class number
        return self.volDict[result]

    def _defineInput(self, args):
        args['--i'] = self._getFileName('input_star', run=self._rLev)

    def _defineOutput(self, args):
        args['--o'] = self._getExtraPath('run_%02d/relion' % self._rLev)


    def _fillVolSetFromIter(self, volSet, it):
        volSet.setSamplingRate(self.inputParticles.get().getSamplingRate())
        modelStar = md.MetaData('model_classes@' +
                                self._getFileName('modelFinal', iter=it))
        print (it, "modelStarrrrrrrrrrrrrrr")
        for row in md.iterRows(modelStar):
            fn = row.getValue('rlnReferenceImage')
            fnMrc = fn + ":mrc"
            itemId = self._getClassId(fn)
            classDistrib = row.getValue('rlnClassDistribution')
            accurracyRot = row.getValue('rlnAccuracyRotations')
            accurracyTras = row.getValue('rlnAccuracyTranslations')
            resol = row.getValue('rlnEstimatedResolution')

            if classDistrib > 0:
                vol = em.Volume()
                self._invertScaleVol(fnMrc)
                vol.setFileName(self._getOutputVolFn(fnMrc))
                vol.setObjId(itemId)
                vol._rlnClassDistributionl = em.Float(classDistrib)
                vol._rlnAccuracyRotations = em.Foat(accurracyRot)
                vol._rlnAccuracyTranslations = em.Float(accurracyTras)
                vol._rlnEstimatedResolution = em.Float(resol)
                volSet.append(vol)


    def _getOutputVolFn(self, fn):
        return self._getExtraPath(replaceBaseExt(fn, '_origSize.mrc'))

    def _convertRef(self):
        # if self.finalVols == VOL_ZERO:
        totalVolumes = self._getExtraPath("*.vol")
        fnList = glob(totalVolumes)
        fnListl = sorted(fnList)
        # fnList = fnList.sort()
        print (fnListl, 'fnListsorted')
        sizeList = len(fnListl)
        # print (sizeList, "sizeList")
        refMd = md.MetaData()
        pseudoFn = 'pseudoatoms.vol'
        inputFn = self._getExtraPath(pseudoFn)
        # print (fnList, 'fnListttttttt1')

        counter = 0
        for vol in random.sample(fnListl, sizeList):
            # print (vol, "vol")
            subsetSize = self.numOfVols.get()
            minSize = min(subsetSize, sizeList)
            # print (minSize, "minSize")

            row = md.Row()
            print (row, "row")
            row.setValue(md.RLN_MLMODEL_REF_IMAGE, vol)
            row.addToMd(refMd)
            row.setValue(md.RLN_MLMODEL_REF_IMAGE, inputFn)
            # row.addToMd(refMd)
            counter += 1
            print(counter, "counterrrr")
            if counter == minSize:
                # print (minSize, "minSize")
                break

        refMd.write(self._getRefStar())

        # if self.finalVols == VOL_ONE:
        #     totalVolumes = self._getExtraPath("*.vol")
        #     fnList = glob(totalVolumes)
        #     fnListl = sorted(fnList)
        #     sizeList = len(fnList)
        #     tempList = []
        #     print (range(5, sizeList-1, 6), 'range.1')
        #     for i in range(5, sizeList-1, 6):
        #         tempList.append(fnListl[i])
        #     fnList = tempList
        #     sizelist_a= len(fnList)
        #
        #     print (tempList, 'tempList.1')
        #     print (fnList, 'fnListttttttt1.1')
        #     refMd = md.MetaData()
        #     pseudoFn = 'pseudoatoms.vol'
        #     inputFn = self._getExtraPath(pseudoFn)
        #     counter = 0
        #     print (fnList, "fnlistttt")
        #     print (sizelist_a, "sizelistttt")
        #     for vol in random.sample(fnList, sizelist_a):
        #         # print (vol, "vol")
        #         subsetSize = self.numOfVols.get()
        #         minSize = min(subsetSize, sizeList)
        #         # print (minSize, "minSize")
        #
        #         row = md.Row()
        #         # print (row, "row")
        #         row.setValue(md.RLN_MLMODEL_REF_IMAGE, vol)
        #         row.addToMd(refMd)
        #         row.setValue(md.RLN_MLMODEL_REF_IMAGE, inputFn)
        #         # row.addToMd(refMd)
        #         counter += 1
        #         print(counter, "counterrrr")
        #         if counter == minSize:
        #             # print (minSize, "minSize")
        #             break
        #
        #     refMd.write(self._getRefStar())

        # # new_list = [fnList[1], fnList[3], fnList[5]]
        # if self.maskMode == NMA_MASK_THRE:
        #     fnMask = self._getExtraPath('mask.vol')
        #     maskParams = '-i %s -o %s --select below %f --substitute binarize' \
        #                  % (fnIn, fnMask, self.maskThreshold.get())
        #     self.runJob('xmipp_transform_threshold', maskParams,
        #                 numberOfMpi=1, numberOfThreads=1)
        # elif self.maskMode == NMA_MASK_FILE:
        #     fnMask = getImageLocation(self.volumeMask.get())
        #
        # if self.finalVols == VOL_ONE:
        #     fnList = glob(totalVolumes)
        #     sizeList = len(fnList)
        #     counter = 0
        #     for vol in random.sample(fnList, sizeList):
        #         print (vol, "vol")
        #         subsetSize = self.numOfVols.get()
        #         minSize = min(subsetSize, sizeList)
        #         print (minSize, "minSize")
        #
        #         row = md.Row()
        #         print (row, "row")
        #         row.setValue(md.RLN_MLMODEL_REF_IMAGE, vol)
        #         row.addToMd(refMd)
        #         row.setValue(md.RLN_MLMODEL_REF_IMAGE, inputFn)
        #         # row.addToMd(refMd)
        #         counter += 1
        #         print(counter, "counterrrr")
        #         if counter == minSize:
        #             print (minSize, "minSize")
        #             break
        #
        #     refMd.write(self._getRefStar())

            # print (fnList, 'fnListtttttttttt1')
        # el
        #     print (fnList, 'fnListtttttttttt2')
        # else:
        #     tempList = []
        #     for i in [1, 3, 5]:
        #         tempList.append(fnList[i])
        #     fnList = tempList
        #     print (fnList, 'fnListtttttttttt3')



        print (inputFn, "inputFnnnnnnnnnnnnnnnnn")


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

    def _scaleImages(self,indx, img):
        fn = img.getFileName()
        index = img.getIndex()
        newFn = self._getTmpPath('particles_subset.mrcs')
        xdim = self._getNewDim()

        ih = em.ImageHandler()
        image = ih.read((index, fn))
        image.scale(xdim, xdim)

        image.write((indx, newFn))

        img.setFileName(newFn)
        img.setIndex(indx)
        img.setSamplingRate(self._getPixeSize())

    def _getPixeSize(self):
        partSet = self.inputParticles.get()
        oldSize = partSet.getXDim()
        newSize  = self._getNewDim()
        pxSize = partSet.getSamplingRate() * oldSize / newSize
        return pxSize
    def _getNewDim(self):
        tgResol = self.getAttributeValue('targetResol', 0)
        partSet = self.inputParticles.get()
        size = partSet.getXDim()
        nyquist = 2 * partSet.getSamplingRate()
        if tgResol > nyquist:
            newSize = long(round(size * nyquist / tgResol))
            if newSize % 2 == 1:
                newSize += 1
            return newSize
        else:
            return size

    def _getParams(self, args):
        return ' '.join(['%s %s' % (k, str(v)) for k, v in args.iteritems()])

    def _setNormalArgs(self, args):
        maskDiameter = self.maskDiameterA.get()
        pixelSize = self._getPixeSize()

        if maskDiameter <= 0:
            maskDiameter = pixelSize * self._getNewDim()

        self._defineInput(args)
        args.update({'--particle_diameter': maskDiameter,
                     '--angpix': pixelSize,
                     })
        self._setCTFArgs(args)

        if self.maskZero == MASK_FILL_ZERO:
            args['--zero_mask'] = ''

        args['--K'] = self.numOfVols.get()

        if self.limitResolEStep > 0:
            args['--strict_highres_exp'] = self.limitResolEStep.get()

        if self.IS_3D:
            if not self.isMapAbsoluteGreyScale:
                args['--firstiter_cc'] = ''
            args['--ini_high'] = self.initialLowPassFilterA.get()
            args['--sym'] = self.symmetryGroup.get()
            args['--pad'] = 1 if self.skipPadding else 2

        refArg = self._getRefArg()

        if refArg:
            args['--ref'] = refArg

        self._setBasicArgs(args)

    def _setComputeArgs(self, args):
        args['--pool'] = self.pooledParticles.get()

        if not self.combineItersDisc:
            args['--dont_combine_weights_via_disc'] = ''

        if not self.useParallelDisk:
            args['--no_parallel_disc_io'] = ''

        if self.allParticlesRam:
            args['--preread_images'] = ''
        else:
            if self._getScratchDir():
                args['--scratch_dir'] = self._getScratchDir()

        if self.doGpu:
            args['--gpu'] = self.gpusToUse.get()
        args['--j'] = self.numberOfThreads.get()

    def _getProgram(self, program='relion_refine'):
        """ Get the program name depending on the MPI use or not. """
        if self.numberOfMpi > 1:
            program += '_mpi'
        return program

    def _setCTFArgs(self, args):
        # CTF stuff
        if self.doCTF.get():
            args['--ctf'] = ''

        if self.hasReferenceCTFCorrected.get():
            args['--ctf_corrected_ref'] = ''

        if self.inputParticles.get().isPhaseFlipped():
            args['--ctf_phase_flipped'] = ''

        if self.ignoreCTFUntilFirstPeak.get:
            args['--ctf_intact_first_peak'] = ''

    def _setBasicArgs(self, args):
        """ Return a dictionary with basic arguments. """
        self._defineOutput(args)
        args.update({'--flatten_solvent': '',
                     '--norm': '',
                     '--scale': '',
                     '--oversampling': self.oversampling.get(),
                     '--tau2_fudge': self.regularisationParamT.get()
                     })
        args['--iter'] = self._getnumberOfIters()

    def _getnumberOfIters(self):
        return self.numberOfIterations.get()

    def _getScratchDir(self):
        """ Returns the scratch dir value without spaces.
         If none, the empty string will be returned.
        """
        scratchDir = self.scratchDir.get() or ''
        return scratchDir.strip()

    def _getRefArg(self):
        return self._getRefStar()

    def _getRefStar(self):
        return self._getExtraPath("input_references.star")

    def _lastIter(self):
        return self._getIterNumber(-1)

    def _getIterNumber(self, index):
        """ Return the list of iteration files, give the iterTemplate. """
        result = None
        files = sorted(glob(self._iterTemplate))
        print("files to know Iters: ", files)
        if files:
            f = files[index]
            s = self._iterRegex.search(f)
            if s:
                result = long(s.group(1))  # group 1 is 3 digits iteration
                # number
        return result

    def _firstIter(self):
        return self._getIterNumber(0) or 1


    # def _convertInput(self, imgSet):
    #     newDim = self._getNewDim()
    #     bg = newDim / 2
    #
    #     args = '--operate_on %s --operate_out %s --norm --bg_radius %d'
    #
    #     params = args % (self._getFileName('input_star'),
    #                      self._getFileName('preprocess_parts_star'), bg)
    #     self.runJob(self._getProgram(program='relion_preprocess'), params)
    #
    #     from pyworkflow.utils import moveFile
    #
    #     moveFile(self._getFileName('preprocess_parts'),
    #              self._getTmpPath('particles_subset.mrcs'))
    #
    # def _invertScaleVol(self, fn):
    #     xdim = self._getInputParticles().getXDim()
    #     outputFn = self._getOutputVolFn(fn)
    #     ih = em.ImageHandler()
    #     img = ih.read(fn)
    #     img.scale(xdim, xdim, xdim)
    #     img.write(outputFn)
    #
    # def _getOutputVolFn(self, fn):
    #     return self._getExtraPath(replaceBaseExt(fn, '_origSize.mrc'))