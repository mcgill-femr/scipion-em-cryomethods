from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)
from pyworkflow.utils import Message
from pwem.objects import Volume
from pwem.protocols import ProtAnalysis3D

from cryomethods.functions import NumpyImgHandler, bfactor


class ProtLocBFactor(ProtAnalysis3D):
    """
    Given a map and a mask the protocol estimates the local B factor.
    """
    _label = 'local b-factor map'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        add = form.addParam  # shortcut

        add('vol', PointerParam, pointerClass='Volume',
            label='Map', important=True,
            help='Map for determining its local b-factors.')

        add('mask_in_molecule', PointerParam, pointerClass='VolumeMask',
            label='Mask selecting the molecule',
            help='Tight binary map that differenciates between the '
                 'macromolecule (1) and the background (0).')

        add('min_res', FloatParam, expertLevel=LEVEL_ADVANCED,
            label='Minimum resolution',
            default=15,
            help='Minimun resolution of the sweeping resolution range in '
                 'Angstroms. A value of 15-10 Angstroms is normally good.')

        add('max_res', FloatParam,
            label='Maximum resolution',
            help='Maximum resolution (in Angstroms) of the resolution range. '
                 'Provide here the obtained FSC global resolution value.')

        add('noise_threshold', FloatParam, expertLevel=LEVEL_ADVANCED,
            label='Noise threshold',
            default=0.9,
            help='Percentile of noise used to discriminate signal from noise. '
                 'Good values are 0.9 or 0.95.')

        add('f_voxel_width', FloatParam, expertLevel=LEVEL_ADVANCED,
            label='Frequency selection width in bins',
            default=4.8,
            help='Number of frequency bins used for the width of the bandpass '
                 'filter applied in Fourier Space.')

        add('only_above_noise', BooleanParam,
            label='Use only points above noise?',
            default=False,
            help='True: does not take into consideration points in each local '
                 'Guinier plot that are below the noise level to calculate '
                 'B-factors. In this case, voxels not processed will be '
                 'labeled as NaN. If False is selected, all points of each '
                 'local Guinier plot are taken into consideration (default).')

        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions ----------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('computeBFactorStep')
        self._insertFunctionStep('createOutputStep')

    def computeBFactorStep(self):
        volume_path = self.vol.get().getFileName()
        vol = NumpyImgHandler.loadMrc(volume_path)

        mask_path = self.mask_in_molecule.get().getFileName()
        mask = NumpyImgHandler.loadMrc(mask_path)

        voxel_size = self.vol.get().getSamplingRate()
        min_res = self.min_res.get()
        max_res = self.max_res.get()
        num_points = 10  # NOTE: hardcoded
        noise_threshold = self.noise_threshold.get()
        f_voxel_width = self.f_voxel_width.get()
        only_above_noise = self.only_above_noise.get()

        bmap = bfactor(vol, mask, voxel_size, min_res, max_res,
                       num_points, noise_threshold, f_voxel_width,
                       only_above_noise)

        bmap_path = self._getExtraPath('bmap.mrc')

        NumpyImgHandler.saveMrc(bmap, bmap_path)

    def createOutputStep(self):
        bmap = Volume()
        bmap.setFileName(self._getExtraPath("bmap.mrc"))
        bmap.setSamplingRate(self.vol.get().getSamplingRate())
        self._defineOutputs(bmap=bmap)
        self._defineSourceRelation(self.vol, bmap)

    # --------------------------- INFO functions ------------------------------

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
        summary.append(" ")
        return summary

    def _citations(self):
        return ['Vargas2021']
