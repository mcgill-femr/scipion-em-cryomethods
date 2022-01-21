from pyworkflow.protocol.params import PointerParam, FloatParam, LEVEL_ADVANCED
from pyworkflow.utils import Message
from pwem.objects import Volume
from pwem.protocols import ProtAnalysis3D

from cryomethods.functions import NumpyImgHandler, occupancy


class ProtLocOccupancy(ProtAnalysis3D):
    """
    Given a map and a mask, the protocol estimates the local occupancy.
    """
    _label = 'local occupancy map'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        add = form.addParam  # shortcut

        add('vol', PointerParam, pointerClass='Volume',
            label='Map', important=True,
            help='Map for determining its local occupancy.')

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

        add('protein_threshold', FloatParam, expertLevel=LEVEL_ADVANCED,
            label='Protein threshold',
            default=0.25,
            help='Percentile to select a typical protein signal.')

        add('f_voxel_width', FloatParam, expertLevel=LEVEL_ADVANCED,
            label='Frequency selection width in bins',
            default=4.8,
            help='Number of frequency bins used for the width of the bandpass '
                 'filter applied in Fourier Space.')

        # NOTE: if we had a parallel running version, we could add:
        #   form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions ----------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('computeOccupancyStep')
        self._insertFunctionStep('createOutputStep')

    def computeOccupancyStep(self):
        volume_path = self.vol.get().getFileName()
        vol = NumpyImgHandler.loadMrc(volume_path)

        mask_path = self.mask_in_molecule.get().getFileName()
        mask = NumpyImgHandler.loadMrc(mask_path)

        voxel_size = self.vol.get().getSamplingRate()
        min_res = self.min_res.get()
        max_res = self.max_res.get()
        num_points = 10  # NOTE: hardcoded
        protein_threshold = self.protein_threshold.get()
        f_voxel_width = self.f_voxel_width.get()

        omap = occupancy(vol, mask, voxel_size, min_res, max_res,
                         num_points, protein_threshold, f_voxel_width)

        omap_path = self._getExtraPath('omap.mrc')

        NumpyImgHandler.saveMrc(omap, omap_path)

    def createOutputStep(self):
        omap = Volume()
        omap.setFileName(self._getExtraPath("omap.mrc"))
        omap.setSamplingRate(self.vol.get().getSamplingRate())
        self._defineOutputs(omap=omap)
        self._defineSourceRelation(self.vol, omap)

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
