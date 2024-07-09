from pyworkflow.protocol.params import (PointerParam, FloatParam,
                                        StringParam, BooleanParam,
                                        EnumParam, IntParam, LEVEL_ADVANCED)
import pyworkflow.utils as pwutils
from pwem.objects import Volume
from pwem.objects.data import SetOfParticles
from pwem.protocols import ProtAnalysis3D
from pwem import ALIGN_NONE, ALIGN_PROJ
from cryomethods.convert import writeSetOfParticles
from cryomethods.functions import NumpyImgHandler, occupancy
import numpy as np, mrcfile
import os
from pwem.constants import NO_INDEX

class ProtLocProb(ProtAnalysis3D):
    """
    Given a map and the number of moments, the protocol estimates the local probability map.
    """
    _label = 'local probability map'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles",
                      help='Select the input images from the project.')
        form.addParam('symmetryGroup', StringParam, default='c1',
                      label="Symmetry group",
                      help='See [[https://relion.readthedocs.io/'
                           'en/latest/Reference/Conventions.html#symmetry]'
                           '[Relion Symmetry]] page for a description '
                           'of the symmetry format accepted by Relion')
        form.addParam('maxRes', FloatParam, default=-1,
                      label="Maximum resolution (A)",
                      help='Maximum resolution (in Angstrom) to consider \n'
                           'in Fourier space (default Nyquist).')
        form.addParam('numMom', IntParam, default=5,
                      label="Number of moments",
                      help='Number of moments to estimate. Maximum 4')

        form.addParallelSection(threads=1, mpi=1)


    # --------------------------- INSERT steps functions ----------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        # Añadir argumentos tanto a la de reconstruct como a la de process. a la de process
        # tiene que ser para que se haga para cada momento y a la de reconstruct es cambiar el nombre
        # del volumen final para que el de orden 3 no sobreescriba al de orden 2 y asi sucesivam

        num_mom = self.numMom.get()

        for m in range(1, num_mom+1):
            self._insertFunctionStep('_processParticles',m)
            self._insertFunctionStep('reconstructStep',m)

        self._insertFunctionStep('createOutputStep')


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

    def createOutputStep(self):

        pass

    def reconstructStep(self, m_index=1):

        print("ENTRO")
        volume_name = 'output_volume' + '_' + str(m_index) + '.mrc'
        imgSet = self.inputParticles.get()

        params = ' --i %s' % self._getExtraPath('inputParticles.star') #input_particles
        params += ' --o %s' % self._getExtraPath(volume_name) #output_volume
        params += ' --sym %s' % self.symmetryGroup.get()
        params += ' --pad 2 --subset -1 --class -1 '

        # Addition of the Sampling rate and the maximum resolution
        params += ' --angpix %0.5f' % imgSet.getSamplingRate()
        params += ' --maxres %0.3f' % self.maxRes.get()

        # 3D reconstruction with all the defined parameters
        self.runJob('relion_reconstruct', params)

        #self._processParticles()


    def _processParticles(self, m_index=1):
        print(m_index)

        imgSet = self.inputParticles.get()
        imgStar = self._getExtraPath('inputParticles.star')


        # Nos situamos en la ruta de la nueva carpeta creada, my_folder
        mypath = os.path.join(self._getPath(), "my_folder")
        os.makedirs(mypath, exist_ok=True)

        print(mypath)
        print(os.getcwd())

        #print(imgSetNew)
        #print("JV")

        # Hacemos una copia del set de partículas inicial, que apuntara a la nueva localizacion
        imgSetNew = self._createSetOfParticles()
        imgSetNew.copyInfo(imgSet)

        for i, img in enumerate(imgSet):
            loc = img.getLocation()

            #print('LOCALIZACION INICIAL:', loc)
            nombre_part = os.path.basename(loc[1])

            part = NumpyImgHandler.loadMrcSlice(str(loc[0])+'@'+loc[1], writable=True)
            loc_new_folder = (loc[0], os.path.join(mypath, str(loc[0]) + '_' + nombre_part))

            #img.setLocation(loc_new_folder)
            #print('NUEVA LOCALIZACION', img.getLocation())

            if m_index == 1:
                NumpyImgHandler.saveMrc(part, loc_new_folder[1])
            else:
                #print('NUMERO DE MOMENTOS', n)
                part_n = np.power(part, m_index)
                #print(part_n)
                NumpyImgHandler.saveMrc(part_n, loc_new_folder[1])

        print(f'------------------GUARDAR MOMENTO ORDEN {m_index}------------------')
        imgSetNew.copyItems(imgSet, updateItemCallback=self._setFileName)
        writeSetOfParticles(imgSetNew, imgStar,
                            outputDir=self._getExtraPath(),
                            alignType=ALIGN_PROJ)


    # --------------------------- INFO functions ------------------------------

    def _getOutStack(self, index, fn):
        """ Return the output stack filename based on the input. """
        mypath = os.path.join(self._getPath(), "my_folder")
        nombre_part = os.path.basename(fn)
        loc_new_folder = (index, os.path.join(mypath, str(index) + '_' + nombre_part))
        print("------------")
        #print("------------")
        print(loc_new_folder)
        return loc_new_folder[1]

    def _setFileName(self, item, row=None):
        index, fn = item.getLocation()
        #ctf = item.getCTF()

        #dU = ctf.getDefocusU()
        #dV = ctf.getDefocusV()

        #ctf.setDefocusU(dU*2)
        #ctf.setDefocusV(dV*2)

        #item.setCTF(ctf)

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
