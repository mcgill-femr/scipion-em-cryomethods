import pyworkflow.protocol.params as params
from cryomethods import Plugin
from .protocol_base import ProtocolBase
from pwem.protocols import ProtCTFMicrographs
from pwem.objects import CTFModel, Float
import matplotlib.pyplot as plt
import numpy as np

class Protdctf_compare(ProtCTFMicrographs):
    """
    Compare CTF per micrograph results .
    """
    _label = 'dctf_compare'

    def __init__(self, **args):
        ProtCTFMicrographs.__init__(self, **args)
    # --------------- DEFINE param functions ---------------

    def _defineParams(self, form):
        form.addSection('Params')
        group = form.addGroup('General')

        group.addParam('refCTFSet', params.PointerParam, allowsNull=False,
                       pointerClass='SetOfCTF',
                       label="Reference CTF set",
                       help='Select the reference CTF sets.')

        group.addParam('testCTFSet', params.PointerParam, allowsNull=False,
                       pointerClass='SetOfCTF',
                       label="Test CTF set",
                       help='Select the CTF set to compare.')

    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('runCTFCompareStep')
        self._insertFunctionStep('createOutputStep')

    def convertInputStep(self):
        self.rSet = self.refCTFSet.get()
        self.tSet = self.testCTFSet.get()

    def recalculate(self):
        pass

    def runCTFCompareStep(self):
        ctfDict1 = {ctf.getObjId(): ctf.clone() for ctf
                    in self.rSet.iterItems()}

        ctfDict2 = {ctf.getObjId(): ctf.clone() for ctf
                    in self.tSet.iterItems()}

        newIDs = list(set(ctfDict1).intersection(set(ctfDict2)))
        self.ctfResults = self._createSetOfCTF()

        numElems = len(newIDs)
        self.error_defocusU = np.zeros(numElems)
        self.error_defocusV = np.zeros(numElems)
        self.error_defocusAngle = np.zeros(numElems)
        self.error_resolution = np.zeros(numElems)

        self.relative_error_defocusU = np.zeros(numElems)
        self.relative_error_defocusV = np.zeros(numElems)
        self.relative_defocusAngle = np.zeros(numElems)
        self.relative_error_resolution = np.zeros(numElems)

        for index, id in enumerate(newIDs):
            print(id,index)
            print("ctfDict1[id]:",ctfDict1[id])
            print("ctfDict2[id]:",ctfDict2[id])

            ctf = CTFModel()
            ctf.setStandardDefocus(ctfDict1[id].getDefocusU(),
                                   ctfDict1[id].getDefocusV(),
                                   ctfDict1[id].getDefocusAngle())
            ctf.setResolution(ctfDict1[id].getResolution())

            ctf._error_defocusU = Float(ctfDict1[id].getDefocusU()-ctfDict2[id].getDefocusU())
            ctf._error_defocusV = Float(ctfDict1[id].getDefocusV()-ctfDict2[id].getDefocusV())
            ctf._error_defocusAngle = Float(ctfDict1[id].getDefocusAngle()-ctfDict2[id].getDefocusAngle())
            ctf._error_resolution = Float(ctfDict1[id].getResolution()-ctfDict2[id].getResolution())

            print("CTF_2",ctf)

            #ctf._relative_error_defocusU = Float(float(ctf._error_defocusU)/float(ctfDict1[id].getDefocusU()))
            #ctf._relative_error_defocusV = Float(float(ctf._error_defocusV)/float(ctfDict1[id].getDefocusV()))

            #if ctfDict1[id].getDefocusAngle() != 0:
            #    ctf._relative_error_defocusAngle = Float(float(ctf._error_defocusAngle)/float(ctfDict1[id].getDefocusAngle()))
            #else:
            #    ctf._relative_error_defocusAngle = 0
            #ctf._relative_error_resolution = Float(float(ctf._error_resolution)/float(ctfDict1[id].getResolution()))

            print("CTF_3",ctf)

            self.ctfResults.append(ctf)

            self.error_defocusU[index] = ctf._error_defocusU
            self.error_defocusV[index] = ctf._error_defocusV
            self.error_defocusAngle[index] = ctf._error_defocusAngle
            self.error_resolution[index] = ctf._error_resolution

            #self.relative_error_defocusU[index] = ctf._relative_error_defocusU
            #self.relative_error_defocusV[index] = ctf._relative_error_defocusV
            #self.relative_defocusAngle[index] = ctf._relative_error_defocusAngle
            #self.relative_error_resolution[index] = ctf._error_resolution
            print(".-------.")

    def createOutputStep(self):

        media_error_defocusU = np.mean(self.error_defocusU)
        std_error_defocusU = np.std(self.error_defocusU)
        rmse_error_defocusU = np.sqrt(np.mean((self.error_defocusU - media_error_defocusU) ** 2))
        mae_error_defocusU = np.mean(np.abs(self.error_defocusU - media_error_defocusU))

        print(" ")
        print("------------------")
        print("media_error_defocusU:", media_error_defocusU)
        print("std_error_defocusU:", std_error_defocusU)
        print("rmse_error_defocusU:", rmse_error_defocusU)
        print("mae_error_defocusU:", mae_error_defocusU)
        print(" ")
        print("------------------")

        media_error_defocusV = np.mean(self.error_defocusV)
        std_error_defocusV = np.std(self.error_defocusV)
        rmse_error_defocusV = np.sqrt(np.mean((self.error_defocusV - media_error_defocusV) ** 2))
        mae_error_defocusV = np.mean(np.abs(self.error_defocusV - media_error_defocusV))

        print(" ")
        print("------------------")
        print("media_error_defocusV:", media_error_defocusV)
        print("std_error_defocusV:", std_error_defocusV)
        print("rmse_error_defocusV:", rmse_error_defocusV)
        print("mae_error_defocusV:", mae_error_defocusV)
        print(" ")
        print("------------------")

        media_error_defocusAngle = np.mean(self.error_defocusAngle)
        std_error_defocusAngle = np.std(self.error_defocusAngle)
        rmse_error_defocusAngle = np.sqrt(np.mean((self.error_defocusAngle - media_error_defocusAngle) ** 2))
        mae_error_defocusAngle = np.mean(np.abs(self.error_defocusAngle - media_error_defocusAngle))

        print(" ")
        print("------------------")
        print("media_error_defocusAngle:", media_error_defocusAngle)
        print("std_error_defocusAngle:", std_error_defocusAngle)
        print("rmse_error_defocusAngle:", rmse_error_defocusAngle)
        print("mae_error_defocusAngle:", mae_error_defocusAngle)
        print(" ")
        print("------------------")

        media_error_resolution = np.mean(self.error_resolution)
        std_error_resolution = np.std(self.error_resolution)
        rmse_error_resolution = np.sqrt(np.mean((self.error_resolution - media_error_resolution) ** 2))
        mae_error_resoltuion = np.mean(np.abs(self.error_resolution - media_error_resolution))

        print(" ")
        print("------------------")
        print("media_error_resolution:", media_error_resolution)
        print("std_error_resolution:", std_error_resolution)
        print("rmse_error_resolution:", rmse_error_resolution)
        print("mae_error_resoltuion:", mae_error_resoltuion)
        print(" ")
        print("------------------")

        #self.relative_error_defocusU
        #self.relative_error_defocusV
        #self.relative_defocusAngle
        #self.relative_error_resolution

        self._defineOutputs(ctfResults=self.ctfResults)

    def _validate(self):
        return []

    def _citations(self):
        return []

    def _summary(self):
        return []

    def _methods(self):
        return []







