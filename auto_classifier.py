# **************************************************************************
# *
# * Authors:     Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
# *              Javier Vargas Balbuena (javier.vargasbalbuena@mcgill.ca)
# *
# * Department of Anatomy and Cell Biology, McGill University
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
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
from collections import Counter

import pyworkflow.em as em
import pyworkflow.protocol.constants as cons
from protocol_base import ProtocolRelionBase


class ProtAutoClassifier(ProtocolRelionBase):
    _label = 'auto classifier'
    IS_VOLSELECTOR = False

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        ProtocolRelionBase._defineParams(form)
        self._defineCTFParams(form, expertLev=em.LEVEL_NORMAL)

    # -------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self.finished = False
        self._evalIdsList = []
        self._doneList = []

        self._initialize()
        self._insertFunctionStep('convertInputStep', self._getResetDeps())
        fDeps = self._insertLevels()
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=fDeps, wait=True)

    def _insertEvaluationStep(self, clsStep):
        evalDep = self._insertFunctionStep('evaluationStep',
                                           prerequisites=[clsStep])
        return [evalDep]

    def _insertLevelSteps(self):
        deps = []
        clsNumber = self.numberOfClasses.get()
        levelRuns = clsNumber^self._level
        for i in range(1, levelRuns+1):
            clsStep = self._insertClassifyStep()
            self._setNewEvalIds(levelRuns)
            evStep = self._insertEvaluationStep(clsStep)
            deps.append(evStep)
        return deps

    def _stepsCheck(self):
        if self._level == 3: # condition to stop the cycle
            self.finished = True
        outputStep = self._getFirstJoinStep()
        if self.finished:  # Unlock createOutputStep if finished all jobs
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)
        else:
            if Counter(self._evalIdsList) == Counter(self._doneList):
                self._level += 1
                fDeps = self._insertLevelSteps()

                if outputStep is not None:
                    outputStep.addPrerequisites(*fDeps)
                self.updateSteps()

    # -------------------------- STEPS functions -------------------------------
    def convertInputStep(self, resetDeps):
        pass
        # """ Create the input file in STAR format as expected by Relion.
        # If the input particles comes from Relion, just link the file.
        # Params:
        #     particlesId, volumesId: use this parameters just to force redo of
        #     convert if either the input particles and/or input volumes are
        #     changed.
        # """
        # self._imgFnList = []
        # imgSet = self._getInputParticles()
        # imgStar = self._getFileName('input_star')
        #
        # subset = em.SetOfParticles(filename=":memory:")
        #
        # newIndex = 1
        # for img in imgSet.iterItems(orderBy='RANDOM()', direction='ASC'):
        #     self._scaleImages(newIndex, img)
        #     newIndex += 1
        #     subset.append(img)
        #     subsetSize = self.subsetSize.get()
        #     minSize = min(subsetSize, imgSet.getSize())
        #     if subsetSize   > 0 and subset.getSize() == minSize:
        #         break
        # conv.writeSetOfParticles(subset, imgStar, self._getExtraPath(),
        #                     alignType=em.ALIGN_NONE,
        #                     postprocessImageRow=self._postprocessParticleRow)
        # self._convertInput(subset)
        # self._convertRef()

    def evaluationStep(self):
        pass



    # -------------------------- UTILS functions -------------------------------
    def _setSamplingArgs(self, args):
        """ Set sampling related params. """
        if self.doImageAlignment:
            args['--healpix_order'] = self.angularSamplingDeg.get()
            args['--offset_range'] = self.offsetSearchRangePix.get()
            args['--offset_step']  = (self.offsetSearchStepPix.get() *
                                      self._getSamplingFactor())
            if self.localAngularSearch:
                args['--sigma_ang'] = self.localAngularSearchRange.get() / 3.
        else:
            args['--skip_align'] = ''

    def _getResetDeps(self):
        return "%s, %s" % (self._getInputParticles().getObjId(),
                           self.inputVolumes.get().getObjId())

    def _setNewEvalIds(self, levelRuns):
        self._evalIdsList.append("%s.%s" %(self._level, levelRuns))

    def _getFirstJoinStepName(self):
        # This function will be used for streaming, to check which is
        # the first function that need to wait for all micrographs
        # to have completed, this can be overwritten in subclasses
        # (eg in Xmipp 'sortPSDStep')
        return 'createOutputStep'

    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == self._getFirstJoinStepName():
                return s
        return None
