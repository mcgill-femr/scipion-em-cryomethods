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
import re
from collections import Counter

import pyworkflow.em as em
import pyworkflow.em.metadata as md
import pyworkflow.protocol.constants as cons
import pyworkflow.protocol.params as params
from protocol_base import ProtocolBase
from convert import writeSetOfParticles
from pyworkflow.utils import makePath


class ProtAutoClassifier(ProtocolBase):
    _label = 'auto classifier'
    IS_VOLSELECTOR = False

    def __init__(self, **args):
        ProtocolBase.__init__(self, **args)
        self._level = 1
        self._rLev = 1

    def _createFilenameTemplates(self):
        """ Centralize how files are called for iterations and references. """
        self.levDir = self._getExtraPath('lev_%(lev)02d/')
        self.rLevDir = self._getExtraPath('lev_%(lev)02d/rLev_%(rLev)02d/')
        self.rLevIter = self.rLevDir + 'relion_it%(iter)03d_'
        # add to keys, data.star, optimiser.star and sampling.star
        myDict = {
                  'input_star': self.levDir + 'id%(rLevId)s_particles.star',
                  'data': self.rLevIter + 'data.star',
                  # 'model': self.extraIter + 'model.star',
                  # 'optimiser': self.extraIter + 'optimiser.star',
                  # 'selected_volumes': self._getTmpPath('selected_volumes_xmipp.xmd'),
                  # 'movie_particles': self._getPath('movie_particles.star'),
                  # 'dataFinal': self._getExtraPath("relion_data.star"),
                  # 'modelFinal': self._getExtraPath("relion_model.star"),
                  # 'finalvolume': self._getExtraPath("relion_class%(ref3d)03d.mrc:mrc"),
                  # 'preprocess_parts': self._getPath("preprocess_particles.mrcs"),
                  # 'preprocess_parts_star': self._getPath("preprocess_particles.star"),
                  #
                  # 'data_scipion': self.extraIter + 'data_scipion.sqlite',
                  # 'volumes_scipion': self.extraIter + 'volumes.sqlite',
                  #
                  # 'angularDist_xmipp': self.extraIter + 'angularDist_xmipp.xmd',
                  'all_avgPmax_xmipp': self._getTmpPath('iterations_avgPmax_xmipp.xmd'),
                  'all_changes_xmipp': self._getTmpPath('iterations_changes_xmipp.xmd'),
                  }
        for key in self.FILE_KEYS:
            myDict[key] = self.rLevIter + '%s.star' % key
            key_xmipp = key + '_xmipp'
            myDict[key_xmipp] = self.rLevDir + '%s.xmd' % key
        # add other keys that depends on prefixes
        for p in self.PREFIXES:
            myDict['%smodel' % p] = self.rLevDir + '%smodel.star' % p
            myDict['%svolume' % p] = self.rLevDir + p + 'class%(ref3d)03d.mrc:mrc'

        self._updateFilenamesDict(myDict)

    def _createIterTemplates(self):
        """ Setup the regex on how to find iterations. """
        self._iterTemplate = self._getFileName('data', lev=self._level,
                                               rLev=self._rLev,
                                               iter=0).replace( '000', '???')
        # Iterations will be identify by _itXXX_ where XXX is the iteration
        # number and is restricted to only 3 digits.
        self._classRegex = re.compile('_class(\d{2,2})_')

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        self._defineInputParams(form)
        self._defineCTFParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineOptimizationParams(form, expertLev=cons.LEVEL_NORMAL)
        form.addParam('doImageAlignment', params.BooleanParam, default=True,
                      label='Perform image alignment?',
                      help='If set to No, then rather than performing both '
                           'alignment and classification, only classification '
                           'will be performed. This allows the use of very '
                           'focused masks.This requires that the optimal '
                           'orientations of all particles are already '
                           'calculated.')
        self._defineSamplingParams(form, expertLev=cons.LEVEL_NORMAL,
                                   cond='doImageAlignment')
        self._defineAdditionalParams(form)

    # -------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._evalIdsList = []
        self._doneList = []

        self._initialize()
        fDeps = self._insertLevelSteps()
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=fDeps, wait=True)

    def _insertLevelSteps(self):
        deps = []
        clsNumber = self.numberOfClasses.get()
        levelRuns = clsNumber**(self._level - 1)

        for rLev in range(1, levelRuns+1):
            self._rLev = rLev # Just to generate the proper input star file.
            self._insertFunctionStep('convertInputStep', self._getResetDeps(),
                                     self.copyAlignment, rLev,
                                     prerequisites=deps)
            self._insertClassifyStep()
            self._setNewEvalIds(rLev)
            evStep = self._insertEvaluationStep(rLev)
            deps.append(evStep)
        return deps

    def _insertEvaluationStep(self, rLev):
        evalDep = self._insertFunctionStep('evaluationStep', rLev)
        return [evalDep]

    def _stepsCheck(self):
        print('Just passing through this')
        self.finished = False
        if self._level == 3: # condition to stop the cycle
            self.finished = True
        outputStep = self._getFirstJoinStep()
        if self.finished:  # Unlock createOutputStep if finished all jobs
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)
        else:
            print('Not finished, cheking if previous step finished',
                  self._evalIdsList, self._doneList)
            if Counter(self._evalIdsList) == Counter(self._doneList):
                print('_evalIdsList == _doneList')
                self._level += 1
                fDeps = self._insertLevelSteps()

                if outputStep is not None:
                    outputStep.addPrerequisites(*fDeps)
                self.updateSteps()

    # -------------------------- STEPS functions -------------------------------
    def convertInputStep(self, resetDeps, copyAlignment, rLev):
        """ Create the input file in STAR format as expected by Relion.
        If the input particles comes from Relion, just link the file.
        """
        makePath(self._getRunPath(self._level, rLev))
        imgStar = self._getFileName('input_star')

        if self._level == 1:
            imgSet = self._getInputParticles()
            self.info("Converting set from '%s' into '%s'" %
                      (imgSet.getFileName(), imgStar))

            # Pass stack file as None to avoid write the images files
            # If copyAlignment is set to False pass alignType to ALIGN_NONE
            alignType = imgSet.getAlignment() if copyAlignment \
                else em.ALIGN_NONE

            hasAlign = alignType != em.ALIGN_NONE
            alignToPrior = hasAlign and self._getBoolAttr('alignmentAsPriors')
            fillRandomSubset = hasAlign and self._getBoolAttr('fillRandomSubset')

            writeSetOfParticles(imgSet, imgStar, self._getExtraPath(),
                                alignType=alignType,
                                postprocessImageRow=self._postprocessParticleRow,
                                fillRandomSubset=fillRandomSubset)
            if alignToPrior:
                self._copyAlignAsPriors(imgStar, alignType)

            if self.doCtfManualGroups:
                self._splitInCTFGroups(imgStar)

            if self._getRefArg():
                self._convertRef()
        else:
            pass

    def evaluationStep(self, rLev):
        rLevId = self._getRunLevId(self._level, rLev)
        imgStar = self._getFileName('data', iter=self._lastIter(),
                                    lev=self._level, rLev=rLev)

        mdStar = md.MetaData(imgStar)

        for row in md.iterRows(mdStar, sortByLabel=md.RLN_PARTICLE_CLASS):
            pass

        print('Executing evaluation step')
        self._doneList.append(rLevId)

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
        self._evalIdsList.append(self._getRunLevId(self._level, levelRuns))

    def _getRunLevId(self, level, levelRuns):
        return "%s.%s" %(level, levelRuns)

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

    def _getLevelPath(self, level, *paths):
        return self._getExtraPath('lev_%02d' % level, *paths)

    def _getRunPath(self, level, runLevel, *paths):
        return self._getLevelPath(level, 'rLev_%02d' % runLevel, *paths)

    def _defineInputOutput(self, args):
        args['--i'] = self._getFileName('input_star',lev=self._level,
                                        rLev=self._rLev)
        args['--o'] = self._getRunPath(self._level, self._rLev, 'relion')

    def _getBoolAttr(self, attr=''):
        return getattr(attr, False)