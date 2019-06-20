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
import os
import re
import numpy as np
from scipy import stats
from glob import glob
from collections import Counter, defaultdict

import pyworkflow.em as em
import pyworkflow.em.metadata as md
import pyworkflow.protocol.constants as cons
from pyworkflow.utils import (makePath, copyFile, replaceBaseExt)

from cryomethods import Plugin
from cryomethods.convert import (writeSetOfParticles, rowToAlignment,
                                 relionToLocation, loadMrc, saveMrc,
                                 alignVolumes, applyTransforms)

from .protocol_base import ProtocolBase


class Prot3DAutoClassifier(ProtocolBase):
    _label = '3D auto classifier'
    IS_2D = False
    IS_AUTOCLASSIFY = True


    def __init__(self, **args):
        ProtocolBase.__init__(self, **args)


    def _initialize(self):
        """ This function is mean to be called after the
        working dir for the protocol have been set.
        (maybe after recovery from mapper)
        """
        self._createFilenameTemplates()
        self._createIterTemplates()

    def _createFilenameTemplates(self):
        """ Centralize how files are called for iterations and references. """
        self.levDir = self._getExtraPath('lev_%(lev)02d/')
        self.rLevDir = self._getExtraPath('lev_%(lev)02d/rLev_%(rLev)02d/')
        self.rLevIter = self.rLevDir + 'relion_it%(iter)03d_'
        # add to keys, data.star, optimiser.star and sampling.star
        myDict = {
                'input_star': self.levDir + 'input_rLev-%(rLev)03d.star',
                'outputData': self.levDir + 'output_data.star',
                'map': self.levDir + 'map_id-%(id)s.mrc',
                'avgMap': self.levDir + 'map_average.mrc',
                'relionMap': self.rLevDir + 'relion_it%(iter)03d_class%(ref3d)03d.mrc',
                'outputModel': self.levDir + 'output_model.star',
                'model': self.rLevIter + 'model.star',
                'data': self.rLevIter + 'data.star',
                'rawFinalModel': self._getExtraPath('raw_final_model.star'),
                'rawFinalData': self._getExtraPath('raw_final_data.star'),
                'finalModel': self._getExtraPath('final_model.star'),
                'finalData': self._getExtraPath('final_data.star'),
                'finalAvgMap': self._getExtraPath('map_average.mrc'),
                'optimiser': self.rLevIter + 'optimiser.star',
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
                # 'all_avgPmax_xmipp': self._getTmpPath(
                #     'iterations_avgPmax_xmipp.xmd'),
                # 'all_changes_xmipp': self._getTmpPath(
                #     'iterations_changes_xmipp.xmd'),
                  }
        for key in self.FILE_KEYS:
            myDict[key] = self.rLevIter + '%s.star' % key
            key_xmipp = key + '_xmipp'
            myDict[key_xmipp] = self.rLevDir + '%s.xmd' % key
        # add other keys that depends on prefixes
        for p in self.PREFIXES:
            myDict['%smodel' % p] = self.rLevIter + '%smodel.star' % p
            myDict[
                '%svolume' % p] = self.rLevDir + p + 'class%(ref3d)03d.mrc:mrc'

        self._updateFilenamesDict(myDict)

    def _createIterTemplates(self, rLev=None):
        """ Setup the regex on how to find iterations. """
        rLev = self._rLev if rLev is None else rLev
        print("level rLev in _createIterTemplates:", self._level, rLev)

        self._iterTemplate = self._getFileName('data', lev=self._level,
                                               rLev=rLev,
                                               iter=0).replace('000', '???')
        # Iterations will be identify by _itXXX_ where XXX is the iteration
        # number and is restricted to only 3 digits.
        self._iterRegex = re.compile('_it(\d{3,3})_')
        self._classRegex = re.compile('_class(\d{2,2}).')

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        self._defineInputParams(form)
        self._defineReferenceParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineCTFParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineOptimizationParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineSamplingParams(form, expertLev=cons.LEVEL_NORMAL,
                                   cond='doImageAlignment')
        self._defineAdditionalParams(form)

    # -------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._level = 0
        self._rLev = 1
        self._evalIdsList = []
        self._doneList = []
        self._constStop = []
        self.stopDict = {}
        self.stopResLog = {}
        self._mapsDict = {}
        self._clsIdDict = {}

        self._initialize()

        deps = self._insertLevelSteps()
        self._insertFunctionStep('mergeClassesStep', wait=True,
                                 prerequisites=deps)
        self._insertFunctionStep('createOutputStep')

    def _insertLevelSteps(self):
        deps = []
        levelRuns = self._getLevRuns(self._level)

        self._insertFunctionStep('convertInputStep',
                                 self._getResetDeps(),
                                 self.copyAlignment,
                                 prerequisites=deps)
        if self._level == 0:
            deps = self._instertLev0Step()
        else:
            for rLev in levelRuns:
                clsDepList = []
                self._rLev = rLev  # Just to generate the proper input star file.
                classDep = self._insertClassifyStep()
                clsDepList.append(classDep)
                self._setNewEvalIds()

            evStep = self._insertEvaluationStep(clsDepList)
            deps.append(evStep)
        return deps

    def _instertLev0Step(self):
        self._insertClassifyStep(K=1)
        self._insertFunctionStep('resLogStep')
        for i in range(1, 3, 1):
            self._insertFunctionStep('runContinueClasfStep', i)
            self._insertFunctionStep('resLogStep')
        dep = self._insertFunctionStep('addDoneListStep')
        self._setNewEvalIds()
        return [dep]

    def _insertClassifyStep(self, **kwargs):
        """ Prepare the command line arguments before calling Relion. """
        # Join in a single line all key, value pairs of the args dict
        normalArgs = {}
        basicArgs = {}
        self._setNormalArgs(normalArgs)
        self._setBasicArgs(basicArgs)
        if kwargs:
            for key, value in kwargs.items():
                newKey = '--%s' % key
                normalArgs[newKey] = value

        return self._insertFunctionStep('runClassifyStep', normalArgs,
                                        basicArgs, self._rLev)

    def _insertEvaluationStep(self, deps):
        evalDep = self._insertFunctionStep('evaluationStep', prerequisites=deps)
        return evalDep

    def _stepsCheck(self):
        print('_stepsCheck???')
        if Counter(self._evalIdsList) == Counter(self._doneList):
            print ('pass trough here: '
                   'Counter(self._evalIdsList) == Counter(self._doneList)')

            mergeStep = self._getFirstJoinStep()
            if self._condToStop() and self._level > 0:
                print('_condToStop ???')
                # Unlock mergeClassesStep if finished all jobs
                if mergeStep and mergeStep.isWaiting():
                    self._level += 1
                    mergeStep.setStatus(cons.STATUS_NEW)

            else:
                self._level += 1
                fDeps = self._insertLevelSteps()
                if mergeStep is not None:
                    mergeStep.addPrerequisites(*fDeps)
                self.updateSteps()

    # -------------------------- STEPS functions -------------------------------
    def convertInputStep(self, resetDeps, copyAlignment):
        """ Create the input file in STAR format as expected by Relion.
        If the input particles comes from Relion, just link the file.
        """

        if self._level == 0:
            imgStar = self._getFileName('input_star', lev=self._level, rLev=1)

            makePath(self._getRunPath(self._level, 1))
            imgSet = self._getInputParticles()
            self.info("Converting set from '%s' into '%s'" %
                      (imgSet.getFileName(), imgStar))

            # Pass stack file as None to avoid write the images files
            # If copyAlignment is set to False pass alignType to ALIGN_NONE
            alignType = imgSet.getAlignment() if copyAlignment else em.ALIGN_NONE

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

            self._convertVol(em.ImageHandler(), self.inputVolumes.get())

        elif self._level == 1:
            makePath(self._getRunPath(self._level, 1))
            imgStarLev0 = self._getFileName('input_star', lev=0, rLev=1)
            imgStar = self._getFileName('input_star', lev=self._level, rLev=1)
            copyFile(imgStarLev0, imgStar)

        else:
            lastCls = None
            prevStar = self._getFileName('outputData', lev=self._level - 1)
            mdData = md.MetaData(prevStar)

            for row in md.iterRows(mdData, sortByLabel=md.RLN_PARTICLE_CLASS):
                clsPart = row.getValue(md.RLN_PARTICLE_CLASS)
                if clsPart != lastCls:
                    makePath(self._getRunPath(self._level, clsPart))

                    if lastCls is not None:
                        print("writing %s" % fn)
                        mdInput.write(fn)
                    paths = self._getRunPath(self._level, clsPart)
                    makePath(paths)
                    print ("Path: %s and newRlev: %d" % (paths, clsPart))
                    lastCls = clsPart
                    mdInput = md.MetaData()
                    fn = self._getFileName('input_star', lev=self._level,
                                           rLev=clsPart)
                objId = mdInput.addObject()
                row.writeToMd(mdInput, objId)
            print("writing %s and ending the loop" % fn)
            mdInput.write(fn)

    def runContinueClasfStep(self, i):
        args = {}

        iters = self._lastIter(1)
        imgStar = self._getFileName('data', iter=iters, lev=self._level, rLev=1)

        mdData = self._getMetadata(imgStar)
        size = mdData.size()/2
        mdAux1 = self._getMetadata()
        mdAux2 = self._getMetadata()
        mdAux1.randomize(mdData)
        mdAux2.selectPart(mdAux1, 1, size)
        mdAux2.write(imgStar)

        self._setBasicArgs(args)
        self._setContinueArgs(args, 1)
        self._setComputeArgs(args)
        args['--iter'] = iters+1
        args['--skip_align'] = ''
        del args['--healpix_order']
        del args['--offset_range']
        del args['--offset_step']
        del args['--gpu']

        paramsCont = self._getParams(args)

        self._runClassifyStep(paramsCont)

    def resLogStep(self):
        iters = self._lastIter(1)
        modelFn = self._getFileName('model', iter=iters,lev=self._level, rLev=1)
        modelMd = self._getMetadata('model_classes@' + modelFn)
        ssnr = modelMd.getValue(md.RLN_MLMODEL_ESTIM_RESOL_REF, 1)

        imgStar = self._getFileName('data', iter=iters, lev=self._level, rLev=1)
        mdData = self._getMetadata(imgStar)
        size = mdData.size()
        self.stopResLog[size] = 1/ssnr

    def addDoneListStep(self):
        rLevId = self._getRunLevId(level=0, rLev=1)
        self._doneList.append(rLevId)

    def runClassifyStep(self, normalArgs, basicArgs, rLev):
        self._createIterTemplates(rLev)  # initialize files to know iterations
        self._setComputeArgs(normalArgs)
        params = self._getParams(normalArgs)
        self._runClassifyStep(params)

        for i in range(7, 75, 1):
            basicArgs['--iter'] = i
            self._setContinueArgs(basicArgs, rLev)
            self._setComputeArgs(basicArgs)
            paramsCont = self._getParams(basicArgs)

            stop = self._stopRunCondition(rLev, i)
            if not stop:
                self._runClassifyStep(paramsCont)
            else:
                break

    def _runClassifyStep(self, params):
        """ Execute the relion steps with the give params. """
        self.runJob(self._getProgram(), params)

    def evaluationStep(self):
        Plugin.setEnviron()
        print('Starting evaluation step')
        print('which level: ', self._level)
        self._copyLevelMaps()
        self._evalStop()
        self._mergeMetaDatas()
        self._getAverageVol()
        self._alignVolumes()
        print('Finishing evaluation step')

    def mergeClassesStep(self):
        levelRuns = []
        listMd = []

        makePath(self._getLevelPath(self._level))
        # matrix = self._estimatePCA()
        listVol = self._getFinalMaps()
        matrix, _ = self._mrcToNp(listVol)

        labels = self._clusteringData(matrix)
        print("labels: ", labels)
        prevStar = self._getFileName('rawFinalData')
        mdData = md.MetaData(prevStar)

        for row in md.iterRows(mdData, sortByLabel=md.RLN_PARTICLE_CLASS):
            clsPart = row.getValue(md.RLN_PARTICLE_CLASS)
            newClass = labels[clsPart-1] + 1
            row.setValue(md.RLN_PARTICLE_CLASS, newClass)
            listMd.append((newClass, row))

        res = defaultdict(list)
        for k, v in listMd: res[k].append(v)

        for key, listMd in res.iteritems():
            levelRuns.append(key)
            makePath(self._getRunPath(self._level, key))
            mdInput = md.MetaData()
            fn = self._getFileName('input_star', lev=self._level, rLev=key)

            for rowMd in listMd:
                objId = mdInput.addObject()
                rowMd.writeToMd(mdInput, objId)
            mdInput.write(fn)

        mapIds = self._getFinalMapIds()
        print("final mapIds:", mapIds)

        #-----metadata to save all final models-------
        finalModel = self._getFileName('finalModel')
        finalMd = self._getMetadata(finalModel)

        #-----metadata to save all final particles-----
        finalData = self._getFileName('finalData')
        finalDataMd = self._getMetadata(finalData)

        for rLev in levelRuns:
            self._rLev = rLev
            iters = 15
            args = {}

            self._setNormalArgs(args)
            self._setComputeArgs(args)

            args['--K'] = 1
            args['--iter'] = iters
            mapId = mapIds[rLev-1]
            args['--ref'] = self._getRefArg(mapId)

            params = self._getParams(args)
            self.runJob(self._getProgram(), params)

            modelFn = self._getFileName('model', iter=iters,
                                        lev=self._level, rLev=rLev)
            modelMd = self._getMetadata('model_classes@' + modelFn)

            refLabel = md.RLN_MLMODEL_REF_IMAGE
            imgRow = md.getFirstRow(modelMd)
            fn = imgRow.getValue(refLabel)

            mapId = self._getRunLevId(rLev=rLev)
            newMap = self._getMapById(mapId)
            imgRow.setValue(refLabel, newMap)

            copyFile(fn, newMap)
            self._mapsDict[fn] = mapId

            imgRow.addToMd(finalMd)

            dataFn = self._getFileName('data', iter=iters,
                                       lev=self._level, rLev=rLev)
            dataMd = self._getMetadata(dataFn)
            for row in md.iterRows(dataMd):
                row.setValue(md.RLN_PARTICLE_CLASS, rLev)
                row.addToMd(finalDataMd)

        finalDataMd.write(finalData)
        finalMd.write('model_classes@' + finalModel)

    def createOutputStep(self):
        partSet = self.inputParticles.get()

        classes3D = self._createSetOfClasses3D(partSet)
        self._fillClassesFromIter(classes3D)

        self._defineOutputs(outputClasses=classes3D)
        self._defineSourceRelation(self.inputParticles, classes3D)

        # create a SetOfVolumes and define its relations
        volumes = self._createSetOfVolumes()
        volumes.setSamplingRate(partSet.getSamplingRate())

        for class3D in classes3D:
            vol = class3D.getRepresentative()
            vol.setObjId(class3D.getObjId())
            volumes.append(vol)

        self._defineOutputs(outputVolumes=volumes)
        self._defineSourceRelation(self.inputParticles, volumes)

        self._defineSourceRelation(self.inputVolumes, classes3D)
        self._defineSourceRelation(self.inputVolumes, volumes)

    # -------------------------- UTILS functions -------------------------------
    def _setSamplingArgs(self, args):
        """ Set sampling related params. """
        if self.doImageAlignment:
            args['--healpix_order'] = self.angularSamplingDeg.get()
            args['--offset_range'] = self.offsetSearchRangePix.get()
            args['--offset_step'] = (self.offsetSearchStepPix.get() *
                                     self._getSamplingFactor())
            if self.localAngularSearch:
                args['--sigma_ang'] = self.localAngularSearchRange.get() / 3.
        else:
            args['--skip_align'] = ''

    def _setContinueArgs(self, args, rLev):
        continueIter = self._lastIter(rLev)
        args['--continue'] = self._getFileName('optimiser', lev=self._level,
                                               rLev=rLev, iter=continueIter)

    def _getResetDeps(self):
        return "%s, %s" % (self._getInputParticles().getObjId(),
                           self.inputVolumes.get().getObjId())

    def _setNewEvalIds(self):
        self._evalIdsList.append(self._getRunLevId())

    def _getRunLevId(self, level=None, rLev=None):
        lev = self._level if level is None else level
        rLevel = self._rLev if rLev is None else rLev
        return "%02d.%03d" % (lev, rLevel)

    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == 'mergeClassesStep':
                return s
        return None

    def _getLevelPath(self, level, *paths):
        return self._getExtraPath('lev_%02d' % level, *paths)

    def _getRunPath(self, level, runLevel, *paths):
        return self._getLevelPath(level, 'rLev_%02d' % runLevel, *paths)

    def _defineInput(self, args):
        args['--i'] = self._getFileName('input_star', lev=self._level,
                                        rLev=self._rLev)

    def _defineOutput(self, args):
        args['--o'] = self._getRunPath(self._level, self._rLev, 'relion')

    def _getBoolAttr(self, attr=''):
        return getattr(attr, False)

    def _getLevRuns(self, level):
        if level == 1:
            return [1]
        else:
            l = level - 1
            mapsIds = [k for k,v in self.stopDict.items() if v is False]
            mapsLevelIds = [k for k in mapsIds if k.split('.')[0] == '%02d' % l]
            rLevList = [int(k.split('.')[-1]) for k,v in self.stopDict.items()
                        if v is False and k.split('.')[0] == '%02d' % l]

            print "level",  l, "self.stopDict",  self.stopDict
            print "mapsIds", mapsIds, "mapsLevelIds", mapsLevelIds
            print "rLevList", rLevList

            return rLevList

    def _getRefArg(self, mapId=None):
        if self._level <= 1:
            return self._convertVolFn(self.inputVolumes.get())
        if mapId is None:
            mapId = self._getRunLevId(level=self._level - 1)
        return self._getMapById(mapId)

    def _convertVolFn(self, inputVol):
        """ Return a new name if the inputFn is not .mrc """
        index, fn = inputVol.getLocation()
        return self._getTmpPath(replaceBaseExt(fn, '%02d.mrc' % index))

    def _convertVol(self, ih, inputVol):
        outputFn = self._convertVolFn(inputVol)

        if outputFn:
            xdim = self._getInputParticles().getXDim()
            img = ih.read(inputVol)
            img.scale(xdim, xdim, xdim)
            img.write(outputFn)

        return outputFn

    def _getMapById(self, mapId):
        level = int(mapId.split('.')[0])
        return self._getFileName('map', lev=level, id=mapId)

    def _copyLevelMaps(self):
        noOfLevRuns = self._getLevRuns(self._level)
        print('_copyLevelMaps, noOfLevRuns: ', noOfLevRuns)
        claseId = 0
        for rLev in noOfLevRuns:
            iters = self._lastIter(rLev)
            print ("last iteration  _copyLevelMaps:", iters)

            modelFn = self._getFileName('model', iter=iters,
                                        lev=self._level, rLev=rLev)
            modelMd = md.MetaData('model_classes@' + modelFn)
            for row in md.iterRows(modelMd):
                claseId += 1
                fn = row.getValue(md.RLN_MLMODEL_REF_IMAGE)

                mapId = self._getRunLevId(rLev=claseId)
                newFn = self._getMapById(mapId)
                print(('copy from %s to %s' % (fn, newFn)))
                copyFile(fn, newFn)
                self._mapsDict[fn] = mapId

    def _condToStop(self):
        outModel = self._getFileName('outputModel', lev=self._level)
        return False if os.path.exists(outModel) else True

    def _stopRunCondition(self, rLev, iter):
        x = np.array([])
        y = np.array([])

        for i in range(iter-6, iter, 1):
            print("iteration: ", i)
            x = np.append(x, i)
            modelFn = self._getFileName('model', iter=i,
                                        lev=self._level, rLev=rLev)
            print('filename: ', modelFn)
            modelMd = md.RowMetaData('model_general@' + modelFn)
            y = np.append(y, modelMd.getValue(md.RLN_MLMODEL_AVE_PMAX))
            print('values to stimate the slope: ', x, y, modelFn)

        slope = abs(np.polyfit(x, y, 1)[0])
        print("Slope: ", slope)
        return True if slope <= 0.01 else False

    def _evalStop(self):
        noOfLevRuns = self._getLevRuns(self._level)
        print("dataModel's loop to evaluate stop condition")

        x = [np.log10(k) for k, v in self.stopResLog.items()]
        y = [v for k, v in self.stopResLog.items()]
        slope, intercept, _, _, _ = stats.linregress(x, y)

        for rLev in noOfLevRuns:
            iters = self._lastIter(rLev)
            modelFn = self._getFileName('model', iter=iters,
                                        lev=self._level, rLev=rLev)
            modelMd = md.MetaData('model_classes@' + modelFn)
            partSize = md.getSize(self._getFileName('input_star',
                                                    lev=self._level, rLev=rLev))

            for row in md.iterRows(modelMd):
                fn = row.getValue(md.RLN_MLMODEL_REF_IMAGE)
                mapId = self._mapsDict[fn]
                ssnr = 1/row.getValue('rlnEstimatedResolution')
                classSize = row.getValue('rlnClassDistribution') * partSize
                const = ssnr*np.log10(classSize)

                expectedRes = slope*np.log10(classSize) + intercept

                ptcStop = self.minPartsToStop.get()

                print("Values: ssnr %0.4f, parts %d, expectedRes %0.4f, "
                      "slope %0.4f, intercept %0.4f" %(ssnr, classSize,
                      expectedRes, slope, intercept))

                if ssnr < expectedRes or classSize < ptcStop:
                    self.stopDict[mapId] = True
                    if not bool(self._clsIdDict):
                        self._clsIdDict[mapId] = 1
                    else:
                        classId = sorted(self._clsIdDict.values())[-1] + 1
                        self._clsIdDict[mapId] = classId
                else:
                    self.stopDict[mapId] = False

    def _mergeMetaDatas(self):
        noOfLevRuns = self._getLevRuns(self._level)

        print("entering in the loop to merge dataModel")
        for rLev in noOfLevRuns:
            rLevId = self._getRunLevId(rLev=rLev)
            self._lastCls = None

            self._mergeModelStar(rLev)
            self._mergeDataStar(rLev)

            self._doneList.append(rLevId)
        print("finished _mergeMetaDatas function")

    def _mergeModelStar(self, rLev):
        iters = self._lastIter(rLev)
        print ("last iteration: _mergeModelStar ", iters)


        #metadata to save all models that continues
        outModel = self._getFileName('outputModel', lev=self._level)
        outMd = self._getMetadata(outModel)

        #metadata to save all final models
        finalModel = self._getFileName('rawFinalModel')
        finalMd = self._getMetadata(finalModel)
        print "final MD has at beggining: ", finalMd, "\n*******************\n"

        #read model metadata
        modelFn = self._getFileName('model', iter=iters,
                                    lev=self._level, rLev=rLev)
        modelMd = md.MetaData('model_classes@' + modelFn)

        for row in md.iterRows(modelMd):
            refLabel = md.RLN_MLMODEL_REF_IMAGE
            fn = row.getValue(refLabel)
            mapId = self._mapsDict[fn]
            newMap = self._getMapById(mapId)
            row.setValue(refLabel, newMap)
            row.writeToMd(modelMd, row.getObjId())
            if self.stopDict[mapId]:
                print "this MapId %s is finished" % mapId
                row.addToMd(finalMd)
            else:
                print "this MapId %s is not finished" % mapId
                row.addToMd(outMd)

        if outMd.size() != 0:
            outMd.write(outModel)

        if finalMd.size() != 0:
            print "final MD has: ", finalMd
            finalMd.write('model_classes@' + finalModel)

    def _mergeDataStar(self, rLev):
        iters = self._lastIter(rLev)
        print ("last iteration _mergeDataStar:", iters)

        #metadata to save all particles that continues
        outData = self._getFileName('outputData', lev=self._level)
        outMd = self._getMetadata(outData)

        #metadata to save all final particles
        finalData = self._getFileName('rawFinalData')
        finalMd = self._getMetadata(finalData)

        imgStar = self._getFileName('data', iter=iters,
                                    lev=self._level, rLev=rLev)
        mdData = md.MetaData(imgStar)

        for row in md.iterRows(mdData, sortByLabel=md.RLN_PARTICLE_CLASS):
            clsPart = row.getValue(md.RLN_PARTICLE_CLASS)
            rMap = self._getFileName('relionMap', lev=self._level,
                                     iter=iters,
                                     ref3d=clsPart, rLev=rLev)
            mapId = self._mapsDict[rMap]
            if self.stopDict[mapId]:
                classId = self._clsIdDict[mapId]
                row.setValue(md.RLN_PARTICLE_CLASS, classId)
                row.addToMd(finalMd)
            else:
                classId = int(mapId.split('.')[1])
                row.setValue(md.RLN_PARTICLE_CLASS, classId)
                row.addToMd(outMd)

        if finalMd.size() != 0:
            finalMd.write(finalData)

        if outMd.size() != 0:
            outMd.write(outData)

    def _getMetadata(self, file='filepath'):
        fList = file.split("@")
        return md.MetaData(file) if os.path.exists(fList[-1]) else md.MetaData()

    def _getAverageVol(self, listVol=[]):
        listVol = self._getPathMaps() if not bool(listVol) else listVol

        print('creating average map: ', listVol)
        try:
            avgVol = self._getFileName('avgMap', lev=self._level)
        except:
            avgVol = self._getPath('map_average.mrc')
        npAvgVol, dType = self._doAverageMaps(listVol)
        print('alignining each volume vs. reference')
        print('saving average volume')
        saveMrc(npAvgVol.astype(dType), avgVol)

    def _doAverageMaps(self, listVol):
        for vol in listVol:
            npVol = loadMrc(vol, False)
            if vol == listVol[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)
            npAvgVol += npVol

        npAvgVol = np.divide(npAvgVol, len(listVol))
        return npAvgVol, dType


    def _getPathMaps(self, filename="*.mrc"):
        filesPath = self._getLevelPath(self._level, filename)
        return sorted(glob(filesPath))

    def _alignVolumes(self):
        # Align all volumes inside a level
        Plugin.setEnviron()
        listVol = self._getPathMaps()
        print('reading volumes as numpy arrays')
        avgVol = self._getFileName('avgMap', lev=self._level)
        npAvgVol = loadMrc(avgVol, writable=False)
        dType = npAvgVol.dtype

        print('alignining each volume vs. reference')
        for vol in listVol:
            npVolAlign = loadMrc(vol, False)
            npVolFlipAlign = np.fliplr(npVolAlign)

            axis, shifts, angles, score = alignVolumes(npVolAlign, npAvgVol)
            axisf, shiftsf, anglesf, scoref = alignVolumes(npVolFlipAlign,
                                                           npAvgVol)
            if scoref > score:
                npVol = applyTransforms(npVolFlipAlign, shiftsf, anglesf, axisf)
            else:
                npVol = applyTransforms(npVolAlign, shifts, angles, axis)

            print('saving rot volume %s' % vol)
            saveMrc(npVol.astype(dType), vol)

    def _mrcToNp(self, volList):
        listNpVol = []
        for vol in volList:
            volNp = loadMrc(vol, False)
            dim = volNp.shape[0]
            lenght = dim**3
            volList = volNp.reshape(lenght)
            listNpVol.append(volList)
        return listNpVol, listNpVol[0].dtype

    def _estimatePCA(self):
        listVol = self._getFinalMaps()

        volNp = loadMrc(listVol[0], False)
        dim = volNp.shape[0]
        dType = volNp.dtype

        matProj, newBaseAxis = self._doPCA(listVol)

        for i, volNewBaseList in enumerate(newBaseAxis):
            volBase = volNewBaseList.reshape((dim, dim, dim))
            nameVol = 'volume_base_%02d.mrc' % (i+1)
            print('-------------saving map %s-----------------' % nameVol)
            print('Dimensions are: ', volBase.shape)
            saveMrc(volBase.astype(dType),
                    self._getLevelPath(self._level, nameVol))
            print('------------map %s stored------------------' % nameVol)
        return matProj

    def _doPCA(self, listVol):
        npAvgVol, _ = self._doAverageMaps(listVol)

        listNpVol, _ = self._mrcToNp(listVol)

        covMatrix = np.cov(listNpVol)
        u, s, vh = np.linalg.svd(covMatrix)
        cuttOffMatrix = sum(s) *0.95
        sCut = 0

        for i in s:
            if cuttOffMatrix > 0:
                cuttOffMatrix = cuttOffMatrix - i
                sCut += 1
            else:
                break

        eigValsFile = 'eigenvalues.txt'
        self._createMFile(s, eigValsFile)

        eigVecsFile = 'eigenvectors.txt'
        self._createMFile(vh, eigVecsFile)

        vhDel = np.transpose(np.delete(vh, np.s_[sCut:vh.shape[1]], axis=0))
        self._createMFile(vhDel, 'matrix_vhDel.txt')

        newBaseAxis = vhDel.T.dot(listNpVol)
        matProj = np.transpose(np.dot(newBaseAxis, np.transpose(listNpVol)))
        projFile = 'projection_matrix.txt'
        self._createMFile(matProj, projFile)
        return matProj, newBaseAxis

    def _getFinalMaps(self):
        return [self._getMapById(k) for k in self._getFinalMapIds()]

    def _getFinalMapIds(self):
        return [k for k, v in self.stopDict.items() if v is True]

    def _lastIter(self, rLev=None):
        self._createIterTemplates(rLev)
        return self._getIterNumber(-1)

    def _clusteringData(self, matProj):
        method = self.classMethod.get()
        if method == 0:
            return self._doSklearnKmeans(matProj)
        else:
            return self._doSklearnAffProp(matProj)

    def _doSklearnKmeans(self, matProj):
        from sklearn.cluster import KMeans
        print('projections: ', matProj.shape[1])
        kmeans = KMeans(n_clusters=matProj.shape[1]).fit(matProj)
        return kmeans.labels_

    def _doSklearnAffProp(self, matProj):
        from sklearn.cluster import AffinityPropagation
        ap = AffinityPropagation(damping=0.9).fit(matProj)
        print("cluster_centers", ap.cluster_centers_)
        return ap.labels_

    # def _getDistance(self, m1, m2, neg=False):
    #     #estimatation of the distance bt row vectors
    #     distances = np.zeros(( m1.shape[0], m1.shape[1]))
    #     for i, row in enumerate(m2):
    #         distances[:, i] = np.linalg.norm(m1 - row, axis=1)
    #     if neg == True:
    #         distances = -distances
    #     return distances

    def _createMFile(self, matrix, name='matrix.txt'):
        f = open(name, 'w')
        for list in matrix:
            s = "%s\n" % list
            f.write(s)
        f.close()

    def _loadClassesInfo(self):
        """ Read some information about the produced Relion 3D classes
        from the *model.star file.
        """
        self._classesInfo = {}  # store classes info, indexed by class id

        # this is for the level
        # mdModel = self._getFileName('outputModel', lev=self._level)

        #this file is the final file model
        mdModel = self._getFileName('finalModel')
        modelStar = md.MetaData('model_classes@' + mdModel)

        for classNumber, row in enumerate(md.iterRows(modelStar)):
            index, fn = relionToLocation(row.getValue('rlnReferenceImage'))
            # Store info indexed by id, we need to store the row.clone() since
            # the same reference is used for iteration
            self._classesInfo[classNumber + 1] = (index, fn, row.clone())

    def _fillClassesFromIter(self, clsSet):
        """ Create the SetOfClasses3D from a given iteration. """
        self._loadClassesInfo()
        dataStar = self._getFileName('finalData')
        clsSet.classifyItems(updateItemCallback=self._updateParticle,
                             updateClassCallback=self._updateClass,
                             itemDataIterator=md.iterRows(dataStar))

    def _updateParticle(self, item, row):
        item.setClassId(row.getValue(md.RLN_PARTICLE_CLASS))
        item.setTransform(rowToAlignment(row, em.ALIGN_PROJ))

        item._rlnLogLikeliContribution = em.Float(
            row.getValue('rlnLogLikeliContribution'))
        item._rlnMaxValueProbDistribution = em.Float(
            row.getValue('rlnMaxValueProbDistribution'))
        item._rlnGroupName = em.String(row.getValue('rlnGroupName'))

    def _updateClass(self, item):
        classId = item.getObjId()
        if classId in self._classesInfo:
            index, fn, row = self._classesInfo[classId]
            fn += ":mrc"
            item.setAlignmentProj()
            item.getRepresentative().setLocation(index, fn)
            item._rlnclassDistribution = em.Float(
                row.getValue('rlnClassDistribution'))
            item._rlnAccuracyRotations = em.Float(
                row.getValue('rlnAccuracyRotations'))
            item._rlnAccuracyTranslations = em.Float(
                row.getValue('rlnAccuracyTranslations'))

