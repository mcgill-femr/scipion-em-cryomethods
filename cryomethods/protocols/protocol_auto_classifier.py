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
import copy
import numpy as np
from glob import glob
from collections import Counter

import pyworkflow.em as em
import pyworkflow.em.metadata as md
import pyworkflow.protocol.constants as cons
import pyworkflow.protocol.params as params
from pyworkflow.utils import makePath, copyFile, replaceBaseExt

from cryomethods import Plugin
from cryomethods.convert import (writeSetOfParticles, rowToAlignment,
                                 relionToLocation, loadMrc, saveMrc,
                                 alignVolumes, applyTransforms)

from .protocol_base import ProtocolBase


class ProtAutoClassifier(ProtocolBase):
    _label = 'auto classifier'
    IS_AUTOCLASSIFY = True

    def __init__(self, **args):
        ProtocolBase.__init__(self, **args)
        self._level = 1
        self._rLev = 1

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
            'map': self.levDir + 'map_rLev-%(rLev)03d.mrc',
            'avgMap': self.levDir + 'map_average.mrc',
            'relionMap': self.rLevDir + 'relion_it%(iter)03d_class%(ref3d)03d.mrc',
            'outputModel': self.levDir + 'output_model.star',
            'data': self.rLevIter + 'data.star',
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
            'all_avgPmax_xmipp': self._getTmpPath(
                'iterations_avgPmax_xmipp.xmd'),
            'all_changes_xmipp': self._getTmpPath(
                'iterations_changes_xmipp.xmd'),
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

    def _createIterTemplates(self):
        """ Setup the regex on how to find iterations. """
        self._iterTemplate = self._getFileName('data', lev=self._level,
                                               rLev=self._rLev,
                                               iter=0).replace('000', '???')
        # Iterations will be identify by _itXXX_ where XXX is the iteration
        # number and is restricted to only 3 digits.
        self._iterRegex = re.compile('_it(\d{3,3})_')
        self._classRegex = re.compile('_class(\d{2,2}).')

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
        levelRuns = self._getLevRuns(self._level)

        self._insertFunctionStep('convertInputStep',
                                 self._getResetDeps(),
                                 self.copyAlignment,
                                 prerequisites=deps)

        for rLev in range(1, levelRuns + 1):
            self._rLev = rLev  # Just to generate the proper input star file.
            self._insertClassifyStep()
            self._setNewEvalIds(rLev)
        evStep = self._insertEvaluationStep()
        deps.append(evStep)
        return deps

    def _insertEvaluationStep(self):
        evalDep = self._insertFunctionStep('evaluationStep')
        return evalDep

    def _stepsCheck(self):
        print('Just passing through this')
        self.finished = False
        if self._level == self.level.get():  # condition to stop the cycle
            self.finished = True
        outputStep = self._getFirstJoinStep()
        if self.finished:  # Unlock createOutputStep if finished all jobs
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)
                print('outputStep After setStatus')
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
    def convertInputStep(self, resetDeps, copyAlignment):
        """ Create the input file in STAR format as expected by Relion.
        If the input particles comes from Relion, just link the file.
        """
        imgStar = self._getFileName('input_star', lev=self._level, rLev=1)

        if self._level == 1:
            makePath(self._getRunPath(self._level, 1))
            imgSet = self._getInputParticles()
            self.info("Converting set from '%s' into '%s'" %
                      (imgSet.getFileName(), imgStar))

            # Pass stack file as None to avoid write the images files
            # If copyAlignment is set to False pass alignType to ALIGN_NONE
            alignType = imgSet.getAlignment() if copyAlignment \
                else em.ALIGN_NONE

            hasAlign = alignType != em.ALIGN_NONE
            alignToPrior = hasAlign and self._getBoolAttr('alignmentAsPriors')
            fillRandomSubset = hasAlign and self._getBoolAttr(
                'fillRandomSubset')

            writeSetOfParticles(imgSet, imgStar, self._getExtraPath(),
                                alignType=alignType,
                                postprocessImageRow=self._postprocessParticleRow,
                                fillRandomSubset=fillRandomSubset)
            if alignToPrior:
                self._copyAlignAsPriors(imgStar, alignType)

            if self.doCtfManualGroups:
                self._splitInCTFGroups(imgStar)

            self._convertVol(em.ImageHandler(), self.inputVolumes.get())
        else:
            noOfLevRuns = self._getLevRuns(self._level)
            lastCls = None

            prevStar = self._getFileName('outputData', lev=self._level - 1)
            mdData = md.MetaData(prevStar)
            print('how many run levels? %d' % noOfLevRuns)

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

    def evaluationStep(self):
        noOfLevRuns = self._getLevRuns(self._level)
        iters = self.numberOfIterations.get()
        self._newClass = 0
        self._clsDict = {}
        outModel = self._getFileName('outputModel', lev=self._level)
        outStar = self._getFileName('outputData', lev=self._level)
        mdInput = md.MetaData()
        outMd = md.MetaData()

        print("entering in the loop to merge dataModel")
        for rLev in range(1, noOfLevRuns + 1):
            rLevId = self._getRunLevId(self._level, rLev)
            self._lastCls = None

            mdModel = self._getFileName('model', iter=iters,
                                        lev=self._level, rLev=rLev)
            print('Filename model star: %s' % mdModel)
            self._mergeDataStar(outStar, mdInput, iters, rLev)
            self._mergeModelStar(outMd, mdModel, rLev)

            self._doneList.append(rLevId)

        outMd.write('model_classes@' + outModel)
        mdInput.write(outStar)

        self._alignVolumes()
        self._estimatePCA()
        print('Finishing evaluation step')

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

    def _getResetDeps(self):
        return "%s, %s" % (self._getInputParticles().getObjId(),
                           self.inputVolumes.get().getObjId())

    def _setNewEvalIds(self, levelRuns):
        self._evalIdsList.append(self._getRunLevId(self._level, levelRuns))

    def _getRunLevId(self, level, levelRuns):
        return "%s.%s" % (level, levelRuns)

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
        args['--i'] = self._getFileName('input_star', lev=self._level,
                                        rLev=self._rLev)
        args['--o'] = self._getRunPath(self._level, self._rLev, 'relion')

    def _getBoolAttr(self, attr=''):
        return getattr(attr, False)

    def _getLevRuns(self, level):
        clsNumber = self.numberOfClasses.get()
        return clsNumber ** (level - 1)

    def _getRefArg(self):
        if self._level == 1:
            return self._convertVolFn(self.inputVolumes.get())
        return self._getFileName('map', lev=self._level - 1, rLev=self._rLev)

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

    def _mergeDataStar(self, outStar, mdInput, iter, rLev):
        imgStar = self._getFileName('data', iter=iter,
                                    lev=self._level, rLev=rLev)
        mdData = md.MetaData(imgStar)

        print("reading %s and begin the loop" % imgStar)
        for row in md.iterRows(mdData, sortByLabel=md.RLN_PARTICLE_CLASS):
            clsPart = row.getValue(md.RLN_PARTICLE_CLASS)
            if clsPart != self._lastCls:
                self._newClass += 1
                self._clsDict['%s.%s' % (rLev, clsPart)] = self._newClass
                self._lastCls = clsPart
                # write symlink to new Maps
                relionFn = self._getFileName('relionMap', lev=self._level,
                                             iter=self._getnumberOfIters(),
                                             ref3d=clsPart, rLev=rLev)
                newFn = self._getFileName('map', lev=self._level,
                                          rLev=self._newClass)
                print(('link from %s to %s' % (relionFn, newFn)))
                copyFile(relionFn, newFn)

            row.setValue(md.RLN_PARTICLE_CLASS, self._newClass)
            row.addToMd(mdInput)
        print("writing %s and ending the loop" % outStar)

    def _mergeModelStar(self, outMd, mdInput, rLev):
        modelStar = md.MetaData('model_classes@' + mdInput)

        for classNumber, row in enumerate(md.iterRows(modelStar)):
            # objId = self._clsDict['%s.%s' % (rLev, classNumber+1)]
            row.addToMd(outMd)

    def _loadClassesInfo(self):
        """ Read some information about the produced Relion 3D classes
        from the *model.star file.
        """
        self._classesInfo = {}  # store classes info, indexed by class id
        mdModel = self._getFileName('outputModel', lev=self._level)

        modelStar = md.MetaData('model_classes@' + mdModel)

        for classNumber, row in enumerate(md.iterRows(modelStar)):
            index, fn = relionToLocation(row.getValue('rlnReferenceImage'))
            # Store info indexed by id, we need to store the row.clone() since
            # the same reference is used for iteration
            self._classesInfo[classNumber + 1] = (index, fn, row.clone())

    def _fillClassesFromIter(self, clsSet):
        """ Create the SetOfClasses3D from a given iteration. """
        self._loadClassesInfo()
        dataStar = self._getFileName('outputData', lev=self._level)
        clsSet.classifyItems(updateItemCallback=self._updateParticle,
                             updateClassCallback=self._updateClass,
                             itemDataIterator=md.iterRows(dataStar,
                                                          sortByLabel=md.RLN_IMAGE_ID))

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

    def _alignVolumes(self):
        Plugin.setEnviron()

        filesPath = self._getLevelPath(self._level, "*.mrc")
        listVol = sorted(glob(filesPath))
        lenVols = len(listVol)
        volRef = listVol.pop(0)

        print('creating average map')
        avgVol = self._getFileName('avgMap', lev=self._level)
        copyFile(volRef, avgVol)

        print('reading volumes as numpy arrays')
        npRef = loadMrc(volRef, writable=False)
        npAvgVol = loadMrc(avgVol, writable=True)

        print('alignining each volume vs. reference')
        for vol in listVol:
            npVolAlign = loadMrc(vol, False)
            npVolFlipAlign = np.fliplr(npVolAlign)

            axis, shifts, angles, score = alignVolumes(npVolAlign, npRef)
            axisf, shiftsf, anglesf, scoref = alignVolumes(npVolFlipAlign,
                                                           npRef)
            if scoref > score:
                npVol = applyTransforms(npVolFlipAlign, shiftsf, anglesf, axisf)
            else:
                npVol = applyTransforms(npVolAlign, shifts, angles, axis)

            npAvgVol += npVol
            print('saving rot volume %s' % vol)
            saveMrc(npVol, vol)

        npAvgVol = np.divide(npAvgVol, lenVols)
        print('saving average volume')
        saveMrc(npAvgVol, avgVol)

    def _estimatePCA(self):
        Plugin.setEnviron()
        avgVol = self._getFileName('avgMap', lev=self._level)
        npAvgVol = loadMrc(avgVol, False)
        listNpVol = []
        m = []
        dType = npAvgVol.dtype

        filePaths = self._getLevelPath(self._level, "map_rLev-???.mrc")
        listVol = sorted(glob(filePaths))

        for vol in listVol:
            volNp = loadMrc(vol, False)
            dim = volNp.shape[0]
            lenght = dim**3
            # Now, using diff volume to estimate PCA
            diffVol = volNp - npAvgVol
            volList = diffVol.reshape(lenght)
            listNpVol.append(volList)

        covMatrix = np.cov(listNpVol)
        u, s, vh = np.linalg.svd(covMatrix)
        cuttOffMatrix = sum(s) *0.95
        sCut = 0

        print('cuttOffMatrix & s: ', cuttOffMatrix, s)


        for i in s:
            print('cuttOffMatrix: ', cuttOffMatrix)
            if cuttOffMatrix > 0:
                print("Pass, i = %s " %i)
                cuttOffMatrix = cuttOffMatrix - i
                sCut += 1
            else:
                break

        print('sCut: ', sCut)

        eigValsFile = self._getLevelPath(self._level, 'eigenvalues.txt')
        self._createMFile(s, eigValsFile)

        eigVecsFile = self._getLevelPath(self._level, 'eigenvectors.txt')
        self._createMFile(vh, eigVecsFile)

        vhDel = np.transpose(np.delete(vh, np.s_[sCut:vh.shape[1]], axis=0))
        self._createMFile(vhDel, 'matrix_vhDel.txt')

        print(' this is the matrix "vhDel": ', vhDel)

        newBaseAxis = vhDel.T.dot(listNpVol)

        for i, volNewBaseList in enumerate(newBaseAxis):
            volBase = volNewBaseList.reshape((dim, dim, dim))
            nameVol = 'volume_base_%02d.mrc' % (i+1)
            print('-------------saving map %s-----------------' % nameVol)
            print('Dimensions are: ', volBase.shape)
            saveMrc(volBase.astype(dType),
                    self._getLevelPath(self._level, nameVol))
            print('------------map %s stored------------------' % nameVol)
        matProj = np.transpose(np.dot(newBaseAxis, np.transpose(listNpVol)))

        projFile = self._getLevelPath(self._level, 'projection_matrix.txt')
        self._createMFile(matProj, projFile)


        #K-means method to split the classes:
        # Number of training data
        n = matProj.shape[0]
        # Number of features in the data
        c = matProj.shape[1]
        print('Data: ', n, 'features:', c)
        # Generate random centers, here we use sigma and mean to ensure it
        # represent the whole data

        mean = np.mean(matProj, axis = 0)
        std = np.std(matProj, axis = 0)
        centers = np.random.randn(sCut,c)*std + mean


        centers_old = np.zeros(centers.shape) # to store old centers
        centers_new = copy.deepcopy(centers) # Store new centers

        clusters = np.zeros(n)
        distances = np.zeros((n,sCut))

        error = np.linalg.norm(centers_new - centers_old)
        print('first error: ', error)
        # When, after an update, the estimate of that center stays the same, exit loop
        print('while loop begins', matProj)
        count = 1
        while (error != 0) and (count <= 10):
            print('Measure the distance to every center')
            for i in range(sCut):
                distances[:,i] = np.linalg.norm(matProj - centers[i], axis=1)
                print('Distances: ', distances[:,i], '\n\n')

            print('Assign all training data to closest center')
            clusters = np.argmin(distances, axis = 1)
            print('clusters: ', clusters)

            centers_old = copy.deepcopy(centers_new)
            print('Calculate mean for every cluster and update the center')
            for i in range(sCut):
                centers_new[i] = np.mean(matProj[clusters == i], axis=0)
            print("----Centers NEW: ", centers_new, centers_old)
            error = np.linalg.norm(centers_new - centers_old)
            count += 1
            print('error: ', error, 'count: ', count)
        print('clusters: ', clusters)


        # matDist = []
        # for list1 in matProj:
        #     rows = []
        #     for list2 in matProj:
        #         v = 0
        #         for i,j in izip(list1, list2):
        #             v += (i - j)**2
        #         rows.append(v**0.5)
        #     matDist.append(rows)
        # distFile = self._getLevelPath(self._level, 'distance_matrix.txt')
        # self._createMFile(matDist, distFile)

    def _createMFile(self, matrix, name='matrix.txt'):
        f = open(name, 'w')
        for list in matrix:
            s = "%s\n" % list
            f.write(s)
        f.close()

