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
from scipy import stats, interpolate
from glob import glob
from collections import Counter, defaultdict

import pyworkflow.em as em
import pyworkflow.em.metadata as md
import pyworkflow.protocol.constants as cons
from pyworkflow.utils import (makePath, copyFile, replaceBaseExt)

from cryomethods import Plugin
from cryomethods.convert import (rowToAlignment, relionToLocation, loadMrc,
                                 saveMrc, alignVolumes, applyTransforms)
from .protocol_base import ProtocolBase


class ProtAutoBase(ProtocolBase):

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
        """ Implemented in subclasses. """
        pass

    def _createIterTemplates(self, rLev=None):
        """ Setup the regex on how to find iterations. """
        rLev = self._rLev if rLev is None else rLev
        self._iterTemplate = self._getFileName('data', lev=self._level,
                                               rLev=rLev,
                                               iter=0).replace('000', '???')
        # Iterations will be identify by _itXXX_ where XXX is the iteration
        # number and is restricted to only 3 digits.
        self._iterRegex = re.compile('_it(\d{3,3})_')

    def _createRLevTemplates(self):
        """ Setup the regex on how to find iterations. """
        self._rLevTemplate = self._getFileName('input_star', lev=self._level,
                                               rLev=0).replace('000', '???')
        self._rLevRegex = re.compile('rLev-(\d{3,3}).')


    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        """ Implemented in subclasses. """
        pass

    # -------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._level = 0 if self.IS_3D and self.useReslog else 1
        self._rLev = 1
        self._evalIdsList = []
        self._doneList = []
        self.stopDict = {}
        self.stopResLog = {}
        self._mapsDict = {}
        self._clsIdDict = {}

        self._initialize()
        deps = self._insertLevelSteps()
        self._insertFunctionStep('mergeClassesStep', wait=True,
                                 prerequisites=deps)
        if self.doGrouping:
            self._insertFunctionStep('runFinalClassifStep')
            self._insertFunctionStep('createFinalFilesStep')
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
        for i in range(2, 10, 1):
            self._rLev = i  # Just to generate the proper input star file.
            self._insertClassifyStep(K=1)
            self._insertFunctionStep('resLogStep', i)
            dep = self._insertFunctionStep('addDoneListStep', i)
            self._setNewEvalIds()
        return [dep]

    def _insertEvaluationStep(self, deps):
        evalDep = self._insertFunctionStep('evaluationStep', prerequisites=deps)
        return evalDep

    def _stepsCheck(self):
        if Counter(self._evalIdsList) == Counter(self._doneList):
            mergeStep = self._getFirstJoinStep()
            if self._condToStop() and self._level > 0:
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
        """ Implemented in subclasses. """
        pass

    def resLogStep(self, rLev):
        iters = self._lastIter(rLev)
        modelFn = self._getFileName('model', iter=iters,lev=self._level,
                                    rLev=rLev)
        modelMd = self._getMetadata('model_class_1@' + modelFn)
        f = self._getFunc(modelMd)
        imgStar = self._getFileName('data', iter=iters, lev=self._level,
                                    rLev=rLev)
        mdData = self._getMetadata(imgStar)
        size = np.math.log10(mdData.size())
        self.stopResLog[size] = f(1)
        print("stopResLog: ", self.stopResLog)

    def addDoneListStep(self, rLev):
        rLevId = self._getRunLevId(level=0, rLev=rLev)
        self._doneList.append(rLevId)

    def evaluationStep(self):
        """ Implemented in subclasses. """
        pass

    def mergeClassesStep(self):

        if self.doGrouping:

            makePath(self._getLevelPath(self._level))
            # matrix = self._estimatePCA()
            listVol = self._getFinalMaps()
            matrix, _ = self._mrcToNp(listVol)

            labels = self._clusteringData(matrix)
            prevStar = self._getFileName('rawFinalData')
            mdData = md.MetaData(prevStar)

            clsChange = 0
            for row in md.iterRows(mdData, sortByLabel=md.RLN_PARTICLE_CLASS):
                clsPart = row.getValue(md.RLN_PARTICLE_CLASS)
                newClass = labels[clsPart-1] + 1
                row.setValue(md.RLN_PARTICLE_CLASS, newClass)
                if not newClass == clsChange:
                    if not clsChange == 0:
                        mdOutput.write(fn)
                    clsChange = newClass
                    fn = self._getFileName('input_star', lev=self._level,
                                           rLev=newClass)
                    mdOutput = md.MetaData()
                row.addToMd(mdOutput)
            mdOutput.write(fn)

        else:
            prevData = self._getFileName('rawFinalData')
            finalData = self._getFileName('finalData')
            prevModel = self._getFileName('rawFinalModel')
            finalModel = self._getFileName('finalModel')
            copyFile(prevData, finalData)
            copyFile(prevModel, finalModel)

    def runFinalClassifStep(self):
        mapIds = self._getFinalMapIds()
        print ("runFinalClassifStep rLev list: ", self._getRLevList())
        for rLev in self._getRLevList():
            makePath(self._getRunPath(self._level, rLev))
            self._rLev = rLev
            iters = 15
            args = {}

            self._setNormalArgs(args)
            self._setComputeArgs(args)

            args['--K'] = 1
            args['--iter'] = iters

            mapId = mapIds[rLev - 1]
            if self.IS_3D:
                args['--ref'] = self._getRefArg(mapId)

            params = self._getParams(args)
            self.runJob(self._getProgram(), params)

    def createFinalFilesStep(self):
        # -----metadata to save all final models-------
        finalModel = self._getFileName('finalModel')
        finalModelMd = self._getMetadata()

        # -----metadata to save all final particles-----
        finalData = self._getFileName('finalData')
        finalDataMd = self._getMetadata()

        for rLev in self._getRLevList():
            it = self._lastIter(rLev)
            modelFn = self._getFileName('model', iter=it,
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

            imgRow.addToMd(finalModelMd)

            dataFn = self._getFileName('data', iter=it,
                                       lev=self._level, rLev=rLev)
            dataMd = self._getMetadata(dataFn)
            for row in md.iterRows(dataMd):
                row.setValue(md.RLN_PARTICLE_CLASS, rLev)
                row.addToMd(finalDataMd)

        finalDataMd.write(finalData)
        finalModelMd.write('model_classes@' + finalModel)

    def createOutputStep(self):
        """ Implemented in subclasses. """
        pass

    # -------------------------- UTILS functions -------------------------------
    def _setSamplingArgs(self, args):
        """ Implemented in subclasses. """
        pass

    def _getResetDeps(self):
        """ Implemented in subclasses. """
        pass

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
            return rLevList

    def _getRefArg(self, mapId=None):
        if self.IS_3D:
            if self._level <= 1:
                return self._convertVolFn(self.inputVolumes.get())
            if mapId is None:
                mapId = self._getRunLevId(level=self._level - 1)
            return self._getMapById(mapId)
        else:  # 2D
            if self.referenceAverages.get():
                return self._getRefStar()
        return None # No --ref should be used at this point


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
        """ Implemented in subclasses. """
        pass

    def _copyLevelMaps(self):
        noOfLevRuns = self._getLevRuns(self._level)
        claseId = 0
        for rLev in noOfLevRuns:
            iters = self._lastIter(rLev)
            modelFn = self._getFileName('model', iter=iters,
                                        lev=self._level, rLev=rLev)
            modelMd = md.MetaData('model_classes@' + modelFn)

            for row in md.iterRows(modelMd):
                claseId += 1
                fn = row.getValue(md.RLN_MLMODEL_REF_IMAGE)
                clasDist = row.getValue('rlnClassDistribution')
                if self._getClasDistCond(clasDist):
                    print("Meth: _copyLevelMaps, clasDist: ", clasDist)
                    mapId = self._getRunLevId(rLev=claseId)
                    newFn = self._getMapById(mapId)
                    ih = em.ImageHandler()
                    ih.convert(fn, newFn)
                    self._mapsDict[fn] = mapId

    def _condToStop(self):
        outModel = self._getFileName('outputModel', lev=self._level)
        return False if os.path.exists(outModel) else True

    def _evalStop(self):
        noOfLevRuns = self._getLevRuns(self._level)

        if self.IS_3D and self.useReslog:
            slope, y0, err = self._getReslogVars()

        for rLev in noOfLevRuns:
            iters = self._lastIter(rLev)
            modelFn = self._getFileName('model', iter=iters,
                                        lev=self._level, rLev=rLev)
            modelMd = md.MetaData('model_classes@' + modelFn)
            partSize = md.getSize(self._getFileName('input_star',
                                                    lev=self._level, rLev=rLev))
            clsId = 1
            for row in md.iterRows(modelMd):
                fn = row.getValue(md.RLN_MLMODEL_REF_IMAGE)
                clasDist = row.getValue('rlnClassDistribution')
                classSize = clasDist * partSize

                if self._getClasDistCond(clasDist):
                    mapId = self._mapsDict[fn]
                    ptcStop = self.minPartsToStop.get()
                    if classSize < 0.95*partSize:
                        if self.IS_3D and self.useReslog:
                            suffixSsnr = 'model_class_%d@' % clsId
                            ssnrMd = md.MetaData(suffixSsnr + modelFn)
                            f = self._getFunc(ssnrMd)
                            ExpcVal = slope*np.math.log10(classSize) + y0
                            val = f(1) + (2*err)
                            clsId += 1
                            print("StopValues: \n"
                                  "Val SSnr=1: %0.4f, parts %d, ExpcVal "
                                  "%0.4f" % (val, classSize, ExpcVal))
                            evalSlope = val < ExpcVal and self._level > 2
                        else:
                            evalSlope = False
                    else:
                        evalSlope = True

                    if classSize < ptcStop or evalSlope:
                        self.stopDict[mapId] = True
                        if not bool(self._clsIdDict):
                            self._clsIdDict[mapId] = 1
                        else:
                            classId = sorted(self._clsIdDict.values())[-1] + 1
                            self._clsIdDict[mapId] = classId
                    else:
                        self.stopDict[mapId] = False

    def _getClasDistCond(self, clasDist):
        return clasDist > 0.05

    def _getReslogVars(self):
        x = [k for k, v in self.stopResLog.items()]
        y = [v for k, v in self.stopResLog.items()]
        slope, y0, _, _, err = stats.linregress(x, y)
        print("EvalStop mx+n: m: %0.4f, n %0.4f,  err %0.4f" % (slope, y0, err))
        return slope, y0, err

    def _mergeMetaDatas(self):
        noOfLevRuns = self._getLevRuns(self._level)

        for rLev in noOfLevRuns:
            rLevId = self._getRunLevId(rLev=rLev)
            self._lastCls = None

            self._mergeModelStar(rLev)
            self._mergeDataStar(rLev, callback=self._getRelionFn)

            self._doneList.append(rLevId)

    def _getRelionFn(self, iters, rLev, clsPart):
        "Implemented in subclasses"
        pass

    def _mergeModelStar(self, rLev):
        iters = self._lastIter(rLev)

        #metadata to save all models that continues
        outModel = self._getFileName('outputModel', lev=self._level)
        outMd = self._getMetadata(outModel)

        #metadata to save all final models
        finalModel = self._getFileName('rawFinalModel')
        finalMd = self._getMetadata(finalModel)

        #read model metadata
        modelFn = self._getFileName('model', iter=iters,
                                    lev=self._level, rLev=rLev)
        modelMd = md.MetaData('model_classes@' + modelFn)

        for row in md.iterRows(modelMd):
            refLabel = md.RLN_MLMODEL_REF_IMAGE
            fn = row.getValue(refLabel)
            clasDist = row.getValue('rlnClassDistribution')

            if self._getClasDistCond(clasDist):
                mapId = self._mapsDict[fn]
                newMap = self._getMapById(mapId)
                row.setValue(refLabel, newMap)
                row.writeToMd(modelMd, row.getObjId())
                if self.stopDict[mapId]:
                    row.addToMd(finalMd)
                else:
                    row.addToMd(outMd)

        if outMd.size() != 0:
            outMd.write(outModel)

        if finalMd.size() != 0:
            finalMd.write('model_classes@' + finalModel)

    def _mergeDataStar(self, rLev, callback):
        iters = self._lastIter(rLev)

        #metadata to save all particles that continues
        outData = self._getFileName('outputData', lev=self._level)
        outMd = self._getMetadata(outData)

        #metadata to save all final particles
        finalData = self._getFileName('rawFinalData')
        finalMd = self._getMetadata(finalData)
        imgStar = self._getFileName('data', iter=iters,
                                    lev=self._level, rLev=rLev)
        mdData = md.MetaData(imgStar)

        def _getMapId(rMap):
            try:
                return self._mapsDict[rMap]
            except:
                return None

        for row in md.iterRows(mdData, sortByLabel=md.RLN_PARTICLE_CLASS):
            clsPart = row.getValue(md.RLN_PARTICLE_CLASS)
            rMap = callback(iters, rLev, clsPart)
            mapId = _getMapId(rMap)

            while mapId is None:
                for clsPart in range(1, self.numberOfClasses.get()+1):
                    rMap = callback(iters, rLev, clsPart)
                    mapId = _getMapId(rMap)
                    if mapId is not None:
                        break

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

        try:
            avgVol = self._getFileName('avgMap', lev=self._level)
        except:
            avgVol = self._getPath('map_average.mrc')
        npAvgVol, dType = self._doAverageMaps(listVol)
        saveMrc(npAvgVol.astype(dType), avgVol)

    def _doAverageMaps(self, listVol):
        for vol in listVol:
            npVol = self._getVolNp(vol)
            std = 2 * npVol.std()
            npMask = 1 * (npVol > std)

            if vol == listVol[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)
            npAvgVol += npVol

        npAvgVol = npAvgVol / len(listVol)
        npAvgMask = 1 * (npAvgVol < 0.99)
        # npAvgVol *= npAvgMask
        return npAvgMask, dType

    def _getVolNp(self, vol):
        mapNp = loadMrc(vol, False)
        std = 2 * mapNp.std()
        npMask = 1 * (mapNp > std)
        mapNp = mapNp * npMask
        return mapNp

    def _getFunc(self, modelMd):
        resolution = []
        ssnr = []
        for row in md.iterRows(modelMd):
            resolution.append(row.getValue('rlnResolution'))
            ssnr.append(row.getValue('rlnSsnrMap'))
            if ssnr[-1] == 0.0:
                break

        f = interpolate.interp1d(ssnr, resolution)
        return f

    def _getPathMaps(self, filename="*.mrc"):
        filesPath = self._getLevelPath(self._level, filename)
        return sorted(glob(filesPath))

    def _alignVolumes(self):

        # Align all volumes inside a level
        Plugin.setEnviron()
        listVol = self._getPathMaps()
        avgVol = self._getFileName('avgMap', lev=self._level)
        npAvgVol = loadMrc(avgVol, writable=False)
        dType = npAvgVol.dtype

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

            saveMrc(npVol.astype(dType), vol)

    def _mrcToNp(self, volList, avgVol=None):
        """ Implemented in subclasses. """
        pass

    def _estimatePCA(self):
        listVol = self._getFinalMaps()

        volNp = loadMrc(listVol[0], False)
        dim = volNp.shape[0]
        dType = volNp.dtype

        matProj, newBaseAxis = self._doPCA(listVol)

        for i, volNewBaseList in enumerate(newBaseAxis):
            volBase = volNewBaseList.reshape((dim, dim, dim))
            nameVol = 'volume_base_%02d.mrc' % (i+1)
            saveMrc(volBase.astype(dType),
                    self._getLevelPath(self._level, nameVol))
        return matProj

    def _doPCA(self, listVol):
        npAvgVol, _ = self._doAverageMaps(listVol)

        listNpVol, _ = self._mrcToNp(listVol, avgVol=npAvgVol)

        covMatrix = np.cov(listNpVol)
        u, s, vh = np.linalg.svd(covMatrix)
        cuttOffMatrix = sum(s) * 1
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
        print("How many PCA eigenvalues: ", sCut)
        return matProj, s

    def _getFinalMaps(self):
        return [self._getMapById(k) for k in self._getFinalMapIds()]

    def _getFinalMapIds(self):
        return [k for k, v in self.stopDict.items() if v is True]

    def _clusteringData(self, matProj):
        method = self.classMethod.get()
        if method == 0:
            return self._doSklearnKmeans(matProj)
        else:
            return self._doSklearnAffProp(matProj)

    def _doSklearnKmeans(self, matProj):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=matProj.shape[1]).fit(matProj)
        return kmeans.labels_

    def _doSklearnAffProp(self, matProj):
        from sklearn.cluster import AffinityPropagation
        ap = AffinityPropagation(damping=0.5).fit(matProj)
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

    def _getRLevList(self):
        """ Return the list of iteration files, give the rLevTemplate. """
        self._createRLevTemplates()
        result = []
        files = sorted(glob(self._rLevTemplate))
        if files:
            for f in files:
                s = self._rLevRegex.search(f)
                if s:
                    result.append(int(s.group(1)))
        return result