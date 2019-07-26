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
import copy
import random
import numpy as np
from glob import glob
from collections import Counter

import pyworkflow.em as em
import pyworkflow.em.metadata as md
from pyworkflow.em.convert import ImageHandler
import pyworkflow.protocol.constants as cons
import pyworkflow.protocol.params as params
from pyworkflow.utils import (makePath, copyFile, replaceBaseExt)

from cryomethods import Plugin
from cryomethods.convert import (writeSetOfParticles, rowToAlignment,
                                 relionToLocation, loadMrc, saveMrc,
                                 alignVolumes, applyTransforms)

from .protocol_auto_base import ProtAutoBase


class Prot2DAutoClassifier(ProtAutoBase):
    _label = '2D auto classifier'
    IS_2D = True
    IS_AUTOCLASSIFY = True

    def __init__(self, **args):
        ProtAutoBase.__init__(self, **args)

    def _createFilenameTemplates(self):
        """ Centralize how files are called for iterations and references. """
        self.levDir = self._getExtraPath('lev_%(lev)02d/')
        self.rLevDir = self._getExtraPath('lev_%(lev)02d/rLev_%(rLev)02d/')
        self.rLevIter = self.rLevDir + 'relion_it%(iter)03d_'
        # add to keys, data.star, optimiser.star and sampling.star
        myDict = {
            'input_star': self.levDir + 'input_rLev-%(rLev)03d.star',
            'outputData': self.levDir + 'output_data.star',
            'image': self.levDir + 'image_id-%(id)s.mrc',
            'avgMap': self.levDir + 'map_average.mrc',
            'relionImage': '%(clsImg)06d@' + self.rLevDir +
                           'relion_it%(iter)03d_classes.mrcs',
            'outputModel': self.levDir + 'output_model.star',
            'data': self.rLevIter + 'data.star',
            'rawFinalModel': self._getExtraPath('raw_final_model.star'),
            'rawFinalData': self._getExtraPath('raw_final_data.star'),
            'finalModel': self._getExtraPath('final_model.star'),
            'finalData': self._getExtraPath('final_data.star'),
            'finalAvgMap': self._getExtraPath('map_average.mrc'),
            'optimiser': self.rLevIter + 'optimiser.star',
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

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        self._defineInputParams(form)
        self._defineCTFParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineOptimizationParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineSamplingParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineAdditionalParams(form)

    # -------------------------- INSERT steps functions ------------------------
    def _insertLevelSteps(self):
        deps = []
        levelRuns = self._getLevRuns(self._level)

        self._insertFunctionStep('convertInputStep',
                                 self._getResetDeps(),
                                 self.copyAlignment,
                                 prerequisites=deps)

        for rLev in levelRuns:
            clsDepList = []
            self._rLev = rLev  # Just to generate the proper input star file.
            classDep = self._insertClassifyStep()
            clsDepList.append(classDep)
            self._setNewEvalIds()

        evStep = self._insertEvaluationStep(clsDepList)
        deps.append(evStep)
        return deps

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
            alignType = imgSet.getAlignment() if copyAlignment else em.ALIGN_NONE

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

    def evaluationStep(self):
        Plugin.setEnviron()
        print('Starting evaluation step')
        print('which level: ', self._level)
        self._copyLevelMaps()
        self._evalStop()
        self._mergeMetaDatas()
        # self._getAverageVol()
        # self._alignVolumes()
        print('Finishing evaluation step')

    def createOutputStep(self):
        partSet = self.inputParticles.get()

        classes2D = self._createSetOfClasses2D(partSet)
        pass
        # self._fillClassesFromIter(classes3D)
        #
        # self._defineOutputs(outputClasses=classes3D)
        # self._defineSourceRelation(self.inputParticles, classes3D)
        #
        # # create a SetOfVolumes and define its relations
        # volumes = self._createSetOfVolumes()
        # volumes.setSamplingRate(partSet.getSamplingRate())
        #
        # for class3D in classes3D:
        #     vol = class3D.getRepresentative()
        #     vol.setObjId(class3D.getObjId())
        #     volumes.append(vol)
        #
        # self._defineOutputs(outputVolumes=volumes)
        # self._defineSourceRelation(self.inputParticles, volumes)
        #
        # self._defineSourceRelation(self.inputVolumes, classes3D)
        # self._defineSourceRelation(self.inputVolumes, volumes)

    # -------------------------- UTILS functions -------------------------------
    def _setSamplingArgs(self, args):
        """ Set sampling related params. """
        # Sampling stuff
        if self.doImageAlignment:
            args['--offset_range'] = self.offsetSearchRangePix.get()
            args['--offset_step']  = (self.offsetSearchStepPix.get() *
                                      self._getSamplingFactor())
            args['--psi_step'] = (self.inplaneAngularSamplingDeg.get() *
                                  self._getSamplingFactor())
        else:
            args['--skip_align'] = ''

    def _getResetDeps(self):
        return "%s" % (self._getInputParticles().getObjId())

    def _getMapById(self, mapId):
        level = int(mapId.split('.')[0])
        return self._getFileName('image', lev=level, id=mapId)

    def _evalStop(self):
        noOfLevRuns = self._getLevRuns(self._level)
        print("dataModel's loop to evaluate stop condition")

        # x = [k for k, v in self.stopResLog.items()]
        # y = [v for k, v in self.stopResLog.items()]
        # f = np.polyfit(x, y, 2)
        # print ("polynomial values: ", f)
        # pol = np.poly1d(f)

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
                mapId = self._mapsDict[fn]
                suffixSsnr = 'model_class_%d@' % clsId
                # ssnrMd = md.MetaData(suffixSsnr + modelFn)
                # val = 1.05 * self._getArea(ssnrMd)
                classSize = row.getValue('rlnClassDistribution') * partSize
                # size = np.math.log10(classSize)
                # ExpcVal = pol(size)
                ptcStop = self.minPartsToStop.get()
                clsId += 1
                print("ValuesStop: parts %d" %classSize)

                if classSize < ptcStop:
                    self.stopDict[mapId] = True
                    if not bool(self._clsIdDict):
                        self._clsIdDict[mapId] = 1
                    else:
                        classId = sorted(self._clsIdDict.values())[-1] + 1
                        self._clsIdDict[mapId] = classId
                else:
                    self.stopDict[mapId] = False

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
            rMap = self._getFileName('relionImage', lev=self._level,
                                     iter=iters,
                                     clsImg=clsPart, rLev=rLev)
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

    def _mrcToNp(self, volList):
        listNpVol = []
        for vol in volList:
            volNp = loadMrc(vol, False)
            dim = volNp.shape[1]
            lenght = dim**2
            volList = volNp.reshape(lenght)
            listNpVol.append(volList)
        return listNpVol, listNpVol[0].dtype
