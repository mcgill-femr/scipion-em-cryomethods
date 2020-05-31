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

import pwem as em
import pwem.emlib.metadata as md
import pyworkflow.protocol.constants as cons
from pyworkflow.object import Float, String
from pyworkflow.utils import makePath

from cryomethods import Plugin
from cryomethods.convert import (writeSetOfParticles, rowToAlignment,
                                 loadMrc)

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
                    print("Path: %s and newRlev: %d" % (paths, clsPart))
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
        print('Finishing evaluation step')

    def createOutputStep(self):
        partSet = self.inputParticles.get()

        classes2D = self._createSetOfClasses2D(partSet)
        self._fillClassesFromIter(classes2D)

        self._defineOutputs(outputClasses=classes2D)
        self._defineSourceRelation(self.inputParticles, classes2D)

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

    def _getRelionFn(self, iters, rLev, clsPart):
        return self._getFileName('relionImage', lev=self._level,
                                 iter=iters, clsImg=clsPart, rLev=rLev)

    def _mrcToNp(self, volList, avgVol=None):
        listNpVol = []
        for vol in volList:
            volNp = loadMrc(vol, False)
            dim = volNp.shape[1]
            lenght = dim**2
            volList = volNp.reshape(lenght)
            listNpVol.append(volList)
        return listNpVol, listNpVol[0].dtype

    def _updateParticle(self, item, row):
        item.setClassId(row.getValue(md.RLN_PARTICLE_CLASS))
        item.setTransform(rowToAlignment(row, em.ALIGN_2D))

        item._rlnLogLikeliContribution = Float(
            row.getValue('rlnLogLikeliContribution'))
        item._rlnMaxValueProbDistribution = Float(
            row.getValue('rlnMaxValueProbDistribution'))
        item._rlnGroupName = String(row.getValue('rlnGroupName'))

