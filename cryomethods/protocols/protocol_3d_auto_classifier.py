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
import numpy as np

import pwem.emlib.metadata as md
import pwem.emlib as emlib
import pyworkflow.protocol.constants as cons
from emtable import Table
from pwem import ALIGN_NONE
from pwem.emlib.image import ImageHandler
from pyworkflow.utils import makePath

from cryomethods import Plugin
from cryomethods.convert import writeSetOfParticles

from .protocol_auto_base import ProtAutoBase


class Prot3DAutoClassifier(ProtAutoBase):
    _label = '3D auto classifier'
    IS_2D = False
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
                'mdataForClass': self._getExtraPath('final_data_class_%(id)s.star'),
        }
        for key in self.FILE_KEYS:
            myDict[key] = self.rLevIter + '%s.star' % key
            key_xmipp = key + '_xmipp'
            myDict[key_xmipp] = self.rLevDir + '%s.xmd' % key
        # add other keys that depends on prefixes
        for p in self.PREFIXES:
            myDict['%smodel' % p] = self.rLevIter + '%smodel.star' % p
            myDict['%svolume' % p] = self.rLevDir + p + 'class%(ref3d)03d.mrc:mrc'

        self._updateFilenamesDict(myDict)

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        self._defineInputParams(form)
        self._defineReferenceParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineCTFParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineOptimizationParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineSamplingParams(form, expertLev=cons.LEVEL_NORMAL)
        self._defineAdditionalParams(form)

    # -------------------------- STEPS functions -------------------------------
    def convertInputStep(self, resetDeps, copyAlignment):
        import random
        """ Create the input file in STAR format as expected by Relion.
        If the input particles comes from Relion, just link the file.
        """
        if self._level == 0:
            makePath(self._getRunPath(self._level, 1))
            imgStar = self._getFileName('input_star', lev=self._level, rLev=0)
            self._convertStar(copyAlignment, imgStar)
            opticsTable = Table(fileName=imgStar, tableName='optics')
            partsTable = Table(fileName=imgStar, tableName='particles')
            self._convertVol(ImageHandler(), self.inputVolumes.get())
            mdSize = partsTable.size()

            for i in range(9, 1, -1):
                makePath(self._getRunPath(self._level, i))
                mStar = self._getFileName('input_star', lev=self._level, rLev=i)
                size = 10000 * i if mdSize >= 100000 else int(mdSize * 0.1 * i)
                print("partsTable: ", size, i, mdSize)
                partsTable._rows = random.sample(partsTable._rows, k=size)
                self.writeStar(mStar, partsTable, opticsTable)

        elif self._level == 1:
            imgStar = self._getFileName('input_star', lev=self._level, rLev=1)
            makePath(self._getRunPath(self._level, 1))
            self._convertStar(copyAlignment, imgStar)

            # find a clever way to avoid volume conversion if its already done.
            self._convertVol(ImageHandler(), self.inputVolumes.get())
        else:
            lastCls = None
            prevStar = self._getFileName('outputData', lev=self._level - 1)
            firstStarFn = self._getFileName('input_star', lev=1, rLev=1)
            # mdData = md.MetaData(prevStar)
            opTable = Table(fileName=firstStarFn, tableName='optics')

            tableIn = Table(fileName=prevStar, tableName='particles')
            cols = [str(c) for c in tableIn.getColumnNames()]

            pTable = Table()
            for row in pTable.iterRows(prevStar, key="rlnClassNumber",
                                       tableName='particles'):
                clsPart = row.rlnClassNumber
                if clsPart != lastCls:
                    makePath(self._getRunPath(self._level, clsPart))

                    if lastCls is not None:
                        print("writing %s" % fn)
                        # mdInput.write(fn)
                        self.writeStar(fn, newPTable, opTable)
                    paths = self._getRunPath(self._level, clsPart)
                    makePath(paths)
                    print ("Path: %s and newRlev: %d" % (paths, clsPart))
                    lastCls = clsPart
                    newPTable = Table(columns=cols, tableName='particles')
                    fn = self._getFileName('input_star', lev=self._level,
                                           rLev=clsPart)
                # objId = mdInput.addObject()
                newPTable.addRow(*row)
                # row.writeToMd(mdInput, objId)
            print("writing %s and ending the loop" % fn)
            self.writeStar(fn, newPTable, opTable)
            # mdInput.write(fn)

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

    def _getMapById(self, mapId):
        level = int(mapId.split('.')[0])
        return self._getFileName('map', lev=level, id=mapId)

    def _getRelionFn(self, iters, rLev, clsPart):
        return self._getFileName('relionMap', lev=self._level,
                                     iter=iters, ref3d=clsPart, rLev=rLev)

    def _getArea(self, modelMd):
        resolution = []
        ssnr = []
        for row in md.iterRows(modelMd):
            resolution.append(row.getValue('rlnResolution'))
            ssnr.append(row.getValue('rlnSsnrMap'))

        area = np.trapz(ssnr, resolution)
        return area

    def _convertStar(self, copyAlignment, imgStar):
        imgSet = self._getInputParticles()
        self.info("Converting set from '%s' into '%s'" %
                  (imgSet.getFileName(), imgStar))

        # Pass stack file as None to avoid write the images files
        # If copyAlignment is set to False pass alignType to ALIGN_NONE
        alignType = imgSet.getAlignment() if copyAlignment else ALIGN_NONE

        hasAlign = alignType != ALIGN_NONE
        alignToPrior = hasAlign and self.alignmentAsPriors.get()
        fillRandomSubset = hasAlign and self.fillRandomSubset.get()

        writeSetOfParticles(imgSet, imgStar,
                            outputDir=self._getExtraPath(),
                            alignType=alignType,
                            postprocessImageRow=self._postprocessParticleRow,
                            fillRandomSubset=fillRandomSubset)
        if alignToPrior:
            self._copyAlignAsPriors(imgStar, alignType)

        if self.doCtfManualGroups:
            self._splitInCTFGroups(imgStar)
