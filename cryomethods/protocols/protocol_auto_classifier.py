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

import pyworkflow.em as em
import pyworkflow.em.metadata as md
import pyworkflow.protocol.constants as cons
from pyworkflow.utils import (makePath, copyFile)

from cryomethods import Plugin
from cryomethods.convert import writeSetOfParticles, loadMrc

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
            alignToPrior = hasAlign and self.alignmentAsPriors.get()
            fillRandomSubset = hasAlign and self.fillRandomSubset.get()

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

    def _doAverageMaps(self, listVol):
        for vol in listVol:
            npVol = loadMrc(vol, False)
            if vol == listVol[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)
            npAvgVol += npVol

        npAvgVol = np.divide(npAvgVol, len(listVol))
        return npAvgVol, dType

    def _getArea(self, modelMd):
        resolution = []
        ssnr = []
        for row in md.iterRows(modelMd):
            resolution.append(row.getValue('rlnResolution'))
            ssnr.append(row.getValue('rlnSsnrMap'))

        area = np.trapz(ssnr, resolution)
        return area

    def _mrcToNp(self, volList):
        listNpVol = []
        for vol in volList:
            volNp = loadMrc(vol, False)
            dim = volNp.shape[0]
            lenght = dim**3
            volList = volNp.reshape(lenght)
            listNpVol.append(volList)
        return listNpVol, listNpVol[0].dtype
