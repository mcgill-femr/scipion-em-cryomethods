# **************************************************************************
# *
# * Authors:     Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
# *              Javier Vargas Balbuena (javier.vargasbalbuena@mcgill.ca)
# *              Satinder kaur (satinder.kaur@mail.mcgill.ca)
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
import pyworkflow.em as em
import pyworkflow.object as pwObj
import pyworkflow.utils as pwUtils
import pyworkflow.em.metadata as md
from cryomethods.convert import writeSetOfParticles
from .protocol_base import ProtocolBase


class ProtInitialVolumeSelector(ProtocolBase):
    """
    Protocol to obtain a better initial volume using as input a set of
    volumes and particles. The protocol uses a small subset (usually 1000/2000)
    particles per classfrom the input set of particles to estimate a better and
    reliable volume(s) to use as initial volume in an automatic way.
    """

    _label = 'volume selector'

    IS_VOLSELECTOR = True

    def __init__(self, **args):
        ProtocolBase.__init__(self, **args)
        self.rLev = pwObj.Integer(0)

    def _createFilenameTemplates(self):
        """ Centralize how files are called for iterations and references. """
        self.extraRun = self._getExtraPath('run_%(ruNum)02d/relion_it%(iter)03d_')
        myDict = {
            'final_particles': self._getExtraPath('Finput_particles.star'),
            'input_star': self._getPath('input_particles_%(run)02d.star'),
            'data_scipion': self.extraRun + 'data_scipion.sqlite',
            'volumes_scipion': self.extraRun + 'volumes.sqlite',
            'volume': self.extraRun + "_class%(ref3d)03d.mrc",
            'data': self.extraRun + 'data.star',
            'model': self.extraRun + 'model.star',
            'selected_volumes': self._getTmpPath('selected_volumes_xmipp.xmd'),

        }
        # add to keys, data.star, optimiser.star and sampling.star
        for key in self.FILE_KEYS:
            myDict[key] = self.extraRun + '%s.star' % key
            key_xmipp = key + '_xmipp'
            myDict[key_xmipp] = self.extraRun + '%s.xmd' % key
        # add other keys that depends on prefixes
        for p in self.PREFIXES:
            myDict['%smodel' % p] = self.extraRun + '%smodel.star' % p
            myDict['%svolume' % p] = self.extraRun + p + \
                                     'class%(ref3d)03d.mrc:mrc'
        self._updateFilenamesDict(myDict)

    def _createIterTemplates(self, run=None):
        run = self.rLev.get() if run is None else run

        """ Setup the regex on how to find iterations. """
        self._iterTemplate = self._getFileName('data', ruNum=run,
                                               iter=0).replace('000', '???')
        # Iterations will be identify by _itXXX_ where XXX is the iteration
        # number and is restricted to only 3 digits.
        self._iterRegex = re.compile('_it(\d{3,3})_')
        self._classRegex = re.compile('_class(\d{3,3}).')

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        self._defineInputParams(form)
        self._defineReferenceParams(form)
        self._defineCTFParams(form)
        self._defineOptimizationParams(form)
        self._defineSamplingParams(form)
        self._defineAdditionalParams(form)

    # -------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self.volDict = {}
        totalVolumes = self.inputVolumes.get().getSize()
        selectedVols = self.numOfVols.get()

        b = np.log((1 - (float(selectedVols) / float(totalVolumes))))
        numOfRuns = 1 if selectedVols >= totalVolumes else int(-3 / b)

        for run in range(1, numOfRuns+1):
            self._createFilenameTemplates()
            self._createIterTemplates(run)
            self._createVolDict()
            self._rLev = run
            resetDeps = self._getResetDeps()
            self._insertFunctionStep('convertInputStep', resetDeps, run)
            self._insertClassifyStep()
        self._insertFunctionStep('mergeVolumesStep', numOfRuns)
        self._rLev += 1
        self._insertFunctionStep('converParticlesStep', self._rLev)
        self._insertClassifyStep()
        self._insertFunctionStep('createOutputStep')

    # -------------------------- STEPS functions -------------------------------
    def convertInputStep(self, resetDeps, run):
        """ Create the input file in STAR format as expected by Relion.
        If the input particles comes from Relion, just link the file.
        Params:
            particlesId, volumesId: use this parameters just to force redo of
            convert if either the input particles and/or input volumes are
            changed.
        """
        self.converParticlesStep(run)
        self._convertRef()

    def converParticlesStep(self, run):
        self._setrLev(run)
        self._imgFnList = []
        imgSet = self._getInputParticles()
        imgStar = self._getFileName('input_star', run=run)
        os.makedirs(self._getExtraPath('run_%02d' % run))
        subset = em.SetOfParticles(filename=":memory:")

        newIndex = 1
        for img in imgSet.iterItems(orderBy='RANDOM()', direction='ASC'):
            self._scaleImages(newIndex, img)
            newIndex += 1
            subset.append(img)
            subsetSize = self.subsetSize.get() * self.numOfVols.get()
            minSize = min(subsetSize, imgSet.getSize())
            if subsetSize > 0 and subset.getSize() == minSize:
                break
        writeSetOfParticles(subset, imgStar, self._getExtraPath(),
                            alignType=em.ALIGN_NONE,
                            postprocessImageRow=self._postprocessParticleRow)
        if self.doCtfManualGroups:
            self._splitInCTFGroups(imgStar)

    def mergeVolumesStep(self, numOfRuns):
        mdOut = md.MetaData()
        volFnList = []
        clsDistList = []
        accList = []
        for run in range(1, numOfRuns + 1):
            it = self._lastIter(run)
            modelFile = self._getFileName('model', ruNum=run, iter=it)
            mdIn = md.MetaData('model_classes@%s' % modelFile)
            for row in md.iterRows(mdIn, md.RLN_MLMODEL_REF_IMAGE):
                volFn = row.getValue(md.RLN_MLMODEL_REF_IMAGE)
                clsDist = row.getValue('rlnClassDistribution')
                accRot = row.getValue('rlnAccuracyRotations')
                if accRot <= 90:
                    volFnList.append(volFn)
                    clsDistList.append(clsDist)
                    accList.append(accRot)

        self.std = np.std(accList)
        score = self._estimateScore(accList, clsDistList)
        tupleList = [(fn, s) for fn, s in zip(volFnList, score)]
        nVols = self.numOfVols.get()

        sortList = sorted(tupleList, reverse=True, key=lambda x: x[1])[0:nVols]
        row = md.Row()
        for val in sortList:

            fn, score = val
            row.setValue(md.RLN_MLMODEL_REF_IMAGE, fn)
            row.addToMd(mdOut)

        mdOut.write(self._getRefStar())

    def createOutputStep(self):
        # create a SetOfVolumes and define its relations
        volumes = self._createSetOfVolumes()
        self._fillVolSetFromIter(volumes)

        self._defineOutputs(outputVolumes=volumes)
        self._defineSourceRelation(self.inputVolumes, volumes)

    # --------------------------- INFO functions -------------------------------
    def _validateNormal(self):
        """ Should be overwritten in subclasses to
        return summary message for NORMAL EXECUTION.
        """
        return []

    def _summaryNormal(self):
        """ Should be overwritten in subclasses to
        return summary message for NORMAL EXECUTION.
        """
        self._initialize()
        lastIter = self._lastIter()

        if lastIter is not None:
            iterMsg = 'run: %d\n' % self.rLev.get()
            iterMsg += 'Iteration %d' % lastIter
        else:
            iterMsg = 'No iteration finished yet.'
        summary = [iterMsg]

        if self._getInputParticles().isPhaseFlipped():
            flipMsg = "Your input images are ctf-phase flipped"
            summary.append(flipMsg)

        return summary

    def _methods(self):
        """ Should be overwritten in each protocol.
        """
        return []

    # -------------------------- UTILS functions ------------------------------
    def _setSamplingArgs(self, args):
        """ Set sampling related params. """
        args['--healpix_order'] = self.angularSamplingDeg.get()
        args['--offset_range'] = self.offsetSearchRangePix.get()
        args['--offset_step'] = (self.offsetSearchStepPix.get() *
                                 self._getSamplingFactor())

    def _getResetDeps(self):
        return "%s, %s, %s" % (self._getInputParticles().getObjId(),
                               self.inputVolumes.get().getObjId(),
                               self.targetResol.get())

    def _getRefStar(self):
        return self._getTmpPath("allVolumes.star")

    def _getClassId(self, volFile):
        result = None
        s = self._classRegex.search(volFile)
        if s:
            result = int(s.group(1)) # group 1 is 2 digits class number
        return self.volDict[result]

    def _defineInput(self, args):
        args['--i'] = self._getFileName('input_star', run=self._rLev)

    def _defineOutput(self, args):
        args['--o'] = self._getExtraPath('run_%02d/relion' % self._rLev)

    def _getIterVolumes(self, run, clean=True):
        """ Return a volumes .sqlite file for this iteration.
        If the file doesn't exists, it will be created by
        converting from this iteration data.star file.
        """
        it = self._lastIter()
        sqlteVols = self._getFileName('volumes_scipion', ruNum=run, iter=it)

        if clean:
            pwUtils.cleanPath(sqlteVols)

        if not os.path.exists(sqlteVols):
            volSet = self.OUTPUT_TYPE(filename=sqlteVols)
            self._fillVolSetFromIter(volSet, rLev=run)
            volSet.write()
            volSet.close()
        return sqlteVols

    def _fillVolSetFromIter(self, volSet, rLev=None, it=None):
        it = self._lastIter() if it is None else it
        rLev = self._rLev if rLev is None else rLev
        volSet.setSamplingRate(self._getInputParticles().getSamplingRate())
        modelFn = self._getFileName('model', ruNum=rLev, iter=it)
        print('modelFn: ', modelFn)
        modelStar = md.MetaData('model_classes@' + modelFn)
        idList = []
        volFnList = []
        clsDistList = []
        accRotList = []
        accTransList = []
        resoList = []

        for row in md.iterRows(modelStar):
            accurracyRot = row.getValue('rlnAccuracyRotations')
            if accurracyRot <= 90:
                fn = row.getValue('rlnReferenceImage')
                print('Volume: ', fn)
                fnMrc = fn + ":mrc"
                itemId = self._getClassId(fn)
                classDistrib = row.getValue('rlnClassDistribution')
                accurracyTras = row.getValue('rlnAccuracyTranslations')
                resol = row.getValue('rlnEstimatedResolution')

                idList.append(itemId)
                volFnList.append(fnMrc)
                clsDistList.append(classDistrib)
                accRotList.append(accurracyRot)
                accTransList.append(accurracyTras)
                resoList.append(resol)
        std = self.std if hasattr(self, 'std') else None
        score = self._estimateScore(accRotList, clsDistList, None, std)
        threshold = 1/float(self.numOfVols.get())
        print("score: ", score)

        for i, s in enumerate(score):
            vol = em.Volume()
            self._invertScaleVol(volFnList[i])
            vol.setFileName(self._getOutputVolFn(volFnList[i]))
            vol.setObjId(idList[i])
            if s <= threshold:
                vol._objEnabled = False
            vol._cmScore = em.Float(s)
            vol._rlnClassDistribution = em.Float(clsDistList[i])
            vol._rlnAccuracyRotations = em.Float(accRotList[i])
            vol._rlnAccuracyTranslations = em.Float(accTransList[i])
            vol._rlnEstimatedResolution = em.Float(resoList[i])
            volSet.append(vol)

    def _convertRef(self):
        ih = em.ImageHandler()
        inputObj = self.inputVolumes.get()
        subset = em.SetOfVolumes(filename=":memory:")
        refMd = md.MetaData()

        for vol in inputObj.iterItems(orderBy='RANDOM()'):
            subset.append(vol)
            subsetSize = self.numOfVols.get()
            minSize = min(subsetSize, inputObj.getSize())
            if subset.getSize() <= minSize:
                row = md.Row()
                newVolFn = self._convertVol(ih, vol)
                row.setValue(md.RLN_MLMODEL_REF_IMAGE, newVolFn)
                row.addToMd(refMd)
            else:
                break
        refMd.write(self._getRefStar())

    def _estimateScore(self, accList, distList, mean=None, std=None):
        mean = np.min(accList) if mean is None else mean
        std = np.std(accList) if std is None else std

        gList = [self._gaussian(x, mean, std) for x in accList]
        weigth = [g*d if not d > 0.5 else d*sum(gList)/len(gList)
                  for g,d in zip(gList, distList)]
        score = [s*len(gList)/sum(gList) for s in weigth]

        return score

    def _gaussian(self, x, mean, std, c=1):
        return (np.exp(-(((x - mean) ** 2) / (c * (std ** 2)))))

    def _setrLev(self, val=None):
        self.rLev.set(self._rLev) if val is None else self.rLev.set(val)
        self._store(self.rLev)
