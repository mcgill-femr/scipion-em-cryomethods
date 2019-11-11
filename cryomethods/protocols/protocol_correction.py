# **************************************************************************
# *
# * Authors:     Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
# *              Satinder Kaur (satinder.kaur@mail.mcgill.ca)
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
from glob import glob
from os.path import exists
import numpy as np
from scipy import stats

import pyworkflow as pw


class ProtocolMapCorrector(pw.em.EMProtocol):
    """ Descrption of the method
    """

    def __init__(self, **args):
        pw.em.EMProtocol.__init__(self, **args)

    # -------------------------- DEFINE param functions -----------------------
    def _defineInputParams(self, form):
        form.addParam('inputVolume', pw.params.PointerParam,
                      pointerClass='Volume',
                      important=True,
                      label='Input Volume',
                      help='Initial 3D map to correct')

        form.addParam('output', pw.params.PointerParam,
                      pointerClass='Volume',
                      important=True,
                      label='output',
                      help='Initial 3D map to correct')



    # -------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('anisotropicCorrectionStep')
        self._insertFunctionStep('sharpeningStep')
        self._insertFunctionStep('outputStep')


    # -------------------------- STEPS functions -------------------------------
    def convertInputStep(self, resetDeps, copyAlignment):
        """ Implemented in subclasses. """
        pass

    def anisotropicCorrectionStep(self, iter):
        self.inputvolume.get()
        pass


    def sharpeningStep(self, iter):
        output_one= self.anisotropicCorrectionStep(output)
        if output_one is None:
            self.inputvolume.get()
        pass

    def runClassifyStep(self, normalArgs, basicArgs, rLev):
        self._createIterTemplates(rLev)  # initialize files to know iterations
        self._setComputeArgs(normalArgs)
        params = self._getParams(normalArgs)
        self._runClassifyStep(params)

        for i in range(11, 75, 1):
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

    def createOutputStep(self):
        """ Implemented in subclasses. """
        pass

    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        self.validatePackageVersion('RELION_CRYOMETHODS_HOME', errors)

        if self._getInputParticles().isOddX():
            errors.append("Relion only works with even values for the "
                          "image dimensions!")

        errors += self._validateNormal()

        return errors

    def _validateNormal(self):
        """ Should be overwritten in subclasses to
        return summary message for NORMAL EXECUTION.
        """
        return []

    def _citations(self):
        cites = []
        return cites

    def _summary(self):
        summary = self._summaryNormal()
        return summary

    def _summaryNormal(self):
        """ Should be overwritten in subclasses to
        return summary message for NORMAL EXECUTION.
        """
        return []



    def _methods(self):
        """ Should be overwritten in each protocol.
        """
        return []

    # -------------------------- UTILS functions ------------------------------
    def _setNormalArgs(self, args):
        maskDiameter = self.maskDiameterA.get()
        pixelSize = self._getPixeSize()

        if maskDiameter <= 0:
            maskDiameter = pixelSize * self._getNewDim()

        self._defineInput(args)
        args.update({'--particle_diameter': maskDiameter,
                     '--angpix': pixelSize,
                     })
        self._setCTFArgs(args)

        if self.maskZero == MASK_FILL_ZERO:
            args['--zero_mask'] = ''

        args['--K'] = self.numOfVols.get() if self.IS_VOLSELECTOR \
                      else self.numberOfClasses.get()

        if self.limitResolEStep > 0:
            args['--strict_highres_exp'] = self.limitResolEStep.get()

        if self.IS_3D:
            if not self.isMapAbsoluteGreyScale:
                args['--firstiter_cc'] = ''
            args['--ini_high'] = self.initialLowPassFilterA.get()
            args['--sym'] = self.symmetryGroup.get()
            args['--pad'] = 1 if self.skipPadding else 2

        refArg = self._getRefArg()
        if refArg:
            args['--ref'] = refArg

        self._setBasicArgs(args)

    def _setCTFArgs(self, args):
        # CTF stuff
        if self.doCTF:
            args['--ctf'] = ''

        if self.hasReferenceCTFCorrected:
            args['--ctf_corrected_ref'] = ''

        if self._getInputParticles().isPhaseFlipped():
            args['--ctf_phase_flipped'] = ''

        if self.ignoreCTFUntilFirstPeak:
            args['--ctf_intact_first_peak'] = ''

    def _setSubsetArgs(self, args):
        if self._doSubsets():
            args['--write_subsets'] = 1
            args['--subset_size'] = self.subsetSize.get()
            args['--max_subsets'] = self.subsetUpdates.get()
            if self._useFastSubsets():
                args['--fast_subsets'] = ''

    def _setBasicArgs(self, args):
        """ Return a dictionary with basic arguments. """
        self._defineOutput(args)
        args.update({'--flatten_solvent': '',
                     '--norm': '',
                     '--scale': '',
                     '--oversampling': self.oversampling.get(),
                     '--tau2_fudge': self.regularisationParamT.get()
                     })
        args['--iter'] = 10

        if not self.IS_VOLSELECTOR:
            self._setSubsetArgs(args)

        self._setSamplingArgs(args)
        self._setMaskArgs(args)

    def _setSamplingArgs(self, args):
        """Should be overwritten in subclasses"""
        pass

    def _setMaskArgs(self, args):
        if self.IS_3D:
            if self.referenceMask.hasValue():
                mask = conv.convertMask(self.referenceMask.get(),
                                        self._getTmpPath())
                args['--solvent_mask'] = mask

            if self.solventMask.hasValue():
                solventMask = conv.convertMask(self.solventMask.get(),
                                               self._getTmpPath())
                args['--solvent_mask2'] = solventMask

            if (self.referenceMask.hasValue() and self.solventFscMask):
                args['--solvent_correct_fsc'] = ''

    def _getSamplingFactor(self):
        return 1 if self.oversampling == 0 else 2 * self.oversampling.get()

    def _setComputeArgs(self, args):
        args['--pool'] = self.pooledParticles.get()

        if not self.combineItersDisc:
            args['--dont_combine_weights_via_disc'] = ''

        if not self.useParallelDisk:
            args['--no_parallel_disc_io'] = ''

        if self.allParticlesRam:
            args['--preread_images'] = ''
        else:
            if self._getScratchDir():
                args['--scratch_dir'] = self._getScratchDir()

        if self.doGpu:
            args['--gpu'] = self.gpusToUse.get()
        args['--j'] = self.numberOfThreads.get()

    def _setContinueArgs(self, args, rLev):
        continueIter = self._lastIter(rLev)
        if self.IS_AUTOCLASSIFY:
            args['--continue'] = self._getFileName('optimiser', lev=self._level,
                                                   rLev=rLev, iter=continueIter)
        else:
            args['--continue'] = self._getFileName('optimiser', ruNum=rLev,
                                                   iter=continueIter)

    def _getParams(self, args):
        return ' '.join(['%s %s' % (k, str(v)) for k, v in args.iteritems()])

    def _getScratchDir(self):
        """ Returns the scratch dir value without spaces.
         If none, the empty string will be returned.
        """
        scratchDir = self.scratchDir.get() or ''
        return scratchDir.strip()

    def _getProgram(self, program='relion_refine'):
        """ Get the program name depending on the MPI use or not. """
        if self.numberOfMpi > 1:
            program += '_mpi'
        return program

    def _getInputParticles(self):
        return self.inputParticles.get()

    def _getIterNumber(self, index):
        """ Return the list of iteration files, give the iterTemplate. """
        result = None
        files = sorted(glob(self._iterTemplate))
        if files:
            f = files[index]
            s = self._iterRegex.search(f)
            if s:
                result = long(s.group(1))  # group 1 is 3 digits iteration
                # number
        return result

    def _lastIter(self, rLev=None):
        self._createIterTemplates(rLev)
        return self._getIterNumber(-1)

    def _firstIter(self):
        return self._getIterNumber(0) or 1

    def _splitInCTFGroups(self, imgStar):
        """ Add a new column in the image star to separate the particles
        into ctf groups """
        conv.splitInCTFGroups(imgStar,
                              self.defocusRange.get(),
                              self.numParticles.get())

    def _getnumberOfIters(self):
        return self.numberOfIterations.get()

    def _fillVolSetFromIter(self, volSet, it):
        volSet.setSamplingRate(self._getInputParticles().getSamplingRate())
        modelStar = md.MetaData('model_classes@' +
                                self._getFileName('model', iter=it))
        for row in md.iterRows(modelStar):
            fn = row.getValue('rlnReferenceImage')
            fnMrc = fn + ":mrc"
            itemId = self._getClassId(fn)
            classDistrib = row.getValue('rlnClassDistribution')
            accurracyRot = row.getValue('rlnAccuracyRotations')
            accurracyTras = row.getValue('rlnAccuracyTranslations')
            resol = row.getValue('rlnEstimatedResolution')

            if classDistrib > 0:
                vol = em.Volume()
                self._invertScaleVol(fnMrc)
                vol.setFileName(self._getOutputVolFn(fnMrc))
                vol.setObjId(itemId)
                vol._rlnClassDistribution = em.Float(classDistrib)
                vol._rlnAccuracyRotations = em.Float(accurracyRot)
                vol._rlnAccuracyTranslations = em.Float(accurracyTras)
                vol._rlnEstimatedResolution = em.Float(resol)
                volSet.append(vol)

    def _getRefArg(self):
        """ Return the filename that will be used for the --ref argument.
        The value will depend if in 2D and 3D or if input references will
        be used.
        It will return None if no --ref should be used. """
        if self.IS_3D:
            inputObj = self.inputVolumes.get()
            if isinstance(inputObj, em.SetOfVolumes):
                # input SetOfVolumes as references
                return self._getRefStar()
        return None  # No --ref should be used at this point

    def _convertVolFn(self, inputVol):
        """ Return a new name if the inputFn is not .mrc """
        index, fn = inputVol.getLocation()
        return self._getTmpPath(replaceBaseExt(fn, '%02d.mrc' % index))

    def _convertVol(self, ih, inputVol):
        outputFn = self._convertVolFn(inputVol)

        if outputFn:
            xdim = self._getNewDim()
            img = ih.read(inputVol)
            img.scale(xdim, xdim, xdim)
            img.write(outputFn)

        return outputFn

    def _getRefStar(self):
        return self._getTmpPath("input_references.star")

    def _convertRef(self):

        ih = em.ImageHandler()
        inputObj = self.inputVolumes.get()
        row = md.Row()
        refMd = md.MetaData()
        for vol in inputObj:
            newVolFn = self._convertVol(ih, vol)
            row.setValue(md.RLN_MLMODEL_REF_IMAGE, newVolFn)
            row.addToMd(refMd)
        refMd.write(self._getRefStar())

    def _getNewDim(self):
        tgResol = self.getAttributeValue('targetResol', 0)
        partSet = self._getInputParticles()
        size = partSet.getXDim()
        nyquist = 2 * partSet.getSamplingRate()

        if tgResol > nyquist:
            newSize = long(round(size * nyquist / tgResol))
            if newSize % 2 == 1:
                newSize += 1
            return newSize
        else:
            return size

    def _getPixeSize(self):
        partSet = self._getInputParticles()
        oldSize = partSet.getXDim()
        newSize  = self._getNewDim()
        pxSize = partSet.getSamplingRate() * oldSize / newSize
        return pxSize

    def _scaleImages(self,indx, img):
        fn = img.getFileName()
        index = img.getIndex()
        newFn = self._getTmpPath('particles_subset.mrcs')
        xdim = self._getNewDim()

        ih = em.ImageHandler()
        image = ih.read((index, fn))
        image.scale(xdim, xdim)

        image.write((indx, newFn))

        img.setFileName(newFn)
        img.setIndex(indx)
        img.setSamplingRate(self._getPixeSize())

    def _convertInput(self, imgSet):
        newDim = self._getNewDim()
        bg = newDim / 2

        args = '--operate_on %s --operate_out %s --norm --bg_radius %d'

        params = args % (self._getFileName('input_star'),
                         self._getFileName('preprocess_parts_star'), bg)
        self.runJob(self._getProgram(program='relion_preprocess'), params)

        from pyworkflow.utils import moveFile

        moveFile(self._getFileName('preprocess_parts'),
                 self._getTmpPath('particles_subset.mrcs'))

    def _stopRunCondition(self, rLev, iter):
        x = np.array([])
        y = np.array([])

        for i in range(iter-5, iter, 1):
            x = np.append(x, i)
            if self.IS_AUTOCLASSIFY:
                modelFn = self._getFileName('model', iter=i,
                                            lev=self._level, rLev=rLev)
            else:
                modelFn = self._getFileName('model', iter=i, ruNum=rLev)

            modelMd = md.RowMetaData('model_general@' + modelFn)
            y = np.append(y, modelMd.getValue(md.RLN_MLMODEL_AVE_PMAX))

        slope, _, _, _, _ = stats.linregress(x, y)
        return True if slope <= 0.001 else False

    def _invertScaleVol(self, fn):
        xdim = self._getInputParticles().getXDim()
        outputFn = self._getOutputVolFn(fn)
        ih = em.ImageHandler()
        img = ih.read(fn)
        img.scale(xdim, xdim, xdim)
        img.write(outputFn)

    def _getOutputVolFn(self, fn):
        return replaceExt(fn, '_origSize.mrc')

    def _postprocessImageRow(self, img, imgRow):
        partId = img.getParticleId()
        imgRow.setValue(md.RLN_PARTICLE_ID, long(partId))
        imgRow.setValue(md.RLN_MICROGRAPH_NAME,
                        "%06d@fake_movie_%06d.mrcs"
                        % (img.getFrameId(), img.getMicId()))

    def _postprocessParticleRow(self, part, partRow):
        if part.hasAttribute('_rlnGroupName'):
            partRow.setValue(md.RLN_MLMODEL_GROUP_NAME,
                             '%s' % part.getAttributeValue('_rlnGroupName'))
        else:
            partRow.setValue(md.RLN_MLMODEL_GROUP_NAME,
                             '%s' % part.getMicId())
        ctf = part.getCTF()
        if ctf is not None and ctf.getPhaseShift():
            partRow.setValue(md.RLN_CTF_PHASESHIFT, ctf.getPhaseShift())

    def _getResetDeps(self):
        """Should be overwritten in subclasses"""
        pass

    def _doSubsets(self):
        # Since 'doSubsets' property is only valid for 2.1+ protocols
        # we need provide a default value for backward compatibility
        return self.getAttributeValue('doSubsets', False)

    def _copyAlignAsPriors(self, imgStar, alignType):
        mdParts = md.MetaData(imgStar)

        # set priors equal to orig. values
        mdParts.copyColumn(md.RLN_ORIENT_ORIGIN_X_PRIOR, md.RLN_ORIENT_ORIGIN_X)
        mdParts.copyColumn(md.RLN_ORIENT_ORIGIN_Y_PRIOR, md.RLN_ORIENT_ORIGIN_Y)
        mdParts.copyColumn(md.RLN_ORIENT_PSI_PRIOR, md.RLN_ORIENT_PSI)
        if alignType == em.ALIGN_PROJ:
            mdParts.copyColumn(md.RLN_ORIENT_ROT_PRIOR, md.RLN_ORIENT_ROT)
            mdParts.copyColumn(md.RLN_ORIENT_TILT_PRIOR, md.RLN_ORIENT_TILT)

        mdParts.write(imgStar)

    def _defineInput(self, args):
        args['--i'] = self._getFileName('input_star')

    def _defineOutput(self, args):
        args['--o'] = self._getExtraPath('relion')