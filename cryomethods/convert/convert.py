# # **************************************************************************
# # *
# # * Authors:     Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
# # *              Javier Vargas Balbuena (javier.vargasbalbuena@mcgill.ca)
# # *
# # * Department of Anatomy and Cell Biology, McGill University
# # *
# # * This program is free software; you can redistribute it and/or modify
# # * it under the terms of the GNU General Public License as published by
# # * the Free Software Foundation; either version 2 of the License, or
# # * (at your option) any later version.
# # *
# # * This program is distributed in the hope that it will be useful,
# # * but WITHOUT ANY WARRANTY; without even the implied warranty of
# # * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # * GNU General Public License for more details.
# # *
# # * You should have received a copy of the GNU General Public License
# # * along with this program; if not, write to the Free Software
# # * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# # * 02111-1307  USA
# # *
# # *  All comments concerning this program package may be sent to the
# # *  e-mail address 'scipion@cnb.csic.es'
# # *
# # **************************************************************************
#
# import os
from os.path import join, basename
# import numpy
# from collections import OrderedDict
#
# from pwem import NO_INDEX, ALIGN_2D, ALIGN_3D, ALIGN_PROJ, ALIGN_NONE
from pwem.emlib.image import ImageHandler
# from pwem.objects import Acquisition, CTFModel, Transform, Coordinate, Particle, \
#     SetOfVolumes, SetOfClasses3D, SetOfClasses2D, SetOfAverages
# from pyworkflow.utils import getExt, commonPath
#
# try:
#     from itertools import izip
# except ImportError:
#     izip = zip
#
import pyworkflow as pw
# from pyworkflow.object import ObjectWrap, String, Integer
#
# import pwem.emlib.metadata as md
#
#
# def loadMrc(fn, writable=True):
#     """Return a NumPy array memory mapped from an existing MRC file.
#
#     The returned NumPy array will have an attribute, Mrc, that is
#     an instance of the Mrc class.  You can use that to access the
#     header or extended header of the file.  For instance, if x
#     was returned by bindFile(), x.Mrc.hdr.Num is the number of x
#     samples, number of y samples, and number of sections from the
#     file's header.
#
#     Positional parameters:
#     fn -- Is the name of the MRC file to bind.
#
#     Keyword parameters:
#     writable -- If True, the returned array will allow the elements of
#     the array to be modified.  The Mrc instance packaged with the
#     array will also allow modification of the header and extended
#     header entries.  Changes made to the array elements or the header
#     will affect the file to which the array is bound.
#     """
#     import mrc
#
#     mode = 'r'
#     if writable:
#         mode = 'r+'
#     a = mrc.Mrc(fn, mode)
#
#     return a.data_withMrc(fn)
#
#
# def saveMrc(npVol, fn):
#     import mrc
#     mrc.save(npVol, fn, ifExists='overwrite')
#
#
# def alignVolumes(volToAlign, VolRef):
#     import frm
#
#     axis, shifts, angles, score = frm.frm_align(VolRef, None, volToAlign,
#                                                 None, None, 20)
#     return axis, shifts, angles, score
#
#
# def applyTransforms(volume, shifts, angles, axis=None):
#     import transform
#
#     npVol = transform.translate3d(volume, shifts[0], shifts[1], shifts[2])
#     volume = transform.rotate3d(npVol, angles[0], angles[1], angles[2],
#                                 center=axis)
#     return volume
#
#
# # This dictionary will be used to map
# # between CTFModel properties and Relion labels
# ACQUISITION_DICT = OrderedDict([
#     ("_amplitudeContrast", md.RLN_CTF_Q0),
#     ("_sphericalAberration", md.RLN_CTF_CS),
#     ("_voltage", md.RLN_CTF_VOLTAGE),
#     ("_magnification", md.RLN_CTF_MAGNIFICATION)
# ])
#
# COOR_DICT = OrderedDict([
#     ("_x", md.RLN_IMAGE_COORD_X),
#     ("_y", md.RLN_IMAGE_COORD_Y)
# ])
#
# COOR_EXTRA_LABELS = [
#     # Additional autopicking-related metadata
#     md.RLN_PARTICLE_AUTOPICK_FOM,
#     md.RLN_PARTICLE_CLASS,
#     md.RLN_ORIENT_PSI
# ]
#
# CTF_DICT = OrderedDict([
#     ("_defocusU", md.RLN_CTF_DEFOCUSU),
#     ("_defocusV", md.RLN_CTF_DEFOCUSV),
#     ("_defocusAngle", md.RLN_CTF_DEFOCUS_ANGLE)
# ])
#
# CTF_PSD_DICT = OrderedDict([
#     ("_psdFile", md.RLN_CTF_IMAGE)
# ])
#
# CTF_EXTRA_LABELS = [
#     md.RLN_CTF_FOM,
#     md.RLN_CTF_PHASESHIFT,
#     # In Relion the ctf also contains acquisition information
#     md.RLN_CTF_Q0,
#     md.RLN_CTF_CS,
#     md.RLN_CTF_VOLTAGE,
#     md.RLN_CTF_MAGNIFICATION,
#     md.RLN_CTF_DETECTOR_PIXEL_SIZE
# ]
#
# # Some extra labels
# IMAGE_EXTRA_LABELS = [
#     md.RLN_SELECT_PARTICLES_ZSCORE,
#     md.RLN_IMAGE_FRAME_NR
# ]
#
# # Extra labels for movie refinement & polishing
# MOVIE_EXTRA_LABELS = [
#     md.RLN_PARTICLE_NR_FRAMES,
#     md.RLN_PARTICLE_NR_FRAMES_AVG,
#     md.RLN_PARTICLE_MOVIE_RUNNING_AVG,
#     md.RLN_PARTICLE_ORI_NAME,
#     md.RLN_MLMODEL_GROUP_NAME,
#     # the following is required for polishing to work
#     md.RLN_PARTICLE_DLL,
#     md.RLN_PARTICLE_PMAX,
#     md.RLN_IMAGE_NORM_CORRECTION,
#     md.RLN_PARTICLE_NR_SIGNIFICANT_SAMPLES,
#     md.RLN_PARTICLE_RANDOM_SUBSET,
#     md.RLN_ORIENT_ORIGIN_X_PRIOR,
#     md.RLN_ORIENT_ORIGIN_Y_PRIOR,
#     md.RLN_ORIENT_PSI_PRIOR,
#     md.RLN_ORIENT_ROT_PRIOR,
#     md.RLN_ORIENT_TILT_PRIOR
# ]
#
# # ANGLES_DICT = OrderedDict([
# #        ("_angleY", md.RLN_ANGLE_Y),
# #        ("_angleY2", md.RLN_ANGLE_Y2),
# #        ("_angleTilt", md.RLN_ANGLE_TILT)
# #        ])
#
# ALIGNMENT_DICT = OrderedDict([
#     ("_rlnOriginX", md.RLN_ORIENT_ORIGIN_X),
#     ("_rlnOriginY", md.RLN_ORIENT_ORIGIN_Y),
#     ("_rlnOriginZ", md.RLN_ORIENT_ORIGIN_Z),
#     ("_rlnAngleRot", md.RLN_ORIENT_ROT),
#     ("_rlnAngleTilt", md.RLN_ORIENT_TILT),
#     ("_rlnAnglePsi", md.RLN_ORIENT_PSI),
# ])
#
#
# def locationToRelion(index, filename):
#     """ Convert an index and filename location
#     to a string with @ as expected in Relion.
#     """
#     if index != NO_INDEX:
#         return "%06d@%s" % (index, filename)
#
#     return filename
#
#
#
def getImageLocation(location):
     return ImageHandler.locationToXmipp(location)
#
#
# def relionToLocation(filename):
#     """ Return a location (index, filename) given
#     a Relion filename with the index@filename structure. """
#     if '@' in filename:
#         indexStr, fn = filename.split('@')
#         return int(indexStr), str(fn)
#     else:
#         return NO_INDEX, str(filename)
#
#
# def setRelionAttributes(obj, objRow, *labels):
#     """ Set an attribute to obj from a label that is not
#     basic ones. The new attribute will be named _rlnLabelName
#     and the datatype will be set correctly.
#     """
#     for label in labels:
#         setattr(obj, '_%s' % md.label2Str(label),
#                 objRow.getValueAsObject(label))
#
#
# def objectToRow(obj, row, attrDict, extraLabels=[]):
#     """ This function will convert an EMObject into a XmippMdRow.
#     Params:
#         obj: the EMObject instance (input)
#         row: the XmippMdRow instance (output)
#         attrDict: dictionary with the map between obj attributes(keys) and
#             row MDLabels in Xmipp (values).
#         extraLabels: a list with extra labels that could be included
#             as _xmipp_labelName
#     """
#     if obj.isEnabled():
#         enabled = True
#     else:
#         enabled = False
#     row.setValue(md.RLN_IMAGE_ENABLED, enabled)
#
#     for attr, label in attrDict.items():
#         if hasattr(obj, attr):
#             valueType = md.label2Python(label)
#             row.setValue(label, valueType(getattr(obj, attr).get()))
#
#     attrLabels = attrDict.values()
#
#     for label in extraLabels:
#         attrName = '_' + md.label2Str(label)
#         if label not in attrLabels and hasattr(obj, attrName):
#             value = obj.getAttributeValue(attrName)
#             row.setValue(label, value)
#
#
# def rowToObject(row, obj, attrDict, extraLabels=[]):
#     """ This function will convert from a XmippMdRow to an EMObject.
#     Params:
#         row: the XmippMdRow instance (input)
#         obj: the EMObject instance (output)
#         attrDict: dictionary with the map between obj attributes(keys) and
#             row MDLabels in Xmipp (values).
#         extraLabels: a list with extra labels that could be included
#             as properties with the label name such as: _rlnSomeThing
#     """
#     obj.setEnabled(row.getValue(md.RLN_IMAGE_ENABLED, 1) > 0)
#
#     for attr, label in attrDict.items():
#         value = row.getValue(label)
#         if not hasattr(obj, attr):
#             setattr(obj, attr, ObjectWrap(value))
#         else:
#             getattr(obj, attr).set(value)
#
#     attrLabels = attrDict.values()
#
#     for label in extraLabels:
#         if label not in attrLabels and row.hasLabel(label):
#             labelStr = md.label2Str(label)
#             setattr(obj, '_' + labelStr, row.getValueAsObject(label))
#
#
# def setObjId(obj, mdRow, label=md.RLN_IMAGE_ID):
#     obj.setObjId(mdRow.getValue(label, None))
#
#
# def setRowId(mdRow, obj, label=md.RLN_IMAGE_ID):
#     mdRow.setValue(label, int(obj.getObjId()))
#
#
# def acquisitionToRow(acquisition, ctfRow):
#     """ Set labels values from acquisition to md row. """
#     objectToRow(acquisition, ctfRow, ACQUISITION_DICT)
#
#
# def rowToAcquisition(acquisitionRow):
#     """ Create an acquisition from a row of a meta """
#     if acquisitionRow.containsAll(ACQUISITION_DICT):
#         acquisition = Acquisition()
#         rowToObject(acquisitionRow, acquisition, ACQUISITION_DICT)
#     else:
#         acquisition = None
#
#     return acquisition
#
#
# def setPsdFiles(ctfModel, ctfRow):
#     """ Set the PSD files of CTF estimation related
#     to this ctfModel. The values will be read from
#     the ctfRow if present.
#     """
#     for attr, label in CTF_PSD_DICT.items():
#         if ctfRow.containsLabel(label):
#             setattr(ctfModel, attr, String(ctfRow.getValue(label)))
#
#
# def ctfModelToRow(ctfModel, ctfRow):
#     """ Set labels values from ctfModel to md row. """
#     objectToRow(ctfModel, ctfRow, CTF_DICT, extraLabels=CTF_EXTRA_LABELS)
#
#
# def rowToCtfModel(ctfRow):
#     """ Create a CTFModel from a row of a meta """
#     if ctfRow.containsAll(CTF_DICT):
#         ctfModel = CTFModel()
#         rowToObject(ctfRow, ctfModel, CTF_DICT, extraLabels=CTF_EXTRA_LABELS)
#         ctfModel.standardize()
#         setPsdFiles(ctfModel, ctfRow)
#     else:
#         ctfModel = None
#
#     return ctfModel
#
#
# def geometryFromMatrix(matrix, inverseTransform):
#     from pwem.convert.transformations import (translation_from_matrix,
#                                                        euler_from_matrix)
#
#     if inverseTransform:
#         from numpy.linalg import inv
#         matrix = inv(matrix)
#         shifts = -translation_from_matrix(matrix)
#     else:
#         shifts = translation_from_matrix(matrix)
#     angles = -numpy.rad2deg(euler_from_matrix(matrix, axes='szyz'))
#     return shifts, angles
#
#
# def matrixFromGeometry(shifts, angles, inverseTransform):
#     """ Create the transformation matrix from a given
#     2D shifts in X and Y...and the 3 euler angles.
#     """
#     from pwem.convert.transformations import euler_matrix
#     from numpy import deg2rad
#     radAngles = -deg2rad(angles)
#     M = euler_matrix(radAngles[0], radAngles[1], radAngles[2], 'szyz')
#     if inverseTransform:
#         from numpy.linalg import inv
#         M[:3, 3] = -shifts[:3]
#         M = inv(M)
#     else:
#         M[:3, 3] = shifts[:3]
#
#     return M
#
#
# def alignmentToRow(alignment, alignmentRow, alignType):
#     """
#     is2D == True-> matrix is 2D (2D images alignment)
#             otherwise matrix is 3D (3D volume alignment or projection)
#     invTransform == True  -> for xmipp implies projection
#                           -> for xmipp implies alignment
#     """
#     is2D = alignType == ALIGN_2D
#     is3D = alignType == ALIGN_3D
#     inverseTransform = alignType == ALIGN_PROJ
#     matrix = alignment.getMatrix()
#     shifts, angles = geometryFromMatrix(matrix, inverseTransform)
#
#     alignmentRow.setValue(md.RLN_ORIENT_ORIGIN_X, shifts[0])
#     alignmentRow.setValue(md.RLN_ORIENT_ORIGIN_Y, shifts[1])
#     # Also set the priors
#     alignmentRow.setValue(md.RLN_ORIENT_ORIGIN_X_PRIOR, shifts[0])
#     alignmentRow.setValue(md.RLN_ORIENT_ORIGIN_Y_PRIOR, shifts[1])
#
#     if is2D:
#         angle = angles[0] + angles[2]
#         alignmentRow.setValue(md.RLN_ORIENT_PSI, -angle)
#         flip = bool(numpy.linalg.det(matrix[0:2, 0:2]) < 0)
#         if flip:
#             print("FLIP in 2D not implemented")
#     elif is3D:
#         raise Exception("3D alignment conversion for Relion not implemented. "
#                         "It seems the particles were generated with an "
#                         "incorrect alignment type. You may either re-launch "
#                         "the protocol that generates the particles "
#                         "with angles or set 'Consider previous alignment?' "
#                         "to No")
#     else:
#         alignmentRow.setValue(md.RLN_ORIENT_ORIGIN_Z, shifts[2])
#         alignmentRow.setValue(md.RLN_ORIENT_ROT, angles[0])
#         alignmentRow.setValue(md.RLN_ORIENT_TILT, angles[1])
#         alignmentRow.setValue(md.RLN_ORIENT_PSI, angles[2])
#         # Also set the priors
#         alignmentRow.setValue(md.RLN_ORIENT_ROT_PRIOR, angles[0])
#         alignmentRow.setValue(md.RLN_ORIENT_TILT_PRIOR, angles[1])
#         alignmentRow.setValue(md.RLN_ORIENT_PSI_PRIOR, angles[2])
#
#
# def rowToAlignment(alignmentRow, alignType):
#     """
#     is2D == True-> matrix is 2D (2D images alignment)
#             otherwise matrix is 3D (3D volume alignment or projection)
#     invTransform == True  -> for xmipp implies projection
#     """
#     if alignType == ALIGN_3D:
#         raise Exception("3D alignment conversion for Relion not implemented.")
#
#     is2D = alignType == ALIGN_2D
#     inverseTransform = alignType == ALIGN_PROJ
#     if alignmentRow.containsAny(ALIGNMENT_DICT):
#         alignment = Transform()
#         angles = numpy.zeros(3)
#         shifts = numpy.zeros(3)
#         shifts[0] = alignmentRow.getValue(md.RLN_ORIENT_ORIGIN_X, 0.)
#         shifts[1] = alignmentRow.getValue(md.RLN_ORIENT_ORIGIN_Y, 0.)
#         if not is2D:
#             angles[0] = alignmentRow.getValue(md.RLN_ORIENT_ROT, 0.)
#             angles[1] = alignmentRow.getValue(md.RLN_ORIENT_TILT, 0.)
#             angles[2] = alignmentRow.getValue(md.RLN_ORIENT_PSI, 0.)
#             shifts[2] = alignmentRow.getValue(md.RLN_ORIENT_ORIGIN_Z, 0.)
#         else:
#             angles[2] = - alignmentRow.getValue(md.RLN_ORIENT_PSI, 0.)
#         M = matrixFromGeometry(shifts, angles, inverseTransform)
#         alignment.setMatrix(M)
#     else:
#         alignment = None
#
#     return alignment
#
#
# def coordinateToRow(coord, coordRow, copyId=True):
#     """ Set labels values from Coordinate coord to md row. """
#     if copyId:
#         setRowId(coordRow, coord)
#     objectToRow(coord, coordRow, COOR_DICT, extraLabels=COOR_EXTRA_LABELS)
#     if coord.getMicName():
#         coordRow.setValue(md.RLN_MICROGRAPH_NAME, str(coord.getMicName()))
#     else:
#         if coord.getMicId():
#             coordRow.setValue(md.RLN_MICROGRAPH_NAME, str(coord.getMicId()))
#
#
# def rowToCoordinate(coordRow):
#     """ Create a Coordinate from a row of a meta """
#     # Check that all required labels are present in the row
#     if coordRow.containsAll(COOR_DICT):
#         coord = Coordinate()
#         rowToObject(coordRow, coord, COOR_DICT, extraLabels=COOR_EXTRA_LABELS)
#
#         micName = None
#
#         if coordRow.hasLabel(md.RLN_MICROGRAPH_ID):
#             micId = int(coordRow.getValue(md.RLN_MICROGRAPH_ID))
#             coord.setMicId(micId)
#             # If RLN_MICROGRAPH_NAME is not present, use the id as a name
#             micName = micId
#
#         if coordRow.hasLabel(md.RLN_MICROGRAPH_NAME):
#             micName = coordRow.getValue(md.RLN_MICROGRAPH_NAME)
#
#         coord.setMicName(micName)
#
#     else:
#         coord = None
#
#     return coord
#
#
# def imageToRow(img, imgRow, imgLabel=md.RLN_IMAGE_NAME, **kwargs):
#     # Provide a hook to be used if something is needed to be
#     # done for special cases before converting image to row
#     preprocessImageRow = kwargs.get('preprocessImageRow', None)
#     if preprocessImageRow:
#         preprocessImageRow(img, imgRow)
#
#     setRowId(imgRow, img)  # Set the id in the metadata as MDL_ITEM_ID
#     index, fn = img.getLocation()
#     # check if the is a file mapping
#     filesDict = kwargs.get('filesDict', {})
#     filename = filesDict.get(fn, fn)
#
#     imgRow.setValue(imgLabel, locationToRelion(index, filename))
#
#     if kwargs.get('writeCtf', True) and img.hasCTF():
#         ctfModelToRow(img.getCTF(), imgRow)
#
#     # alignment is mandatory at this point, it shoud be check
#     # and detected defaults if not passed at readSetOf.. level
#     alignType = kwargs.get('alignType')
#
#     if alignType != ALIGN_NONE and img.hasTransform():
#         alignmentToRow(img.getTransform(), imgRow, alignType)
#
#     if kwargs.get('writeAcquisition', True) and img.hasAcquisition():
#         acquisitionToRow(img.getAcquisition(), imgRow)
#
#     # Write all extra labels to the row
#     objectToRow(img, imgRow, {},
#                 extraLabels=IMAGE_EXTRA_LABELS + kwargs.get('extraLabels', []))
#
#     # Provide a hook to be used if something is needed to be
#     # done for special cases before converting image to row
#     postprocessImageRow = kwargs.get('postprocessImageRow', None)
#     if postprocessImageRow:
#         postprocessImageRow(img, imgRow)
#
#
# def particleToRow(part, partRow, **kwargs):
#     """ Set labels values from Particle to md row. """
#     coord = part.getCoordinate()
#     if coord is not None:
#         coordinateToRow(coord, partRow, copyId=False)
#     if part.hasMicId():
#         partRow.setValue(md.RLN_MICROGRAPH_ID, int(part.getMicId()))
#         # If the row does not contains the micrograph name
#         # use a fake micrograph name using id to relion
#         # could at least group for CTF using that
#         if not partRow.hasLabel(md.RLN_MICROGRAPH_NAME):
#             partRow.setValue(md.RLN_MICROGRAPH_NAME,
#                              'fake_micrograph_%06d.mrc' % part.getMicId())
#     if part.hasAttribute('_rlnParticleId'):
#         partRow.setValue(md.RLN_PARTICLE_ID, int(part._rlnParticleId.get()))
#     imageToRow(part, partRow, md.RLN_IMAGE_NAME, **kwargs)
#
#
# def rowToParticle(partRow, **kwargs):
#     """ Create a Particle from a row of a meta """
#     img = Particle()
#
#     # Provide a hook to be used if something is needed to be
#     # done for special cases before converting image to row
#     preprocessImageRow = kwargs.get('preprocessImageRow', None)
#     if preprocessImageRow:
#         preprocessImageRow(img, partRow)
#
#     # Decompose Relion filename
#     index, filename = relionToLocation(partRow.getValue(md.RLN_IMAGE_NAME))
#     img.setLocation(index, filename)
#
#     if partRow.containsLabel(md.RLN_PARTICLE_CLASS):
#         img.setClassId(partRow.getValue(md.RLN_PARTICLE_CLASS))
#
#     if kwargs.get('readCtf', True):
#         img.setCTF(rowToCtfModel(partRow))
#
#     # alignment is mandatory at this point, it shoud be check
#     # and detected defaults if not passed at readSetOf.. level
#     alignType = kwargs.get('alignType')
#
#     if alignType != ALIGN_NONE:
#         img.setTransform(rowToAlignment(partRow, alignType))
#
#     if kwargs.get('readAcquisition', True):
#         img.setAcquisition(rowToAcquisition(partRow))
#
#     if kwargs.get('magnification', None):
#         img.getAcquisition().setMagnification(kwargs.get("magnification"))
#
#     setObjId(img, partRow)
#     # Read some extra labels
#     rowToObject(partRow, img, {},
#                 extraLabels=IMAGE_EXTRA_LABELS + kwargs.get('extraLabels', []))
#
#     img.setCoordinate(rowToCoordinate(partRow))
#
#     # copy micId if available from row to particle
#     if partRow.hasLabel(md.RLN_MICROGRAPH_ID):
#         img.setMicId(partRow.getValue(md.RLN_MICROGRAPH_ID))
#
#     # copy particleId if available from row to particle
#     if partRow.hasLabel(md.RLN_PARTICLE_ID):
#         img._rlnParticleId = Integer(partRow.getValue(md.RLN_PARTICLE_ID))
#
#     # copy particleId if available from row to particle
#     if partRow.hasLabel(md.RLN_PARTICLE_RANDOM_SUBSET):
#         img._rln_halfId = Integer(
#             partRow.getValue(md.RLN_PARTICLE_RANDOM_SUBSET))
#
#     # Provide a hook to be used if something is needed to be
#     # done for special cases before converting image to row
#     postprocessImageRow = kwargs.get('postprocessImageRow', None)
#     if postprocessImageRow:
#         postprocessImageRow(img, partRow)
#     return img
#
#
# def readSetOfParticles(filename, partSet, **kwargs):
#     """read from Relion image meta
#         filename: The metadata filename where the image are.
#         imgSet: the SetOfParticles that will be populated.
#         rowToParticle: this function will be used to convert the row to Object
#     """
#     imgMd = md.MetaData(filename)
#     # By default remove disabled items from metadata
#     # be careful if you need to preserve the original number of items
#     if kwargs.get('removeDisabled', True):
#         imgMd.removeDisabled()
#
#     for imgRow in md.iterRows(imgMd):
#         img = rowToParticle(imgRow, **kwargs)
#         partSet.append(img)
#
#     partSet.setHasCTF(img.hasCTF())
#     partSet.setAlignment(kwargs['alignType'])
#
#
# def setOfImagesToMd(imgSet, imgMd, imgToFunc, **kwargs):
#     """ This function will fill Relion metadata from a SetOfMicrographs
#     Params:
#         imgSet: the set of images to be converted to metadata
#         md: metadata to be filled
#         rowFunc: this function can be used to setup the row before
#             adding to meta
#     """
#     if 'alignType' not in kwargs:
#         kwargs['alignType'] = imgSet.getAlignment()
#
#     for img in imgSet:
#         objId = imgMd.addObject()
#         imgRow = md.Row()
#         imgToFunc(img, imgRow, **kwargs)
#         imgRow.writeToMd(imgMd, objId)
#
#
# def writeSetOfParticles(imgSet, starFile,
#                         outputDir, **kwargs):
#     """ This function will write a SetOfImages as Relion meta
#     Params:
#         imgSet: the SetOfImages instance.
#         starFile: the filename where to write the meta
#         filesMapping: this dict will help when there is need to replace images names
#     """
#     filesDict = convertBinaryFiles(imgSet, outputDir)
#     kwargs['filesDict'] = filesDict
#     partMd = md.MetaData()
#     setOfImagesToMd(imgSet, partMd, particleToRow, **kwargs)
#
#     # Remove Magnification from metadata to avoid wrong values of pixel size.
#     # In Relion if Magnification and DetectorPixelSize are in metadata,
#     # pixel size is ignored in the command line.
#     partMd.removeLabel(md.RLN_CTF_MAGNIFICATION)
#
#     blockName = kwargs.get('blockName', 'Particles')
#     partMd.write('%s@%s' % (blockName, starFile))
#
#
# def writeReferences(inputSet, outputRoot, useBasename=False, **kwargs):
#     """
#     Write references star and stack files from SetOfAverages or SetOfClasses2D/3D.
#     Params:
#         inputSet: the input SetOfParticles to be converted
#         outputRoot: where to write the output files.
#         basename: If True, use the basename of the stack for setting path.
#     """
#     refsMd = md.MetaData()
#     stackFile = outputRoot + '.stk'
#     stackName = basename(stackFile) if useBasename else stackFile
#     starFile = outputRoot + '.star'
#     ih = ImageHandler()
#     row = md.Row()
#
#     def _convert(item, i, convertFunc):
#         index = i + 1
#         ih.convert(item, (index, stackFile))
#         item.setLocation(index, stackName)
#         convertFunc(item, row, **kwargs)
#         row.writeToMd(refsMd, refsMd.addObject())
#
#     if isinstance(inputSet, SetOfAverages):
#         for i, img in enumerate(inputSet):
#             _convert(img, i, particleToRow)
#
#     elif isinstance(inputSet, SetOfClasses2D):
#         for i, rep in enumerate(inputSet.iterRepresentatives()):
#             _convert(rep, i, particleToRow)
#
#     elif isinstance(inputSet, SetOfClasses3D):
#         for i, rep in enumerate(inputSet.iterRepresentatives()):
#             _convert(rep, i, imageToRow)
#
#     elif isinstance(inputSet, SetOfVolumes):
#         for i, vol in enumerate(inputSet):
#             _convert(vol, i, imageToRow)
#
#     else:
#         raise Exception('Invalid object type: %s' % type(inputSet))
#
#     refsMd.write(starFile)
#
#
# region Description
def splitInCTFGroups(imgStar, defocusRange=1000, numParticles=10):
    from emtable import Table
    """ Add a new colunm in the image star to separate the particles into ctf groups """
    partsTable = Table(fileName=imgStar, tableName='particles')
    opticsTable = Table(fileName=imgStar, tableName='optics')

    partsTable.sort('rlnDefocusU')

    defocusAll = partsTable.getColumnValues('rlnDefocusU')
    oldDefocusU = defocusAll[0]

    focusGroup = 1
    counter = 0

    groupName = '%s_%03d' % ('ctfgroup', focusGroup)

    for indx in range(0,len(defocusAll)):
        counter = counter + 1
        defocusU = defocusAll[indx]
        if counter < numParticles:
            pass
        else:
            if (defocusU - oldDefocusU) > defocusRange:
                focusGroup = focusGroup + 1
                oldDefocusU = defocusU
                groupName = '%s_%03d' % ('ctfgroup', focusGroup)
                counter = 0
        row = partsTable.__getitem__(indx)
        replaceDict = {'rlnGroupName': groupName}
        partsTable[indx] = row._replace(**replaceDict)

    with open(imgStar, 'w') as f:
        f.write("# Star file generated with Scipion\n")
        f.write("# version 30001\n")
        if opticsTable is not None:
            opticsTable.writeStar(f, tableName='optics')
            f.write("# version 30001\n")
        partsTable.writeStar(f, tableName='particles')

    #mdCount = md.MetaData()
    #mdCount.aggregate(mdAll, md.AGGR_COUNT, md.RLN_MLMODEL_GROUP_NAME,
    #                  md.RLN_MLMODEL_GROUP_NAME, md.MDL_COUNT)
    #print("number of particles per group: ", mdCount)
#
# endregion
#
# def prependToFileName(imgRow, prefixPath):
#     """ Prepend some root name to imageRow filename. """
#     index, imgPath = relionToLocation(imgRow.getValue(md.RLN_IMAGE_NAME))
#     newLoc = locationToRelion(index, os.path.join(prefixPath, imgPath))
#     imgRow.setValue(md.RLN_IMAGE_NAME, newLoc)
#
#
# def relativeFromFileName(imgRow, prefixPath):
#     """ Remove some prefix from filename in row. """
#     index, imgPath = relionToLocation(imgRow.getValue(md.RLN_IMAGE_NAME))
#     newImgPath = os.path.relpath(imgPath, prefixPath)
#     newLoc = locationToRelion(index, newImgPath)
#     imgRow.setValue(md.RLN_IMAGE_NAME, newLoc)
#
#
# def copyOrLinkFileName(imgRow, prefixDir, outputDir, copyFiles=False):
#     index, imgPath = relionToLocation(imgRow.getValue(md.RLN_IMAGE_NAME))
#     baseName = os.path.basename(imgPath)
#     newName = os.path.join(outputDir, baseName)
#     if not os.path.exists(newName):
#         if copyFiles:
#             pw.utils.copyFile(os.path.join(prefixDir, imgPath), newName)
#         else:
#             pw.utils.createLink(os.path.join(prefixDir, imgPath), newName)
#
#     imgRow.setValue(md.RLN_IMAGE_NAME, locationToRelion(index, newName))
#
#
# def setupCTF(imgRow, sampling):
#     """ Do some validations and set some values
#     for Relion import.
#     """
#     imgRow.setValue(md.MDL_SAMPLINGRATE, sampling)
#     # TODO: check if we want to move this behaviour to setup CTFModel by default
#     hasDefocusU = imgRow.containsLabel(md.MDL_CTF_DEFOCUSU)
#     hasDefocusV = imgRow.containsLabel(md.MDL_CTF_DEFOCUSV)
#     hasDefocusAngle = imgRow.containsLabel(md.MDL_CTF_DEFOCUS_ANGLE)
#
#     if hasDefocusU or hasDefocusV:
#         if not hasDefocusU:
#             imgRow.setValue(md.MDL_CTF_DEFOCUSU,
#                             imgRow.getValue(md.MDL_CTF_DEFOCUSV))
#         if not hasDefocusV:
#             imgRow.setValue(md.MDL_CTF_DEFOCUSV,
#                             imgRow.getValue(md.MDL_CTF_DEFOCUSU))
#         if not hasDefocusAngle:
#             imgRow.setValue(md.MDL_CTF_DEFOCUS_ANGLE, 0.)
#
#
# def convertBinaryFiles(imgSet, outputDir, extension='mrcs'):
#     """ Convert binary images files to a format read by Relion.
#     Params:
#         imgSet: input image set to be converted.
#         outputDir: where to put the converted file(s)
#     Return:
#         A dictionary with old-file as key and new-file as value
#         If empty, not conversion was done.
#     """
#     filesDict = {}
#     ih = ImageHandler()
#     outputRoot = os.path.join(outputDir, 'input')
#     # Get the extension without the dot
#     stackFiles = list(imgSet.getFiles())
#     ext = getExt(next(iter(stackFiles)))[1:]
#
#     print('-------------')
#     print(ext)
#     print(type(stackFiles))
#     print(stackFiles)
#     print('-------------')
#
#     rootDir = commonPath(stackFiles)
#
#     def getUniqueFileName(fn, extension):
#         """ Get an unique file for either link or convert files.
#         It is possible that the base name overlap if they come
#         from different runs. (like particles.mrcs after relion preprocess)
#         """
#         newFn = join(outputRoot, pw.utils.replaceBaseExt(fn, extension))
#         newRoot = pw.utils.removeExt(newFn)
#
#         values = filesDict.values()
#         counter = 1
#
#         while newFn in values:
#             counter += 1
#             newFn = '%s_%05d.%s' % (newRoot, counter, extension)
#
#         return newFn
#
#     def createBinaryLink(fn):
#         """ Just create a link named .mrcs to Relion understand
#         that it is a binary stack file and not a volume.
#         """
#         newFn = getUniqueFileName(fn, extension)
#         if not os.path.exists(newFn):
#             pw.utils.createLink(fn, newFn)
#             print("   %s -> %s" % (newFn, fn))
#         return newFn
#
#     def convertStack(fn):
#         """ Convert from a format that is not read by Relion
#         to an spider stack.
#         """
#         newFn = getUniqueFileName(fn, 'stk')
#         ih.convertStack(fn, newFn)
#         print("   %s -> %s" % (newFn, fn))
#         return newFn
#
#     def replaceRoot(fn):
#         """ Link create to the root folder, so just replace that
#         in the name, no need to do anything else.
#         """
#         return fn.replace(rootDir, outputRoot)
#
#     if ext == extension:
#         print("convertBinaryFiles: creating soft links.")
#         print("   Root: %s -> %s" % (outputRoot, rootDir))
#         mapFunc = replaceRoot
#         pw.utils.createLink(rootDir, outputRoot)
#     elif ext == 'mrc' and extension == 'mrcs':
#         print("convertBinaryFiles: creating soft links (mrcs -> mrc).")
#         mapFunc = createBinaryLink
#     elif ext.endswith('hdf'):  # assume eman .hdf format
#         print("convertBinaryFiles: converting stacks. (%s -> %s)"
#               % (extension, ext))
#         mapFunc = convertStack
#     else:
#         mapFunc = None
#
#     if mapFunc is not None:
#         pw.utils.makePath(outputRoot)
#         for fn in stackFiles:
#             newFn = mapFunc(fn)  # convert or link
#             filesDict[fn] = newFn  # map new filename
#
#     return filesDict
#
#
# def convertBinaryVol(vol, outputDir):
#     """ Convert binary volume to a format read by Relion.
#     Params:
#         vol: input volume object to be converted.
#         outputDir: where to put the converted file(s)
#     Return:
#         new file name of the volume (converted or not).
#     """
#
#     ih = ImageHandler()
#
#     # This approach can be extended when
#     # converting from a binary file format that
#     # is not read from Relion
#     def convertToMrc(fn):
#         """ Convert from a format that is not read by Relion
#         to mrc format.
#         """
#         newFn = join(outputDir, pw.utils.replaceBaseExt(fn, 'mrc'))
#         ih.convert(fn, newFn)
#         return newFn
#
#     ext = vol.getFileName()
#
#     if not ext.endswith('.mrc'):
#         fn = convertToMrc(vol.getFileName())
#     else:
#         fn = vol.getFileName()
#     return fn
#
#
def convertMask(img, outputDir):
# Convert binary mask to a format read by Relion and truncate the
#     values between 0-1 values, due to Relion only support masks with this
#     values (0-1).
#     Params:
#         img: input image to be converted.
#         outputDir: where to put the converted file(s)
#     Return:
#         new file name of the mask.

    ih = ImageHandler()
    imgFn = getImageLocation(img.getLocation())
    newFn = join(outputDir, pw.utils.replaceBaseExt(imgFn, 'mrc'))
    ih.truncateMask(imgFn, newFn)
    return newFn
#
#
# def createItemMatrix(item, row, align):
#     item.setTransform(rowToAlignment(row, alignType=align))
#
#
# def readSetOfCoordinates(coordSet, coordFiles, micList=None):
#     """ Read a set of coordinates from given coordinate files
#     associated to some SetOfMicrographs.
#     Params:
#         micSet and coordFiles should have same length and same order.
#         coordSet: empty SetOfCoordinates to be populated.
#     """
#     if micList is None:
#         micList = coordSet.getMicrographs()
#
#     for mic, coordFn in izip(micList, coordFiles):
#
#         if not os.path.exists(coordFn):
#             print("WARNING: Missing coordinates star file: ", coordFn)
#
#         try:
#             readCoordinates(mic, coordFn, coordSet)
#         except Exception:
#             print("WARNING: Error reading coordinates star file: ", coordFn)
#
#
# def readCoordinates(mic, fileName, coordsSet):
#     for row in md.iterRows(fileName):
#         coord = rowToCoordinate(row)
#         coord.setX(coord.getX())
#         coord.setY(coord.getY())
#         coord.setMicrograph(mic)
#         coordsSet.append(coord)
#
#
# def writeSetOfCoordinates(coordSet, outputDir):
#     pass
#
#
# def writeCoordinates(mic, fileName):
#     pass
#
#
# def openStar(fn, extraLabels=False):
#     # We are going to write metadata directy to file to do it faster
#     f = open(fn, 'w')
#     s = """
# data_
#
# loop_
# _rlnCoordinateX
# _rlnCoordinateY
# """
#     if extraLabels:
#         s += "_rlnClassNumber\n"
#         s += "_rlnAutopickFigureOfMerit\n"
#         s += "_rlnAnglePsi\n"
#     f.write(s)
#     return f
#
#
# def writeSetOfCoordinates(posDir, coordSet, getStarFileFunc, scale=1):
#     """ Convert a SetOfCoordinates to Relion star files.
#     Params:
#         posDir: the output directory where to generate the files.
#         coordSet: the input SetOfCoordinates that will be converted.
#          getStarFileFunc: function object that receives the micrograph name
#             and return the coordinates star file (only the base filename).
#         scale: pass a value if the coordinates have a different scale.
#             (for example when extracting from micrographs with a different
#             pixel size than during picking)
#     """
#
#     # Create a dictionary with the pos filenames for each micrograph
#     posDict = {}
#     for mic in coordSet.iterMicrographs():
#         starFile = getStarFileFunc(mic)
#         if starFile is not None:
#             posFn = os.path.basename(starFile)
#             posDict[mic.getObjId()] = join(posDir, posFn)
#
#     f = None
#     lastMicId = None
#
#     extraLabels = coordSet.getFirstItem().hasAttribute('_rlnClassNumber')
#     doScale = abs(scale - 1) > 0.001
#
#     for coord in coordSet.iterItems(orderBy='_micId'):
#         micId = coord.getMicId()
#
#         if micId != lastMicId:
#             if not micId in posDict:
#                 print("Warning: micId %s not found" % micId)
#                 continue
#             # we need to close previous opened file
#             if f:
#                 f.close()
#             f = openStar(posDict[micId], extraLabels)
#             lastMicId = micId
#
#         if doScale:
#             x = coord.getX() * scale
#             y = coord.getY() * scale
#         else:
#             x = coord.getX()
#             y = coord.getY()
#
#         if not extraLabels:
#             f.write("%d %d \n" % (x, y))
#         else:
#             f.write("%d %d %d %0.6f %0.6f\n"
#                     % (x, y,
#                        coord._rlnClassNumber,
#                        coord._rlnAutopickFigureOfMerit,
#                        coord._rlnAnglePsi))
#
#     if f:
#         f.close()
#
#     return posDict.values()
#
#
# def writeMicCoordinates(mic, coordList, outputFn, getPosFunc=None):
#     """ Write the pos file as expected by Xmipp with the coordinates
#     of a given micrograph.
#     Params:
#         mic: input micrograph.
#         coordList: list of (x, y) pairs of the mic coordinates.
#         outputFn: output filename for the pos file .
#         isManual: if the coordinates are 'Manual' or 'Supervised'
#         getPosFunc: a function to get the positions from the coordinate,
#             it can be useful for scaling the coordinates if needed.
#     """
#     if getPosFunc is None:
#         getPosFunc = lambda coord: coord.getPostion()
#
#     extraLabels = coordList[0].hasAttribute('_rlnClassNumber')
#     f = openStar(outputFn, extraLabels)
#
#     for coord in coordList:
#         x, y = getPosFunc(coord)
#         if not extraLabels:
#             f.write("%d %d \n" % (x, y))
#         else:
#             f.write("%d %d %d %0.6f %0.6f\n"
#                     % (x, y,
#                        coord._rlnClassNumber,
#                        coord._rlnAutopickFigureOfMerit,
#                        coord._rlnAnglePsi))
#
#     f.close()
#
#
