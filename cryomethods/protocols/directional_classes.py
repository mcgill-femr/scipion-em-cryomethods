# **************************************************************************
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
# **************************************************************************
"""
Fast directional classification of particles with a 3D angular assignment.

Equivalent in purpose to Xmipp's 'solid angles' protocol (with one
directional class per direction) but WITHOUT CL2D: particles are grouped
purely geometrically, by the angular distance between their projection
direction and a uniform gallery of directions. Each particle ends up in
exactly one class; when a particle is (almost) equidistant to several
directions it is assigned to the minority one.
"""

import json
import os

import mrcfile
import numpy as np

from pyworkflow.protocol.params import (PointerParam, FloatParam, IntParam,
                                        StringParam, BooleanParam,
                                        LEVEL_ADVANCED)
from pyworkflow.object import Float, Integer
from pwem.objects import Transform
from pwem.protocols import ProtAnalysis3D

from xmipp3 import Plugin as XmippPlugin
from xmipp3.convert import (geometryFromMatrix, matrixFromGeometry,
                            getImageLocation)

# A projection along -d is the projection along d mirrored in y (rows).
# This is the array axis to flip in the aligned images (checked with data).
MIRROR_FLIP_AXIS = 0


# ------------------------------------------------------------------------------
# Pure numpy helpers (no Scipion objects, so they can be unit tested)
# ------------------------------------------------------------------------------
def eulerToDirection(rot, tilt):
    """
    Projection direction (unit vector) of Xmipp Euler angles, in degrees.
    It is the third row of the Xmipp Euler matrix, and does not depend on psi.
    Returns an (N, 3) array.
    """
    rot = np.deg2rad(np.asarray(rot, dtype=np.float64))
    tilt = np.deg2rad(np.asarray(tilt, dtype=np.float64))
    return np.stack([np.cos(rot) * np.sin(tilt),
                     np.sin(rot) * np.sin(tilt),
                     np.cos(tilt)], axis=-1)


def directionToEuler(dirs):
    """ Inverse of eulerToDirection: returns (rot, tilt) in degrees. """
    dirs = np.atleast_2d(dirs)
    tilt = np.rad2deg(np.arccos(np.clip(dirs[:, 2], -1.0, 1.0)))
    rot = np.rad2deg(np.arctan2(dirs[:, 1], dirs[:, 0]))
    return rot, tilt


def eulerMatrix(rot, tilt, psi):
    """
    Xmipp Euler matrix (ZYZ, angles in degrees), as Euler_angles2matrix.
    Accepts scalars (returns (3, 3)) or arrays (returns (..., 3, 3)).
    """
    a, b, g = (np.deg2rad(np.asarray(x, dtype=np.float64))
               for x in (rot, tilt, psi))
    ca, sa, cb, sb, cg, sg = (np.cos(a), np.sin(a), np.cos(b), np.sin(b),
                              np.cos(g), np.sin(g))
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    rows = [[cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
            [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
            [sc, ss, cb]]
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def inPlaneAlignment(rot, tilt, psi, shiftX, shiftY, classRot, classTilt,
                     symMatrices, mirrors):
    """
    2D transformation that aligns each particle to the frame of its class.

    For each particle (Xmipp angles rot, tilt, psi and shifts) and its class
    direction g (classRot, classTilt), the symmetric copy E·R closest to g
    (or to -g if mirrors is True) is chosen. The frame of the class is
    F = Euler(g, psi=0), or F = Euler(rot_g + 180, 180 - tilt_g, 0) for
    mirrored particles, whose direction is -g. Then E·R ~ Rz(psiEff)·F.

    Xmipp convention (as in reconstruct_fourier): once its shifts are
    applied, an image is the projection of Euler(rot, tilt, psi), i.e. the
    projection of Euler(rot, tilt, 0) rotated by psi. Hence the transform that
    brings it to the projection of F, written as Xmipp alignment parameters
    for xmipp_transform_geometry --apply_transform, is
    psi' = -psiEff and shift' = R(psiEff)·shift (checked with
    xmipp_phantom_project images of known angles and shifts).

    Returns (psiApply, shiftXApply, shiftYApply, mirrored), angles in degrees.
    """
    E = eulerMatrix(rot, tilt, psi)                           # (n, 3, 3)
    ES = np.einsum('nij,kjl->nkil', E, symMatrices)           # (n, K, 3, 3)
    g = eulerToDirection(classRot, classTilt)                 # (n, 3)
    cos = np.einsum('nkj,nj->nk', ES[:, :, 2, :], g)
    best = np.argmax(np.abs(cos) if mirrors else cos, axis=1)
    rows = np.arange(E.shape[0])
    mirrored = cos[rows, best] < 0

    frameRot = np.where(mirrored, np.asarray(classRot) + 180.0, classRot)
    frameTilt = np.where(mirrored, 180.0 - np.asarray(classTilt), classTilt)
    F = eulerMatrix(frameRot, frameTilt, np.zeros_like(frameRot))
    M = np.einsum('nij,nkj->nik', ES[rows, best], F)          # E·R·F^T
    psiEff = np.arctan2(M[:, 0, 1], M[:, 0, 0])

    c, s = np.cos(psiEff), np.sin(psiEff)
    shiftXApply = c * shiftX - s * shiftY
    shiftYApply = s * shiftX + c * shiftY
    return -np.rad2deg(psiEff), shiftXApply, shiftYApply, mirrored


def fibonacciSphere(angularSampling):
    """
    Quasi-uniform directions over the whole sphere (Fibonacci lattice) with a
    mean spacing of about angularSampling degrees (hexagonal-cell estimate).
    Ordered from the north pole (z=1) to the south pole (z=-1).
    """
    s = np.deg2rad(angularSampling)
    n = max(2, int(round(4.0 * np.pi / (np.sqrt(3.0) / 2.0 * s ** 2))))
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(1.0 - z ** 2)
    phi = i * np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def symmetricCopies(dirs, symMatrices):
    """
    All symmetry-equivalent copies of each direction.
    For Xmipp Euler matrices E and symmetry matrices R, the symmetric
    orientation E·R has projection direction d·R (row vectors).
    Returns an (N, K, 3) array, K = number of symmetry matrices.
    """
    return np.einsum('nj,kjl->nkl', dirs, symMatrices)


def cosineToEquivalents(dirs, refs, mirrors):
    """
    Cosine of the angular distance between each direction and each reference,
    where refs is (G, K, 3) holding the K symmetric copies of each reference.
    The distance is the minimum over copies (maximum cosine). If mirrors is
    True, d and -d are considered the same direction (same central section in
    Fourier space), i.e. |cos| is used.
    Returns an (N, G) array.
    """
    cos = np.einsum('nj,gkj->ngk', dirs, refs)
    if mirrors:
        cos = np.abs(cos)
    return np.clip(cos.max(axis=2), -1.0, 1.0)


# Generic direction (not on any symmetry axis) whose Dirichlet domain is used
# as asymmetric unit
_REFERENCE_DIRECTION = np.array([0.3141, 0.2718, 0.9093]) / np.linalg.norm(
    [0.3141, 0.2718, 0.9093])


def _inAsymmetricUnit(dirs, symMatrices, mirrors):
    """
    Boolean mask of the directions lying in the asymmetric unit, defined as
    the Dirichlet domain of a generic reference direction: a direction belongs
    to it if, among all its symmetric (and mirror, if requested) copies, it is
    itself the closest one to the reference.
    """
    copies = symmetricCopies(dirs, symMatrices)  # (N, K, 3)
    if mirrors:
        copies = np.concatenate([copies, -copies], axis=1)
    return dirs @ _REFERENCE_DIRECTION >= (copies @ _REFERENCE_DIRECTION).max(axis=1) - 1e-9


def buildGallery(angularSampling, symMatrices, mirrors,
                 seamSeparationFactor=0.5, coverageFactor=0.7):
    """
    Uniform gallery of directions restricted to one asymmetric unit.
    1. Fibonacci directions lying in the asymmetric unit.
    2. Directions closer than seamSeparationFactor * angularSampling to an
       already kept one (up to symmetry) are removed: this only happens across
       the borders of the unit, where the lattice is cut.
    3. Coverage holes along those borders are filled by farthest-point
       insertion, until every direction of the sphere is closer than
       coverageFactor * angularSampling to the gallery (up to symmetry).
    Returns a (G, 3) array.
    """
    candidates = fibonacciSphere(angularSampling)
    gallery = candidates[_inAsymmetricUnit(candidates, symMatrices, mirrors)]

    refs = symmetricCopies(gallery, symMatrices)
    cosThr = np.cos(np.deg2rad(seamSeparationFactor * angularSampling))
    kept = []
    for i in range(gallery.shape[0]):
        if kept and cosineToEquivalents(gallery[i:i + 1], refs[kept],
                                        mirrors).max() > cosThr:
            continue
        kept.append(i)
    gallery = gallery[kept]

    dense = fibonacciSphere(angularSampling / 4.0)
    dense = dense[_inAsymmetricUnit(dense, symMatrices, mirrors)]
    best = cosineToEquivalents(dense, symmetricCopies(gallery, symMatrices),
                               mirrors).max(axis=1)
    cosCover = np.cos(np.deg2rad(coverageFactor * angularSampling))
    added = []
    while True:
        worst = np.argmin(best)
        if best[worst] >= cosCover:
            break
        newDir = dense[worst:worst + 1]
        added.append(newDir)
        best = np.maximum(best, cosineToEquivalents(
            dense, symmetricCopies(newDir, symMatrices), mirrors)[:, 0])
    return np.vstack([gallery] + added)


def _chunks(n, chunkSize):
    for start in range(0, n, chunkSize):
        yield start, min(n, start + chunkSize)


def assignToDirections(dirs, gallery, symMatrices, ambiguityTolerance, mirrors,
                       maxChunkElements=2 * 10 ** 7):
    """
    Exclusive assignment of particle directions to gallery directions.

    1. Each particle has a nearest gallery direction, at angle a_min.
       Its candidate classes are the directions at an angle <=
       a_min + ambiguityTolerance (the nearest one is always a candidate).
       A particle with more than one candidate is doubtful: it lies (almost)
       on the border between several directions.
    2. candidateSize[g] = number of particles having g as candidate (doubtful
       particles are counted in all their candidates).
    3. A particle with a single candidate goes to it. A doubtful particle goes
       to the MINORITY candidate, i.e. the one with the smallest
       candidateSize; ties are broken by the smallest gallery index, so the
       result is deterministic.

    Since only border particles are doubtful, every direction keeps the
    particles that are clearly closest to it: the minority rule rebalances
    borders but cannot empty a populated direction.

    Returns a dict with, per particle: 'assigned' (gallery index),
    'nearest' (gallery index), 'angleAssigned', 'angleNearest' (degrees),
    'numCandidates'; and per gallery direction: 'candidateSize'.
    """
    nPart = dirs.shape[0]
    nGal = gallery.shape[0]
    refs = symmetricCopies(gallery, symMatrices)  # (G, K, 3)
    chunkSize = max(1, maxChunkElements // (nGal * refs.shape[1]))
    rows = np.arange(min(chunkSize, nPart))

    def candidatesOf(start, end):
        cos = cosineToEquivalents(dirs[start:end], refs, mirrors)
        nearest = np.argmax(cos, axis=1)
        r = rows[:end - start]
        angleNearest = np.rad2deg(np.arccos(cos[r, nearest]))
        # Angle threshold per particle, expressed as a cosine threshold
        thr = np.cos(np.deg2rad(np.minimum(angleNearest + ambiguityTolerance,
                                           180.0)))
        cand = cos >= thr[:, None]
        cand[r, nearest] = True
        return cos, nearest, angleNearest, cand

    # First pass: candidate sizes and nearest direction
    candidateSize = np.zeros(nGal, dtype=np.int64)
    nearest = np.empty(nPart, dtype=np.int64)
    angleNearest = np.empty(nPart)
    numCandidates = np.empty(nPart, dtype=np.int64)
    for start, end in _chunks(nPart, chunkSize):
        _, near, angNear, cand = candidatesOf(start, end)
        candidateSize += cand.sum(axis=0)
        nearest[start:end] = near
        angleNearest[start:end] = angNear
        numCandidates[start:end] = cand.sum(axis=1)

    # Second pass: minority class among the candidates of each particle.
    # Sort key = (candidateSize, gallery index), encoded as a single integer.
    key = candidateSize * nGal + np.arange(nGal)
    sentinel = np.iinfo(np.int64).max
    assigned = np.empty(nPart, dtype=np.int64)
    angleAssigned = np.empty(nPart)
    for start, end in _chunks(nPart, chunkSize):
        cos, _, _, cand = candidatesOf(start, end)
        best = np.argmin(np.where(cand, key[None, :], sentinel), axis=1)
        assigned[start:end] = best
        angleAssigned[start:end] = np.rad2deg(
            np.arccos(cos[rows[:end - start], best]))

    return {'assigned': assigned,
            'nearest': nearest,
            'angleAssigned': angleAssigned,
            'angleNearest': angleNearest,
            'numCandidates': numCandidates,
            'candidateSize': candidateSize}


def getSymmetryMatrices(symmetryGroup):
    """
    Xmipp symmetry matrices R (identity included) as a (K, 3, 3) array, such
    that the symmetric orientations of an Euler matrix E are E·R.

    They are derived from xmippLib's SymList.symmetricAngles (R = E^T·E_sym for
    a generic orientation E), so the convention is exactly Xmipp's. The
    binding SymList.getSymmetryMatrices is not used: it can hang for some
    groups (e.g. 't') because it allocates symsNo()+1 matrices but only fills
    true_symNo of them.
    """
    if symmetryGroup.strip().lower() == 'c1':
        return np.eye(3)[None, :, :]

    from pwem import emlib
    symList = emlib.SymList()
    symList.readSymmetryFile(symmetryGroup.strip())

    def symmetricMatrices(angles):
        return [eulerMatrix(*a) for a in symList.symmetricAngles(*angles)]

    angles1, angles2 = (17.3, 41.9, 73.1), (-122.7, 108.4, 11.6)
    E1 = eulerMatrix(*angles1)
    mats = []
    for Es in symmetricMatrices(angles1):
        R = E1.T @ Es
        if not any(np.allclose(R, M, atol=1e-6) for M in mats):
            mats.append(R)

    # Check: the matrices must not depend on the orientation used to derive them
    E2 = eulerMatrix(*angles2)
    for Es in symmetricMatrices(angles2):
        if not any(np.allclose(Es, E2 @ R, atol=1e-5) for R in mats):
            raise ValueError('Symmetry group %s is not supported: its symmetric '
                             'orientations are not of the form E*R'
                             % symmetryGroup)
    return np.asarray(mats)


# ------------------------------------------------------------------------------
# Protocol
# ------------------------------------------------------------------------------
class ProtDirectionalClasses(ProtAnalysis3D):
    """
    Groups particles by projection direction using only their 3D angular
    assignment (no 2D classification, no CL2D), so it runs in seconds or
    minutes instead of hours.

    A uniform gallery of directions (asymmetric unit of the symmetry group)
    is built and each particle goes to its nearest direction. Classes are
    exclusive: a doubtful particle, i.e. one whose distance to several
    directions differs by less than the ambiguity tolerance, is assigned to
    the minority one (the direction with the fewest candidate particles).

    The output SetOfClasses2D can be used as 'Input 2D classes' of
    locPDF_classes_abs, in place of Xmipp's 'solid angles'.
    """
    _label = 'directional classes'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      label='Input particles',
                      help='Particles with a 3D angular assignment.')

        form.addParam('symmetryGroup', StringParam, default='c1',
                      label='Symmetry group',
                      help='Xmipp symmetry group (c1, cN, dN, t, o, i1...). '
                           'Symmetry-equivalent directions are merged into '
                           'the same class.')

        form.addParam('angularSampling', FloatParam, default=5.0,
                      label='Angular sampling (deg)',
                      help='Mean angular spacing between the directions of '
                           'the gallery. It controls the number of classes.')

        form.addParam('ambiguityTolerance', FloatParam, default=0.5,
                      label='Ambiguity tolerance (deg)',
                      help='A particle is doubtful when the angle to another '
                           'direction exceeds the angle to its nearest one by '
                           'less than this value. Doubtful particles are '
                           'assigned to the minority direction among their '
                           'candidates; the rest go to their nearest one.\n'
                           '0 = pure nearest-direction assignment. Small '
                           'values (0.5 deg with a 5 deg sampling) balance the '
                           'class borders. Values close to the angular '
                           'sampling make almost every particle doubtful, and '
                           'the minority rule then empties the most populated '
                           'directions (holes in the angular coverage).')

        form.addParam('checkMirrors', BooleanParam, default=True,
                      label='Consider mirrors?',
                      help='If Yes, the directions d and -d are considered the '
                           'same one: both produce the same central section '
                           'of the 3D Fourier transform, so they carry the same '
                           'information for the reconstruction (equivalent to '
                           '--check_mirrors in Xmipp solid angles).')

        form.addParam('maxImagesPerAverage', IntParam, default=100,
                      expertLevel=LEVEL_ADVANCED,
                      label='Max. images per class average',
                      help='Each class gets an average image (its '
                           'representative), computed by aligning its '
                           'particles in-plane with their angular assignment '
                           '(no 2D classification). Only up to this number of '
                           'particles per class, chosen at random, are used, '
                           'to keep it fast. The averages are only used to '
                           'display the classes; they do not affect which '
                           'particles belong to each class.')

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('readDirectionsStep')
        self._insertFunctionStep('assignStep')
        self._insertFunctionStep('computeAveragesStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions -----------------------------
    def readDirectionsStep(self):
        """ Read the angular assignment of every enabled particle. """
        ids, angles, shifts, locations = [], [], [], []
        for particle in self.inputParticles.get().iterItems():
            if not particle.isEnabled():
                continue
            shift, angle = geometryFromMatrix(
                particle.getTransform().getMatrix(), inverseTransform=True)
            ids.append(particle.getObjId())
            angles.append(angle)
            shifts.append(shift[:2])
            locations.append(getImageLocation(particle))

        angles = np.asarray(angles).reshape(-1, 3)
        shifts = np.asarray(shifts).reshape(-1, 2)
        np.savez(self._getDirectionsFn(),
                 ids=np.asarray(ids, dtype=np.int64),
                 rot=angles[:, 0], tilt=angles[:, 1], psi=angles[:, 2],
                 shiftX=shifts[:, 0], shiftY=shifts[:, 1],
                 location=np.asarray(locations))
        self.info('Read the angular assignment of %d particles' % len(ids))

    def assignStep(self):
        data = np.load(self._getDirectionsFn())
        dirs = eulerToDirection(data['rot'], data['tilt'])

        symMatrices = getSymmetryMatrices(self.symmetryGroup.get())
        mirrors = self.checkMirrors.get()
        gallery = buildGallery(self.angularSampling.get(), symMatrices, mirrors)
        self.info('Gallery: %d directions (%d symmetry matrices)'
                  % (gallery.shape[0], symMatrices.shape[0]))

        result = assignToDirections(dirs, gallery, symMatrices,
                                    self.ambiguityTolerance.get(), mirrors)

        # Class ids: non-empty gallery directions renumbered as 1..M
        assigned = result['assigned']
        usedDirections = np.unique(assigned)
        classOfDirection = np.zeros(gallery.shape[0], dtype=np.int64)
        classOfDirection[usedDirections] = np.arange(1, usedDirections.size + 1)
        classSize = np.bincount(assigned, minlength=gallery.shape[0])
        galRot, galTilt = directionToEuler(gallery)

        np.savez(self._getAssignmentFn(),
                 ids=data['ids'],
                 classId=classOfDirection[assigned],
                 angleAssigned=result['angleAssigned'],
                 numCandidates=result['numCandidates'],
                 usedDirections=usedDirections,
                 dirRot=galRot[usedDirections],
                 dirTilt=galTilt[usedDirections],
                 candidateSize=result['candidateSize'][usedDirections],
                 classSize=classSize[usedDirections])

        sizes = classSize[usedDirections]
        doubtful = result['numCandidates'] > 1
        stats = {
            'numParticles': int(assigned.size),
            'numGalleryDirections': int(gallery.shape[0]),
            'numClasses': int(usedDirections.size),
            'emptyDirections': int(gallery.shape[0] - usedDirections.size),
            'classSizeMin': int(sizes.min()),
            'classSizeMedian': float(np.median(sizes)),
            'classSizeMax': int(sizes.max()),
            'doubtfulParticles': int(doubtful.sum()),
            'movedFromNearest': int((assigned != result['nearest']).sum()),
            'angleAssignedMean': float(result['angleAssigned'].mean()),
            'angleAssignedMax': float(result['angleAssigned'].max()),
        }
        with open(self._getStatsFn(), 'w') as f:
            json.dump(stats, f, indent=2)

        with open(self._getExtraPath('directional_classes.txt'), 'w') as f:
            f.write('# classId rot tilt classSize candidateSize\n')
            for c, (rot, tilt, size, cand) in enumerate(
                    zip(galRot[usedDirections], galTilt[usedDirections],
                        sizes, result['candidateSize'][usedDirections]), start=1):
                f.write('%d %.3f %.3f %d %d\n' % (c, rot, tilt, size, cand))

        for key, value in stats.items():
            self.info('%s: %s' % (key, value))

    def computeAveragesStep(self):
        """
        Average of each class: up to maxImagesPerAverage particles per class
        are aligned in-plane to the class frame (xmipp_transform_geometry) and
        averaged; mirrored particles are flipped back before averaging.
        """
        from pwem import emlib

        data = np.load(self._getDirectionsFn())
        assignment = np.load(self._getAssignmentFn())
        classId = assignment['classId']
        numClasses = assignment['usedDirections'].size

        # Random subset of each class (deterministic)
        rng = np.random.default_rng(0)
        order = np.argsort(classId, kind='stable')
        bounds = np.searchsorted(classId[order], np.arange(1, numClasses + 2))
        maxImages = self.maxImagesPerAverage.get()
        selected = []
        for c in range(numClasses):
            members = order[bounds[c]:bounds[c + 1]]
            if members.size > maxImages:
                members = np.sort(rng.choice(members, maxImages, replace=False))
            selected.append(members)
        selected = np.concatenate(selected)
        selClass = classId[selected]

        psiApply, shiftXApply, shiftYApply, mirrored = inPlaneAlignment(
            data['rot'][selected], data['tilt'][selected], data['psi'][selected],
            data['shiftX'][selected], data['shiftY'][selected],
            assignment['dirRot'][selClass - 1], assignment['dirTilt'][selClass - 1],
            getSymmetryMatrices(self.symmetryGroup.get()),
            self.checkMirrors.get())

        md = emlib.MetaData()
        for i, idx in enumerate(selected):
            objId = md.addObject()
            md.setValue(emlib.MDL_IMAGE, str(data['location'][idx]), objId)
            md.setValue(emlib.MDL_ITEM_ID, int(data['ids'][idx]), objId)
            md.setValue(emlib.MDL_REF, int(selClass[i]), objId)
            md.setValue(emlib.MDL_ANGLE_PSI, float(psiApply[i]), objId)
            md.setValue(emlib.MDL_SHIFT_X, float(shiftXApply[i]), objId)
            md.setValue(emlib.MDL_SHIFT_Y, float(shiftYApply[i]), objId)
        fnSel = self._getTmpPath('average_images.xmd')
        fnAligned = self._getTmpPath('aligned_images.mrcs')
        md.write(fnSel)

        self.runJob('xmipp_transform_geometry',
                    '-i %s -o %s --save_metadata_stack %s --apply_transform '
                    '--keep_input_columns'
                    % (fnSel, fnAligned, self._getTmpPath('aligned_images.xmd')),
                    env=XmippPlugin.getEnviron(), numberOfMpi=1)

        with mrcfile.mmap(fnAligned, mode='r') as mrc:
            images = mrc.data
            averages = np.zeros((numClasses,) + images.shape[1:], np.float64)
            for i in range(selected.size):
                img = images[i]
                if mirrored[i]:
                    img = np.flip(img, axis=MIRROR_FLIP_AXIS)
                averages[selClass[i] - 1] += img
        counts = np.bincount(selClass - 1, minlength=numClasses)
        averages /= counts[:, None, None]

        mrcfile.write(self._getAveragesFn(), averages.astype(np.float32),
                      overwrite=True,
                      voxel_size=self.inputParticles.get().getSamplingRate())
        self.info('Class averages written to %s (%d images, %d mirrored)'
                  % (self._getAveragesFn(), selected.size, mirrored.sum()))
        os.remove(fnAligned)

    def createOutputStep(self):
        assignment = np.load(self._getAssignmentFn())
        self._classOfParticle = dict(zip(assignment['ids'].tolist(),
                                         assignment['classId'].tolist()))
        self._angleOfParticle = dict(zip(assignment['ids'].tolist(),
                                         assignment['angleAssigned'].tolist()))
        self._candidatesOfParticle = dict(zip(assignment['ids'].tolist(),
                                              assignment['numCandidates'].tolist()))
        self._classInfo = {
            c: (rot, tilt, cand)
            for c, rot, tilt, cand in zip(
                range(1, assignment['usedDirections'].size + 1),
                assignment['dirRot'].tolist(),
                assignment['dirTilt'].tolist(),
                assignment['candidateSize'].tolist())}

        inputParticles = self.inputParticles.get()
        classes2D = self._createSetOfClasses2D(inputParticles)
        classes2D.classifyItems(updateItemCallback=self._updateParticle,
                                updateClassCallback=self._updateClass)

        self._checkExclusiveClasses(classes2D, assignment['ids'].size)

        self._defineOutputs(outputClasses=classes2D)
        self._defineSourceRelation(self.inputParticles, classes2D)

    # --------------------------- UTILS functions -----------------------------
    def _checkExclusiveClasses(self, classes, numAssigned):
        """
        Verify on the written output that every particle belongs to exactly one
        class, and record the result in the stats file (shown in the summary).
        Raises RuntimeError if a particle is in more than one class.
        """
        seen = set()
        repeated = []
        numMemberships = 0
        for cls in classes:
            for particleId in cls.getIdSet():
                numMemberships += 1
                if particleId in seen:
                    repeated.append(particleId)
                seen.add(particleId)

        with open(self._getStatsFn()) as f:
            stats = json.load(f)
        stats['exclusiveClasses'] = not repeated
        stats['particlesInClasses'] = len(seen)
        stats['classMemberships'] = numMemberships
        stats['particlesInSeveralClasses'] = len(set(repeated))
        with open(self._getStatsFn(), 'w') as f:
            json.dump(stats, f, indent=2)

        if repeated:
            raise RuntimeError('%d particles belong to more than one class '
                               '(e.g. %s)' % (len(set(repeated)), repeated[:5]))
        if len(seen) != numAssigned:
            raise RuntimeError('%d particles were assigned to a class but %d '
                               'are in the output classes'
                               % (numAssigned, len(seen)))
        self.info('Check passed: each of the %d particles belongs to exactly '
                  'one class' % len(seen))

    def _updateParticle(self, item, row):
        classId = self._classOfParticle.get(item.getObjId())
        if classId is None:
            item._appendItem = False
            return
        item.setClassId(classId)
        item._cryoDirAngle = Float(self._angleOfParticle[item.getObjId()])
        item._cryoDirCandidates = Integer(self._candidatesOfParticle[item.getObjId()])

    def _updateClass(self, item):
        classId = item.getObjId()
        rot, tilt, cand = self._classInfo[classId]
        item._cryoDirRot = Float(rot)
        item._cryoDirTilt = Float(tilt)
        item._cryoCandidateSize = Integer(cand)

        # Representative: class average, in the frame of the class direction
        representative = item.getRepresentative()
        representative.setLocation(classId, self._getAveragesFn())
        representative.setSamplingRate(
            self.inputParticles.get().getSamplingRate())
        transform = Transform()
        transform.setMatrix(matrixFromGeometry(np.zeros(3),
                                               np.array([rot, tilt, 0.0]),
                                               inverseTransform=True))
        representative.setTransform(transform)

    def _getDirectionsFn(self):
        return self._getExtraPath('particle_directions.npz')

    def _getAssignmentFn(self):
        return self._getExtraPath('directional_assignment.npz')

    def _getAveragesFn(self):
        return self._getExtraPath('class_averages.mrcs')

    def _getStatsFn(self):
        return self._getExtraPath('directional_stats.json')

    # --------------------------- INFO functions ------------------------------
    def _validate(self):
        errors = []
        if self.angularSampling.get() <= 0:
            errors.append('The angular sampling must be positive.')
        if self.ambiguityTolerance.get() < 0:
            errors.append('The ambiguity tolerance cannot be negative.')
        if self.symmetryGroup.get().strip().lower().startswith('t'):
            # xmippLib SymList.readSymmetryFile('t') never returns in the
            # installed Xmipp version, so the matrices cannot be obtained
            errors.append('Tetrahedral symmetry is not supported: Xmipp '
                          'hangs when reading this symmetry group.')
        return errors

    def _summary(self):
        summary = []
        if not os.path.exists(self._getStatsFn()):
            summary.append('Output not ready yet.')
            return summary
        with open(self._getStatsFn()) as f:
            stats = json.load(f)
        summary.append('Particles: %d' % stats['numParticles'])
        if 'exclusiveClasses' in stats:
            if stats['exclusiveClasses']:
                summary.append('CHECK OK: each particle belongs to exactly one '
                               'class (%d particles, %d memberships)'
                               % (stats['particlesInClasses'],
                                  stats['classMemberships']))
            else:
                summary.append('CHECK FAILED: %d particles belong to several '
                               'classes' % stats['particlesInSeveralClasses'])
        summary.append('Directional classes: %d (of %d gallery directions)'
                       % (stats['numClasses'], stats['numGalleryDirections']))
        summary.append('Class size min/median/max: %d / %.1f / %d'
                       % (stats['classSizeMin'], stats['classSizeMedian'],
                          stats['classSizeMax']))
        summary.append('Doubtful particles (several candidate directions): %d'
                       % stats['doubtfulParticles'])
        summary.append('Assigned to a minority class instead of the nearest: %d'
                       % stats['movedFromNearest'])
        summary.append('Angle to assigned direction mean/max: %.2f / %.2f deg'
                       % (stats['angleAssignedMean'], stats['angleAssignedMax']))
        return summary

    def _methods(self):
        return ['Particles were grouped into directional classes according to '
                'their angular assignment (angular sampling %.1f deg, symmetry '
                '%s). Particles closer than %.1f deg to the border between '
                'several directions were assigned to the least populated one.'
                % (self.angularSampling.get(), self.symmetryGroup.get(),
                   self.ambiguityTolerance.get())]
