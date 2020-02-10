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


class NumpyImgHandler(object):
    """ Class to provide several Numpy arrays manipulation utilities. """

    def __init__(self):
        # Now it will use Xmipp image library
        # to read and write most of formats, in the future
        # if we want to be independent of Xmipp, we should have
        # our own image library

        pass

    @classmethod
    def loadMrc(cls, fn, writable=True):
        """Return a NumPy array memory mapped from an existing MRC file.

        The returned NumPy array will have an attribute, Mrc, that is
        an instance of the Mrc class.  You can use that to access the
        header or extended header of the file.  For instance, if x
        was returned by bindFile(), x.Mrc.hdr.Num is the number of x
        samples, number of y samples, and number of sections from the
        file's header.

        Positional parameters:
        fn -- Is the name of the MRC file to bind.

        Keyword parameters:
        writable -- If True, the returned array will allow the elements of
        the array to be modified.  The Mrc instance packaged with the
        array will also allow modification of the header and extended
        header entries.  Changes made to the array elements or the header
        will affect the file to which the array is bound.
        """
        import mrc

        mode = 'r'
        if writable:
            mode = 'r+'
        a = mrc.Mrc(fn, mode)

        return a.data_withMrc(fn)

    @classmethod
    def saveMrc(cls, npVol, fn):
        import mrc
        mrc.save(npVol, fn, ifExists='overwrite')

    @classmethod
    def alignVolumes(cls, volToAlign, VolRef):
        import frm

        axis, shifts, angles, score = frm.frm_align(VolRef, None, volToAlign,
                                                    None, None, 20)
        return axis, shifts, angles, score

    @classmethod
    def applyTransforms(cls, volume, shifts, angles, axis=None):
        import transform

        npVol = transform.translate3d(volume, shifts[0], shifts[1], shifts[2])
        volume = transform.rotate3d(npVol, angles[0], angles[1], angles[2],
                                    center=axis)
        return volume

    @classmethod
    def doAverageMap(cls, mapFnList):
        """ Returns the average map and the type of the data.
        """
        for vol in mapFnList:
            npVol = cls.loadMrc(vol, False)
            if vol == mapFnList[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)
            npAvgVol += npVol

        npAvgVol = npAvgVol / len(mapFnList)
        return npAvgVol, dType

    @classmethod
    def getMapByThreshold(cls, vol, mult=3.0):
        """Returns a map masked by a threshold value, given by the standard
        deviation of the map.
        """
        volNp = cls.loadMrc(vol, False)
        std = mult * volNp.std()
        npMask = 1 * (volNp >= std)
        mapNp = volNp * npMask
        return mapNp


