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
        mrc.save(npVol.astype('float32'), fn, ifExists='overwrite')

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
    def getAverageMap(cls, mapFnList):
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
    def getAvgMapByStd(cls, mapFnList, mult=1):
        """ Returns the average map and the type of the data.
        """
        for vol in mapFnList:
            npVol = cls.getMapByStd(vol, mult)
            if vol == mapFnList[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)
            npAvgVol += npVol

        npAvgVol = npAvgVol / len(mapFnList)
        return npAvgVol, dType

    @classmethod
    def getMinCommonMask(cls, mapFnList, mult=1):
        """ Returns the average map and the type of the data.
        """
        for vol in mapFnList:
            npVol = cls.loadMrc(vol, False)
            if vol == mapFnList[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)
            npMask = cls.getMaskBelowThreshold(npVol, mult)
            npAvgVol += npMask
        npAvgMask = 1 * (npAvgVol > 0.3*len(mapFnList))
        return npAvgMask, dType

    @classmethod
    def getMapByStd(cls, vol, mult=1.0):
        """Returns a map masked by a threshold value, given by the standard
        deviation of the map.
        """
        volNp = cls.loadMrc(vol, False)
        std = mult * volNp.std()
        npMask = cls.getMaskAboveThreshold(std)
        mapNp = volNp * npMask
        return mapNp

    @classmethod
    def getMaskAboveThreshold(cls, volNp, mult=1):
        std = mult * volNp.std()
        npMask = 1 * (volNp > std)
        return npMask

    @classmethod
    def getMaskBelowThreshold(cls, volNp, mult=1):
        std = mult * volNp.std()
        npMask = 1 * (volNp <= std)
        return npMask

    @classmethod
    def volToList(cls, volNp, avgVol=None, mode='avg'):
        dim = volNp.shape[0]
        lenght = dim**3
        if avgVol is not None:
            if mode == 'avg':
                volNp -= avgVol
                volNp = volNp * cls.getMaskAboveThreshold(volNp, 0)
            else:
                volNp *= avgVol

        volNpList = volNp.reshape(lenght)
        return volNpList, volNpList.dtype

    @classmethod
    def getAllNpList(cls, listVol, mult=1, mode='avg'):
        avgMap, _ = cls.getAvgMapByStd(listVol, mult)
        listNpVol = []
        for volFn in listVol:
            npVol = cls.getMapByStd(volFn, mult)
            npList, _ = cls.volToList(npVol, avgMap, mode)
            listNpVol.append(npList)
        return listNpVol


class MlMethods(object):
    """ Class to provides several matching learning methods used in SPA. """
    def __init__(self):
        pass

    @classmethod
    def getCovMatrixAuto(cls, listVol, mult, mode='avg'):
        npIh = NumpyImgHandler()
        listNpVol = npIh.getAllNpList(listVol, mult, mode)
        covMatrix = np.cov(listNpVol)
        return covMatrix, listNpVol

    @classmethod
    def getCovMatrixManual(cls, listVol,  mult):
        npIh = NumpyImgHandler()
        npAvgVol, _ = npIh.getAverageMap(listVol)
        covMatrix = []
        for vol1 in listVol:
            npVol1 = npIh.loadMrc(vol1, False)
            npList1 = npIh.volToList(npVol1, npAvgVol)
            row = []
            for vol2 in listVol:
                npVol2 = npIh.loadMrc(vol2, False)
                npList2 = npIh.volToList(npVol2, npAvgVol)
                coef = np.cov(npList1, npList2)[0][1]
                row.append(coef)
            covMatrix.append(row)
        return covMatrix

    @classmethod
    def doPCA(cls, covMatrix, cut=1):
        u, s, vh = np.linalg.svd(covMatrix)
        cuttOffMatrix = sum(s) * cut
        sCut = 0
        for i in s:
            if cuttOffMatrix > 0:
                cuttOffMatrix = cuttOffMatrix - i
                sCut += 1
            else:
                break
        eigVecMat = np.delete(vh, np.s_[sCut:vh.shape[1]], axis=0)
        return eigVecMat, s

    @classmethod
    def getMatProjAuto(cls, listNpVol, eigVecMat):
        newBaseAxis = eigVecMat.dot(listNpVol)
        matProj = np.transpose(np.dot(newBaseAxis, np.transpose(listNpVol)))
        return matProj

    @classmethod
    def getMatProjManual(cls, listVol, eigVecMat):
        npIh = NumpyImgHandler()
        matProjTrasp = []
        npAvgVol = npIh.getAverageMap(listVol)
        dim = npAvgVol.shape[0]
        for eigenRow in eigVecMat:
            volBase = np.zeros((dim, dim, dim))
            for (volFn, eigenCoef) in zip(listVol, eigenRow):
                npVol = npIh.loadMrc(volFn, False)
                restNpVol = npVol - npAvgVol
                volBase += eigenCoef * restNpVol

            rowCoef = []
            for volFn in listVol:
                npVol = npIh.loadMrc(volFn, False)
                volRow = npIh.volToList(npVol, npAvgVol)
                npBaseRow = npIh.volToList(volBase)
                npBaseCol = npBaseRow.transpose()
                projCoef = np.dot(volRow, npBaseCol)
                rowCoef.append(projCoef)
            matProjTrasp.append(rowCoef)
        matProj = np.array(matProjTrasp).transpose()
        return matProj

    @classmethod
    def doSklearnKmeans(cls, matProj):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=matProj.shape[1]).fit(matProj)
        return kmeans.labels_

    @classmethod
    def doSklearnAffProp(cls, matProj):
        from sklearn.cluster import AffinityPropagation
        ap = AffinityPropagation(damping=0.5).fit(matProj)
        return ap.labels_

    @classmethod
    def doSklearnSpectralClustering(cls, matProj):
        from sklearn.cluster import SpectralClustering
        op = SpectralClustering(n_clusters=matProj.shape[1]-1).fit(matProj)
        return op.labels_

    @classmethod
    def doSklearnDBSCAN(cls, matProj):
        from sklearn.cluster import DBSCAN
        op = DBSCAN().fit(matProj)
        return op.labels_


def correctAnisotropy(volNp, weight, q, minFreq, maxFreq):
    # minFreq and maxFreq are in normalized frequency (max 0.5)
    size = volNp.shape[0]
    minFr = int(minFreq * size)
    maxFr = int(maxFreq * size)

    return correctAnisotropy(volNp, weight, q, minFr, maxFr)


def num_flat_features(x):
    """
    Flat a matrix and calculate the number of features
    """
    sizes = x.size()[1:]
    num_features = 1
    for s in sizes:
        num_features *= s
    return num_features


def normalize(mat):
    """
    Set a numpy array as standard score and after that scale the data between 0 and 1.
    """
    mean = mat.mean()
    sigma = mat.std()
    mat = (mat - mean) / sigma
    a = mat.min()
    b = mat.max()
    mat = (mat-a)/(b-a)
    return mat

def calcPsd(img):
    """
    Calculate PSD using periodogram
    """
    img_f = np.fft.fft2(img)
    img_f = np.fft.fftshift(img_f)
    img_f = abs(img_f)
    # img_f = img_f * img_f
    rows, cols = img_f.shape
    img_f = img_f / (rows * cols)
    return img_f


def calcAvgPsd(img, windows_size = 256, step_size = 128):
    """
    Calculate PSD using average periodogram
    """
    print(img.shape)
    rows, cols = img.shape
    avg_psd = np.zeros((windows_size, windows_size))
    count = 0
    for i in range(0, rows - windows_size, step_size):
        for j in range(0, cols - windows_size, step_size):
            count +=1
            avg_psd += calcPsd(img[i:i+windows_size, j:j+windows_size])
    avg_psd /= count
    return np.log(avg_psd)
