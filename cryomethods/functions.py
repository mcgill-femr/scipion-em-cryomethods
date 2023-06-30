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
# ***************** *********************************************************
from multiprocessing import Pool

import numpy as np
from numpy import abs, sqrt, exp, log
from numpy.fft import fftn, ifftn


class NumpyImgHandler(object):
    """ Class to provide several Numpy arrays manipulation utilities. """

    def __init__(self):
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
    def load(cls, fn):
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
        return mrc.load(fn)


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
    #mean = mat.mean()
    #sigma = mat.std()
    #mat = (mat - mean) / sigma
    a = mat.min()
    b = mat.max()
    mat = (mat-a)/(b-a)
    return mat


def normalize(mat, max_value, min_value):
    """
    Set a numpy array as standard score and after that scale the data between -1 and 1.
    """
    mat = (mat-min_value)/(max_value-min_value)
    #mat = 2*(mat - 0.5)
    return mat

def calcPsd(img):
    """
    Calculate PSD using periodogram averaging
    """
    img_f = np.fft.fft2(img)
    img_f = np.fft.fftshift(img_f)
    img_f = abs(img_f)
    # img_f = img_f * img_f
    rows, cols = img_f.shape
    img_f = np.log(img_f / (rows * cols))

    q_80 = np.quantile(img_f, 0.98)
    q_20 = np.quantile(img_f, 0.02)
    img_f[img_f >= q_80] = q_80
    img_f[img_f < q_20] = q_20
    img_f = normalize(img_f, q_80, q_20)

    if False:
        x = np.linspace(-1, 1, img_f.shape[0])
        y = np.linspace(-1, 1, img_f.shape[0])
        img_f = img_f - polyfit2d(x, y, img_f, kx=2, ky=2, order=2)

    return img_f


def calcAvgPsd(img, windows_size=256, step_size=128, add_noise=False):
    """
    Calculate PSD using average periodogram
    """
    rows, cols = img.shape
    avg_psd = np.zeros((windows_size, windows_size))
    count = 0
    for i in range(0, rows - windows_size, step_size):
        for j in range(0, cols - windows_size, step_size):
            count += 1
            avg_psd += calcPsd(img[i:i + windows_size, j:j + windows_size])
    avg_psd /= count

    if add_noise:
        avg_psd = avg_psd + np.random.normal(0, 0.01, avg_psd.shape)

    x = np.linspace(-1, 1, windows_size)
    y = np.linspace(-1, 1, windows_size)
    avg_psd = avg_psd - polyfit2d(x, y, avg_psd, kx=2, ky=2, order=2)

    q_plus = np.quantile(avg_psd, 0.99)
    q_minus = np.quantile(avg_psd, 0.01)
    avg_psd[avg_psd >= q_plus] = q_plus
    avg_psd[avg_psd < q_minus] = q_minus
    avg_psd = normalize(avg_psd, q_plus, q_minus)

    return avg_psd


def calcAvgPsd_parallel(img, windows_size=256, step_size=128, num_workers=20):
    """
    Calculate PSD using average periodogram in parallel
    """
    import concurrent.futures
    rows, cols = img.shape
    avg_psd = np.zeros((windows_size, windows_size))
    count = 0
    regions = []
    for i in range(0, rows - windows_size, step_size):
        for j in range(0, cols - windows_size, step_size):
            count += 1
            regions.append((i, j))

    def calc_region_psd(region):
        i, j = region
        return calcPsd(img[i:i + windows_size, j:j + windows_size])

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        psd_futures = [executor.submit(calc_region_psd, region) for region in regions]
        for future in concurrent.futures.as_completed(psd_futures):
            avg_psd += future.result()

    avg_psd /= count

    x = np.linspace(-1, 1, windows_size)
    y = np.linspace(-1, 1, windows_size)
    avg_psd = avg_psd - polyfit2d(x, y, avg_psd, kx=2, ky=2, order=2)

    q_plus = np.quantile(avg_psd, 0.99)
    q_minus = np.quantile(avg_psd, 0.01)
    avg_psd[avg_psd >= q_plus] = q_plus
    avg_psd[avg_psd < q_minus] = q_minus
    avg_psd = normalize(avg_psd, q_plus, q_minus)
    return avg_psd

def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    x, y = np.meshgrid(x, y)

    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j

        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result

    c = np.linalg.lstsq(a.T, np.ravel(z), rcond=None)[0]

    z = x*0
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        z += c[index] * x**i * y**j
    return z


def fftnfreq(n, d=1):
    "Return the Discrete Fourier Transform sample frequencies"
    f = np.fft.fftfreq(n, d)
    return np.meshgrid(f, f, f)


def shell(f):
    "Return a normalized shell of spatial frequencies, around frequency f"
    global f_norm, f_width
    S = exp(- (f_norm - f)**2 / (2 * f_width**2))
    return S / np.sum(S)


def spiral_filter(voxel_n, voxel_size):
    "Return the freq-domain spiral filter for the three dimensions (x, y, z)"
    fx, fy, fz = fftnfreq(voxel_n, d=voxel_size)
    f_norm = sqrt(fx**2 + fy**2 + fz**2)

    def H(fi):
        return -1j * np.nan_to_num(fi / f_norm)

    with np.errstate(invalid='ignore'):  # ignore divide-by-0 warning in one bin
        return H(fx), H(fy), H(fz)


def mask_in_sphere(n):
    "Return a mask selecting voxels of the biggest sphere inside a n*n*n grid"
    coords = np.r_[:n] - n/2
    x, y, z = np.meshgrid(coords, coords, coords)
    r = sqrt(x**2 + y**2 + z**2)
    return r < n/2


def linear_fit_params(x, y):
    "Return the parameters a[...], b[...] when fitting y[:,...] to a+b*x[:]"
    # See https://en.wikipedia.org/wiki/Ordinary_least_squares#Matrix/vector_formulation
    X = np.column_stack((np.ones(len(x)), x))
    return np.tensordot(np.linalg.pinv(X), y, 1)


def amplitude_map(f):
    "Return the amplitude map and noise corresponding to the given frequency"
    global FV, Hx, Hy, Hz, mask_background, threshold, power_norm

    SFV = shell(f) * FV  # volume in frequency space, at frequency f

    v0 = ifftn(SFV)  # volume "at frequency f"

    vx = ifftn(Hx * SFV)  # conjugate volumes at this frequency
    vy = ifftn(Hy * SFV)
    vz = ifftn(Hz * SFV)

    m = sqrt(abs(v0)**2 + abs(vx)**2 + abs(vy)**2 + abs(vz)**2)  # amplitude

    q = np.quantile(m[mask_background], threshold)  # noise level

    # Something related to the SNR (all noise = 0.5 < Cref < 1 = no noise).
    Cref = m / (m + q)  # will be used to weight m

    # Amplitude map and noise at the corresponding frequency.
    Mod_f = log(Cref * m * power_norm)
    noise_f = log(q * power_norm)

    # Amplitude map and estimated noise at the corresponding frequency.
    return Mod_f, noise_f


def bfactor(vol, mask_in_molecule, voxel_size, min_res=15, max_res=2.96,
            num_points=10, noise_threshold=0.9, f_voxel_width=4.8,
            only_above_noise=False):
    "Return a map with the local b-factors at each voxel"
    global FV, Hx, Hy, Hz, mask_background, threshold, power_norm, f_norm, f_width

    voxel_n, _, _ = vol.shape  # vol is a 3d array n*n*n

    # Set some global variables.
    FV = fftn(vol)  # precompute the volume's Fourier transform
    threshold = noise_threshold  # to use in amplitude_map()
    power_norm = sqrt(voxel_n*voxel_n*voxel_n)  # normalization

    # To select voxels with background data (used in the quantile evaluation).
    mask_background = np.logical_and(~mask_in_molecule.astype(bool),
                                     mask_in_sphere(voxel_n))

    # Get ready to select frequencies (using a shell in frequency space).
    fx, fy, fz = fftnfreq(voxel_n, d=voxel_size)
    f_norm = sqrt(fx**2 + fy**2 + fz**2)
    f_width = f_voxel_width / (voxel_n * voxel_size)  # frequency width

    # Define the spiral filter in frequency space.
    Hx, Hy, Hz = spiral_filter(voxel_n, voxel_size)

    # Compute amplitude maps at frequencies between min and max resolution.
    freqs = np.linspace(1/min_res, 1/max_res, num_points)

    with Pool() as pool:
        mods, noises = zip(*pool.map(amplitude_map, freqs))
        Mod = np.array(mods)
        noise = np.array(noises)

    # Compute the local b-factor map.
    f2 = freqs**2
    a_b = linear_fit_params(f2, Mod)  # contains the fit parameters per voxel

    return a_b[1,:,:,:]  # the second parameter of the fit is the "b" map!


def occupancy(vol, mask_in_molecule, voxel_size, min_res=20, max_res=3,
              num_points=10, protein_threshold=0.25, f_voxel_width=4.6):
    "Return a map with the local occupancy at each voxel"
    global f_norm, f_width

    voxel_n, _, _ = vol.shape  # vol is a 3d array n*n*n
    FV = fftn(vol)  # precompute the volume's Fourier transform

    # Get ready to select frequencies (using a shell in frequency space).
    fx, fy, fz = fftnfreq(voxel_n, d=voxel_size)
    f_norm = sqrt(fx**2 + fy**2 + fz**2)
    f_width = f_voxel_width / (voxel_n * voxel_size)  # frequency width

    # Define the spiral filter in frequency space.
    Hx, Hy, Hz = spiral_filter(voxel_n, voxel_size)

    # Compute occupancy maps at frequencies between min and max resolution.
    freqs = linspace(1/min_res, 1/max_res, num_points)

    omap = np.zeros_like(vol)
    for f in freqs:
        SFV = shell(f) * FV  # volume in frequency space, at frequency f

        v0 = ifftn(SFV)  # volume "at frequency f"

        vx = ifftn(Hx * SFV)  # conjugate volumes at this frequency
        vy = ifftn(Hy * SFV)
        vz = ifftn(Hz * SFV)

        m = sqrt(abs(v0)**2 + abs(vx)**2 + abs(vy)**2 + abs(vz)**2)  # amplitude

        q = np.quantile(m[mask_in_molecule], protein_threshold)  # signal level

        omap += (m >= q)  # add 1 to voxels with amplitude above that quantile
    omap /= len(freqs)

    return omap
