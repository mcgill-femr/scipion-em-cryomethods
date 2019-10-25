import os
import re
import copy
import random
import numpy as np
from glob import glob
from collections import Counter

import pyworkflow.em as em
import pyworkflow.em.metadata as md
import pyworkflow.protocol.constants as cons
import pyworkflow.protocol.params as params
from pyworkflow.utils import (makePath, copyFile, replaceBaseExt)

from cryomethods import Plugin
from cryomethods.convert import (loadMrc, saveMrc)

from .protocol_base import ProtocolBase
from xmipp3.convert import getImageLocation
from os.path import join
from cryomethods import Plugin

from matplotlib import *
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata






class ProtLandscapePCA(ProtocolBase):
    _label = 'Control PCA'
    IS_2D = False
    IS_AUTOCLASSIFY = True

    def _initialize(self):
        """ This function is mean to be called after the
        working dir for the protocol have been set.
        (maybe after recovery from mapper)
        """
        self._createFilenameTemplates()
        self._createIterTemplates()

    def _createFilenameTemplates(self):
        """ Centralize how files are called for iterations and references. """
        self.levDir = self._getExtraPath()
        self.rLevDir = self._getExtraPath('lev_%(lev)02d/rLev_%(rLev)02d/')
        self.rLevIter = self.rLevDir + 'relion_it%(iter)03d_'
        # add to keys, data.star, optimiser.star and sampling.star
        myDict = {
            'input_star': self.levDir + 'input_rLev-%(rLev)03d.star',
            'outputData': self.levDir + 'output_data.star',
            'map': self.levDir + 'map_id-%(id)s.mrc',
            'avgMap': self.levDir+ 'map_average.mrc',
            'modelFinal': self.levDir + 'model.star',
            'relionMap': self.rLevDir + 'relion_it%(iter)03d_class%(ref3d)03d.mrc',
            'outputModel': self.levDir + 'output_model.star',
            'data': self.rLevIter + 'data.star',
            'rawFinalModel': self._getExtraPath('raw_final_model.star'),
            'rawFinalData': self._getExtraPath('raw_final_data.star'),
            'finalModel': self._getExtraPath('final_model.star'),
            'finalData': self._getExtraPath('final_data.star'),
            'finalAvgMap': self._getExtraPath('map_average.mrc'),
            'optimiser': self.rLevIter + 'optimiser.star',
            'all_avgPmax_xmipp': self._getTmpPath(
                'iterations_avgPmax_xmipp.xmd'),
            'all_changes_xmipp': self._getTmpPath(
                'iterations_changes_xmipp.xmd'),
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

    def _createIterTemplates(self, rLev=None):
        """ Setup the regex on how to find iterations. """
        rLev = self._rLev if rLev is None else rLev
        print("level rLev in _createIterTemplates:", self._level, rLev)

        self._iterTemplate = self._getFileName('data', lev=self._level,
                                               rLev=rLev,
                                               iter=0).replace('000', '???')
        # Iterations will be identify by _itXXX_ where XXX is the iteration
        # number and is restricted to only 3 digits.
        self._iterRegex = re.compile('_it(\d{3,3})_')
        self._classRegex = re.compile('_class(\d{2,2}).')

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes',
                      important=True,
                      label='Input volumes',
                      help='Initial reference 3D maps')
        form.addSection('Compute')
        form.addParam('useParallelDisk', params.BooleanParam, default=True,
                      label='Use parallel disc I/O?',
                      help='If set to Yes, all MPI slaves will read '
                           'their own images from disc. Otherwise, only '
                           'the master will read images and send them '
                           'through the network to the slaves. Parallel '
                           'file systems like gluster of fhgfs are good '
                           'at parallel disc I/O. NFS may break with many '
                           'slaves reading in parallel.')
        form.addParam('pooledParticles', params.IntParam, default=3,
                      label='Number of pooled particles:',
                      help='Particles are processed in individual batches '
                           'by MPI slaves. During each batch, a stack of '
                           'particle images is only opened and closed '
                           'once to improve disk access times. All '
                           'particle images of a single batch are read '
                           'into memory together. The size of these '
                           'batches is at least one particle per thread '
                           'used. The nr_pooled_particles parameter '
                           'controls how many particles are read together '
                           'for each thread. If it is set to 3 and one '
                           'uses 8 threads, batches of 3x8=24 particles '
                           'will be read together. This may improve '
                           'performance on systems where disk access, and '
                           'particularly metadata handling of disk '
                           'access, is a problem. It has a modest cost of '
                           'increased RAM usage.')

        form.addParam('skipPadding', em.BooleanParam, default=False,
                      label='Skip padding',
                      help='If set to Yes, the calculations will not use '
                           'padding in Fourier space for better '
                           'interpolation in the references. Otherwise, '
                           'references are padded 2x before Fourier '
                           'transforms are calculated. Skipping padding '
                           '(i.e. use --pad 1) gives nearly as good '
                           'results as using --pad 2, but some artifacts '
                           'may appear in the corners from signal that is '
                           'folded back.')

        form.addParam('allParticlesRam', params.BooleanParam, default=False,
                      label='Pre-read all particles into RAM?',
                      help='If set to Yes, all particle images will be '
                           'read into computer memory, which will greatly '
                           'speed up calculations on systems with slow '
                           'disk access. However, one should of course be '
                           'careful with the amount of RAM available. '
                           'Because particles are read in '
                           'float-precision, it will take \n'
                           '( N * (box_size)^2 * 4 / (1024 * 1024 '
                           '* 1024) ) Giga-bytes to read N particles into '
                           'RAM. For 100 thousand 200x200 images, that '
                           'becomes 15Gb, or 60 Gb for the same number of '
                           '400x400 particles. Remember that running a '
                           'single MPI slave on each node that runs as '
                           'many threads as available cores will have '
                           'access to all available RAM.\n\n'
                           'If parallel disc I/O is set to No, then only '
                           'the master reads all particles into RAM and '
                           'sends those particles through the network to '
                           'the MPI slaves during the refinement '
                           'iterations.')
        form.addParam('scratchDir', params.PathParam,
                      condition='not allParticlesRam',
                      label='Copy particles to scratch directory: ',
                      help='If a directory is provided here, then the job '
                           'will create a sub-directory in it called '
                           'relion_volatile. If that relion_volatile '
                           'directory already exists, it will be wiped. '
                           'Then, the program will copy all input '
                           'particles into a large stack inside the '
                           'relion_volatile subdirectory. Provided this '
                           'directory is on a fast local drive (e.g. an '
                           'SSD drive), processing in all the iterations '
                           'will be faster. If the job finishes '
                           'correctly, the relion_volatile directory will '
                           'be wiped. If the job crashes, you may want to '
                           'remove it yourself.')
        form.addParam('combineItersDisc', params.BooleanParam,
                      default=False,
                      label='Combine iterations through disc?',
                      help='If set to Yes, at the end of every iteration '
                           'all MPI slaves will write out a large file '
                           'with their accumulated results. The MPI '
                           'master will read in all these files, combine '
                           'them all, and write out a new file with the '
                           'combined results. All MPI salves will then '
                           'read in the combined results. This reduces '
                           'heavy load on the network, but increases load '
                           'on the disc I/O. This will affect the time it '
                           'takes between the progress-bar in the '
                           'expectation step reaching its end (the mouse '
                           'gets to the cheese) and the start of the '
                           'ensuing maximisation step. It will depend on '
                           'your system setup which is most efficient.')
        form.addParam('doGpu', params.BooleanParam, default=True,
                      label='Use GPU acceleration?',
                      help='If set to Yes, the job will try to use GPU '
                           'acceleration.')
        form.addParam('gpusToUse', params.StringParam, default='',
                      label='Which GPUs to use:', condition='doGpu',
                      help='This argument is not necessary. If left empty, '
                           'the job itself will try to allocate available '
                           'GPU resources. You can override the default '
                           'allocation by providing a list of which GPUs '
                           '(0,1,2,3, etc) to use. MPI-processes are '
                           'separated by ":", threads by ",". '
                           'For example: "0,0:1,1:0,0:1,1"')
        form.addParam('oversampling', params.IntParam, default=1,
                      label="Over-sampling",
                      help="Adaptive oversampling order to speed-up "
                           "calculations (0=no oversampling, 1=2x, 2=4x, etc)")
        form.addParam('extraParams', params.StringParam,
                      default='',
                      label='Additional parameters',
                      help="In this box command-line arguments may be "
                           "provided that are not generated by the GUI. This "
                           "may be useful for testing developmental options "
                           "and/or expert use of the program, e.g:\n"
                           "--dont_combine_weights_via_disc\n"
                           "--verb 1\n"
                           "--pad 2")
        form.addParallelSection(threads=1, mpi=4)

    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._level = 1
        self._rLev = 1
        self._evalIdsList = []
        self._doneList = []
        self._constStop = []
        self.stopDict = {}
        self._mapsDict = {}
        self._clsIdDict = {}

        # self._initialize()
        self._insertFunctionStep('estimatePCAStep')
    #-------------------------step function-----------------------------------
    def _getRefStar(self):
        return self._getExtraPath("input_references.star")

    def _getAverageVol(self,  volSet, listVol=[]):
        self._createFilenameTemplates()
        Plugin.setEnviron()

        inputObj = self.inputVolumes.get()
        fnIn = []
        for i in inputObj:

            a = getImageLocation(i).split(':')[0]
            fnIn.append(a)
        listVol= fnIn

        # listVol = self._getPathMaps() if not bool(listVol) else listVol
        print (listVol, "listVolll")
        print('creating average map: ', listVol)
        avgVol = self._getFileName('avgMap')
        print (avgVol, "avgVol")

        print('alignining each volume vs. reference')
        for vol in listVol:
            print (vol, "vol2")
            npVol = loadMrc(vol, writable=False)

            if vol == listVol[0]:
                dType = npVol.dtype
                npAvgVol = np.zeros(npVol.shape)
            npAvgVol += npVol

        print (npAvgVol, "npAvgVol1")
        npAvgVol = np.divide(npAvgVol, len(listVol))
        print('saving average volume')
        saveMrc(npAvgVol.astype(dType), avgVol)





    def estimatePCAStep(self):
        self._createFilenameTemplates()

        Plugin.setEnviron()
        listNpVol = []

        row = md.Row()
        refMd = md.MetaData()

        ih = em.ImageHandler()
        inputObj = self.inputVolumes.get()

        fnIn= []
        for i in inputObj:
            a = getImageLocation(i).split(':')[0]
            fnIn.append(a)

        for vol in fnIn:
            row.setValue(md.RLN_MLMODEL_REF_IMAGE, vol)
            row.addToMd(refMd)
        refMd.write(self._getRefStar())




        print (fnIn, "fninn")

        # for i in fnIn:
        #     print (i, "vol")
        #     row.setValue(md.RLN_MLMODEL_REF_IMAGE, i)
        #     row.addToMd(refMd)
        # refMd.write(self._getRefStar())

        listVol = fnIn
        self._getAverageVol(listVol)

        avgVol = self._getFileName('avgMap')
        npAvgVol = loadMrc(avgVol, False)
        dType = npAvgVol.dtype
        volNp = loadMrc(listVol.__getitem__(0), False)
        dim = volNp.shape[0]
        lenght = dim ** 3
        cov_matrix = []

        for vol in listVol:
            volNp = loadMrc(vol, False)

            # Now, not using diff volume to estimate PCA
            # diffVol = volNp - npAvgVol
            volList = volNp.reshape(lenght)

            row = []
            b = volList - npAvgVol.reshape(lenght)
            print (b, 'b')
            for j in listVol:
                npVol = loadMrc(j, writable=False)
                volList = npVol.reshape(lenght)
                volList_two = volList - npAvgVol.reshape(lenght)
                print (volList, "vollist")
                temp_a= np.corrcoef(volList_two, b).item(1)
                print (temp_a, "temp_a")
                row.append(temp_a)
                # b= volList_two
                # print (corr, "corr")
            cov_matrix.append(row)

        print (cov_matrix, "covMatrix")
        u, s, vh = np.linalg.svd(cov_matrix)
        cuttOffMatrix = sum(s) * 0.95
        sCut = 0

        print('cuttOffMatrix & s: ', cuttOffMatrix, s)
        for i in s:
            print('cuttOffMatrix: ', cuttOffMatrix)
            if cuttOffMatrix > 0:
                print("Pass, i = %s " % i)
                cuttOffMatrix = cuttOffMatrix - i
                sCut += 1
            else:
                break
        print('sCut: ', sCut)

        eigValsFile ='eigenvalues.txt'
        self._createMFile(s, eigValsFile)

        eigVecsFile = 'eigenvectors.txt'
        self._createMFile(vh, eigVecsFile)

        vhDel = np.transpose(np.delete(vh, np.s_[sCut:vh.shape[1]], axis=0))
        self._createMFile(vhDel, 'matrix_vhDel.txt')

        print(' this is the matrix "vhDel": ', vhDel)


        # insert at 1, 0 is the script path (or '' in REPL)

        mat_one = []
        for vol in listVol:
            volNp = loadMrc(vol, False)
            volList = volNp.reshape(lenght)
            print (volList, "volList")
            row_one = []
            for j in listVol:
                npVol = loadMrc(j, writable=False)
                volList_three = npVol.reshape(lenght)
                j_trans = volList_three.transpose()
                matrix_two = np.dot(volList, j_trans)
                row_one.append(matrix_two)
            mat_one.append(row_one)

        matProj = np.dot(mat_one, vhDel)
        # print (newBaseAxis, "newbase")
        # matProj = np.transpose(np.dot(newBaseAxis, mat_one))
        print (matProj, "matProj")

        classDis=[]

        mf = ('/home/josuegbl/PROCESSING/MAPS_FINALE/raw_final_model.star')
        print (mf, "mf")
        modelFile = md.MetaData('model_classes@' + mf)
        # for row in md.iterRows(modelFile):
        #     classDistrib = row.getValue('rlnClassDistribution')
        #     classDis.append(classDistrib)
        #
        # print (classDis, "clsDist")

        classDis=1


        x_proj = [item[0] for item in matProj]
        y_proj = [item[1] for item in matProj]
        print (x_proj, "x_proj")
        print (y_proj, "y_proj")
        print (len(x_proj), "xlength")
        print (len(y_proj), "ylength")
        print (len(classDis), "Clength")
        xmin = min(x_proj)
        ymin= min(y_proj)
        xmax = max(x_proj)
        ymax = max(y_proj)


        w = []
        for i in classDis:
            a = i * 100
            w.append(a)
        w = [int(i) for i in w]
        xi= np.arange(xmin, xmax, 0.01)
        yi= np.arange(ymin, ymax, 0.01)
        xi, yi = np.meshgrid(xi, yi)


        # set mask
        mask = (xi > 0.5) & (xi < 0.6) & (yi > 0.5) & (yi < 0.6)
        #save coordinates:
        mat_file = 'matProj(1).txt'
        self._createMFile(matProj, mat_file)
        x_file = 'x_proj(1).txt'
        self._createMFile(x_proj, x_file)
        y_file = 'y_proj(1).txt'
        self._createMFile(y_proj, y_file)

        # interpolate
        zi = griddata((x_proj, y_proj), classDis, (xi, yi), method='linear')
        # mask out the field
        zi[mask] = np.nan
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.contourf(xi, yi, zi)
        plt.hexbin(x_proj, y_proj, C=classDis, gridsize=20, mincnt=1, bins='log')
        plt.xlabel('x_pca', fontsize=16)
        plt.ylabel('y_pca', fontsize=16)
        plt.colorbar()
        plt.savefig('interpolated_controlPCA(2).png', dpi=100)
        plt.close(fig)
        # ---------------------plot success--------------------------
        # plt.hexbin(x_proj, y_proj, C=classDis, gridsize=60, bins='log',
        #            cmap='inferno')
        # plt.colorbar()
        # plt.show()

    # ------------------------------------------------------------------

    # for run in range(numOfRuns):
    #     mf = (self._getExtraPath('run_%02d' % run,
    #                              'relion_it%03d_' % iter +
    #                              'model.star'))
    #     print (mf, "mf")
    #     # mdClass.read(mf)
    #     ab = md.MetaData('model_classes@' + mf)
    #
    #     print (ab, "mdClass")
    #
    # clsDist = []
    #
    # for row in md.iterRows(ab):
    #     classDistrib = row.getValue('rlnClassDistribution')
    #     clsDist.append(classDistrib)
    #
    # print (clsDist, "clsDist")
    #
    # imgSet = self.inputParticles.get()
    # totalPart = imgSet.getSize()
    # print (totalPart, "totalPar")
    # K = self.numOfVols.get()
    # colors = cm.rainbow(np.linspace(0, 1, totalPart))
    # colorList = []
    # for i in clsDist:
    #     index = int(len(colors) * i)
    #     colorList.append(colors[index])
    #
    # x_proj = [item[0] for item in matProj]
    # y_proj = [item[1] for item in matProj]
    # print (x_proj, "x_proj")
    # print (y_proj, "y_proj")



    # plt.scatter(x_proj, y_proj, c=colorList, alpha=0.5)
    #
    # plt.show()


    def _getPathMaps(self):
        inputObj = self.inputVolumes.get()
        filesPath = []
        for i in inputObj:
            a = getImageLocation(i)
            filesPath.append(a)

        print (filesPath, "fninn")
        return sorted(glob(filesPath))

    def _createMFile(self, matrix, name='matrix.txt'):
        print (name, "name")
        f = open(name, 'w')        # f = open(name, 'w')
        for list in matrix:
            s = "%s\n" % list
            f.write(s)
        f.close()

    def _getClassId(self, volFile):
        result = None
        s = self._classRegex.search(volFile)
        if s:
            result = int(s.group(1)) # group 1 is 2 digits class number
        return self.volDict[result]

    def _fillVolSetIter(self, volSet):
        volSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._createFilenameTemplates()
        Plugin.setEnviron()
        modelStar = md.MetaData('model_classes@' +
                                self._getFileName('modelFinal'))


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
                vol._rlnClassDistributionl = em.Float(classDistrib)
                vol._rlnAccuracyRotations = em.Foat(accurracyRot)
                vol._rlnAccuracyTranslations = em.Float(accurracyTras)
                vol._rlnEstimatedResolution = em.Float(resol)
                volSet.append(vol)


    def _validate(self):
        errors = []
        return errors