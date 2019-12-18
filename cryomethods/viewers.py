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

import os
from os.path import exists

import pyworkflow.em as em
import pyworkflow.em.viewers.showj as showj
import pyworkflow.em.metadata as md
from pyworkflow.em.viewers.plotter import EmPlotter
import pyworkflow.protocol.params as params
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)

from .protocols.protocol_volume_selector import ProtInitialVolumeSelector
from .convert import relionToLocation

RUN_LAST = 0
RUN_SELECTION = 1

VOLUME_SLICES = 0
VOLUME_CHIMERA = 1

CHIMERADATAVIEW = 0


class VolSelPlotter(EmPlotter):
    """ Class to create several plots with Xmipp utilities"""
    def __init__(self, x=1, y=1, mainTitle="", **kwargs):
        EmPlotter.__init__(self, x, y, mainTitle, **kwargs)

    def plotMd(self, mdObj, mdLabelX, mdLabelY, color='g',**args):
        """ plot metadata columns mdLabelX and mdLabelY
            if nbins is in args then and histogram over y data is made
        """
        if mdLabelX:
            xx = []
        else:
            xx = range(1, mdObj.size() + 1)
        yy = []
        for objId in mdObj:
            if mdLabelX:
                xx.append(mdObj.getValue(mdLabelX, objId))
            yy.append(mdObj.getValue(mdLabelY, objId))

        nbins = args.pop('nbins', None)
        if nbins is None:
            self.plotData(xx, yy, color, **args)
        else:
            self.plotHist(yy, nbins, color, **args)

    def plotMdFile(self, mdFilename, mdLabelX, mdLabelY, color='g', **args):
        """ plot metadataFile columns mdLabelX and mdLabelY
            if nbins is in args then and histogram over y data is made
        """
        mdObj = md.MetaData(mdFilename)
        self.plotMd(mdObj, mdLabelX, mdLabelY, color='g',**args)


class VolumeSelectorViewer(ProtocolViewer):
    """ This protocol serve to analyze the results of Initial
    Volume Selector protocol.
    """
    _targets = [ProtInitialVolumeSelector]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    _label = 'viewer volume selector'

    def _defineParams(self, form):
        self._env = os.environ.copy()
        form.addSection(label='Visualization')
        form.addParam('viewIter', params.EnumParam,
                      choices=['last', 'selection'], default=RUN_LAST,
                      display=params.EnumParam.DISPLAY_LIST,
                      label="Run to visualize",
                      help='*last*: only the last run will be '
                           'visualized.\n'
                           '*selection*: you may specify a range of '
                           'runs.\n'
                           'Examples:\n'
                           '"1,5-8,10" -> [1,5,6,7,8,10]\n'
                           '"2,6,9-11" -> [2,6,9,10,11]\n'
                           '"2 5, 6-8" -> [2,5,6,7,8] ')

        form.addParam('runSelection', params.NumericRangeParam,
                      condition='viewIter==%d' % RUN_SELECTION,
                      label="Runs list",
                      help="Write the iteration list to visualize.")

        group = form.addGroup('Volumes')
        group.addParam('displayVol', params.EnumParam,
                       choices=['slices', 'chimera'], default=VOLUME_SLICES,
                       display=params.EnumParam.DISPLAY_HLIST,
                       label='Display volume with',
                       help='*slices*: display volumes as 2D slices along z '
                            'axis.\n'
                            '*chimera*: display volumes as surface with '
                            'Chimera.')

    def _getVisualizeDict(self):
        visualizeDict = {'displayVol': self._showVolumes
                         }
        self._load()

        # If the is some error during the load, just show that instead
        # of any viewer
        if self._errors:
            for k in visualizeDict.keys():
                visualizeDict[k] = self._showErrors

        return visualizeDict

    def _showErrors(self, param=None):
        views = []
        self.errorList(self._errors, views)
        return views

    def _viewAll(self, *args):
        pass

# ==============================================================================
# ShowVolumes
# ==============================================================================
    def _showVolumes(self, paramName=None):
        if self.displayVol == VOLUME_CHIMERA:
            return self._showVolumesChimera()
        elif self.displayVol == VOLUME_SLICES:
            return self._showVolumesSqlite()

    def _showVolumesSqlite(self):
        """ Write (if it is needed) an sqlite with all volumes selected for
        visualization. """

        view = []
        if (self.viewIter == RUN_LAST and
                getattr(self.protocol, 'outputVolumes', None) is not None):
            fn = self.protocol.outputVolumes.getFileName()

            view.append(self.createView(filename=fn,
                                        viewParams=self._getViewParams()))
        else:
            for r in self._runs:
                volSqlte = self.protocol._getIterVolumes(r)
                view.append(self.createView(filename=volSqlte,
                                            viewParams=self._getViewParams()))
        return view

    def _showVolumesChimera(self):
        """ Create a chimera script to visualize selected volumes. """
        volumes = self._getVolumeNames()

        if len(volumes) > 1:
            cmdFile = self.protocol._getExtraPath('chimera_volumes.cmd')
            f = open(cmdFile, 'w+')
            for volFn in volumes:
                # We assume that the chimera script will be generated
                # at the same folder than relion volumes
                vol = volFn.replace(':mrc', '')
                localVol = os.path.basename(vol)
                if exists(vol):
                    f.write("open %s\n" % localVol)
            f.write('tile\n')
            f.close()
            view = em.ChimeraView(cmdFile)
        else:
            view = em.ChimeraClientView(volumes[0])

        return [view]

#===============================================================================
# Utils Functions
#===============================================================================
    def _getZoom(self):
        # Ensure that classes are shown at least at 128 px to
        # properly see the rlnClassDistribution label.[[
        dim = self.protocol.inputParticles.get().getDim()[0]
        if dim < 128:
            zoom = 128*100/dim
        else:
            zoom = 100
        return zoom

    def _validate(self):
        if self.lastIter is None:
            return ['There are not iterations completed.']

    def _getViewParams(self):
        labels = ('enabled id _filename _cmScore _rlnClassDistribution '
                 '_rlnAccuracyRotations _rlnAccuracyTranslations '
                  '_rlnEstimatedResolution')
        viewParams = {showj.ORDER: labels,
                      showj.MODE: showj.MODE_MD,
                      showj.VISIBLE: labels,
                      showj.RENDER: '_filename',
                      showj.SORT_BY: '_cmScore desc',
                      showj.ZOOM: str(self._getZoom())
                      }
        return viewParams


    def createView(self, filename, viewParams={}):
        return em.viewers.ObjectView(self._project, self.protocol.strId(),
                             filename, viewParams=viewParams)

    def _getRange(self, var, label):
        """ Check if the range is not empty.
        :param var: The variable to retrieve the value
        :param label: the labe used for the message string
        :return: the list with the range of values, empty
        """
        value = var.get()
        if value is None or not value.strip():
            self._errors.append('Provide %s selection.' % label)
            result = []
        else:
            result = self._getListFromRangeString(value)

        return result

    def _load(self):
        """ Load selected iterations and classes 3D for visualization mode. """
        self._refsList = [1]
        self._errors = []

        volSize = self.protocol.numOfVols.get()
        self._refsList = range(1, volSize+1)

        self.protocol._initialize() # Load filename templates
        self.lastIter = self.protocol._lastIter()

        if self.viewIter.get() == RUN_LAST:
            self._runs = [self.protocol.rLev.get()]
        else:
            self._runs = self._getRange(self.runSelection, 'runs')
        from matplotlib.ticker import FuncFormatter
        self._plotFormatter = FuncFormatter(self._formatFreq)

    def _formatFreq(self, value, pos):
        """ Format function for Matplotlib formatter. """
        inv = 999.
        if value:
            inv = 1/value
        return "1/%0.2f" % inv

    def _getVolumeNames(self):
        vols = []
        for r in self._runs:
            it = self.protocol._lastIter(r)
            for ref3d in self._refsList:
                volFn = self.protocol._getFileName('volume', ruNum=r,
                                                   ref3d=ref3d, iter=it)
                vols.append(volFn)
        return vols

    def _getModelStar(self, prefix, it):
        return self.protocol._getFileName(prefix + 'model', iter=it)

#     --------------------------PCA VIEWER----------------------------

PCA_COUNT = 1
PCA_THRESHOLD = 0
MDS= 0
LocallyLinearEmbedding= 1
Isomap= 2
TSNE = 3
linear= 0,
nearest= 1
slinear=2
quadratic= 3
cubic=4


FREQ_LABEL = 'frequency (1/A)'

class PcaLandscapeViewer(ProtocolViewer):
    _label = 'viewer resolution3D'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [ProtLandscapePCA]



    def _defineParams(self, form):
        form.addSection(label='Results')
        form.addParam('plotAutovalues', LabelParam,
                      label="Display cumulative sum of eigen values")
        group = form.addGroup('Landscape')
        group.addParam('interpolateType', EnumParam,
                       choices=['linear', 'nearest',
                                'slinear', 'quadratic', 'cubic'],
                       default=MDS,
                       label="Interpolation Type")
        group.addParam('linear', LabelParam,
                       condition='interpolateType==%d' % linear,
                       label='Linear')
        group.addParam('nearest', LabelParam,
                       condition='interpolateType==%d' % nearest,
                       label='nearest')
        group.addParam('slinear', LabelParam,
                       condition='interpolateType==%d' % slinear,
                       label='slinear')
        group.addParam('quadratic', LabelParam,
                       condition='interpolateType==%d' % quadratic,
                       label='quadratic')
        group.addParam('cubic', LabelParam,
                       condition='interpolateType==%d' % cubic,
                       label='cubic')


        group.addParam('heatMap', EnumParam,
                      choices=['MDS', 'LocallyLinearEmbedding', 'Isomap', 'TSNE'],
                      default=MDS,
                      label='Non-linear Manifold embedding',
                      help='select')
        group.addParam('MDS', LabelParam,
                     condition='heatMap==%d' % MDS,
                     label='heatMapMDS')
        group.addParam('LocallyLinearEmbedding', LabelParam,
                       condition='heatMap==%d' % LocallyLinearEmbedding,
                       label='heatMapLinEmbedding')
        group.addParam('Isomap', LabelParam,
                       condition='heatMap==%d' % Isomap,
                       label='heatMapIsomap')
        group.addParam('TSNE', LabelParam,
                       condition='heatMap==%d' % TSNE,
                       label='heatMapTSNE')
        group.addParam('matplotType', LabelParam,
                       label="View matplot")


        group.addParam('threeDMap', EnumParam,
                       choices=['MDS', 'LocallyLinearEmbedding', 'Isomap',
                                'TSNE'],
                       default=MDS,
                       label='3D Non-linear Manifold embedding',
                       help='select')
        # group.addParam('MDS', LabelParam,
        #                condition='threeDMap==%d' % MDS,
        #                label='3D Plot MDS' )
        # group.addParam('LocallyLinearEmbedding', LabelParam,
        #                condition='threeDMap==%d' % LocallyLinearEmbedding,
        #                label='3D Plot LocallyLinearEmbedding')
        # group.addParam('Isomap', LabelParam,
        #                condition='threeDMap==%d' % Isomap,
        #                label='3D Plot Isomap')
        # group.addParam('TSNE', LabelParam,
        #                condition='threeDMap==%d' % TSNE,
        #                label= '3D Plot Isomap')
        group.addParam('3DmatplotType', LabelParam,
                       label="View 3D matplot")


        group.addParam('binSize', params.IntParam, default=6,
                       label="select bin size")
        group.addParam('neighbourCount', params.IntParam, default=2,
                       label="Select neighbour points")
        group.addParam('pcaCount', params.IntParam, default=10,
                       label="Select number of PC")


        group.addParam('selectPoints', LabelParam,
                       label="Selct points on landscape")
        group.addParam('chimeraView', LabelParam,
                       label="Volumes in Chimera")


    def _getVisualizeDict(self):
        return {'plotAutovalues': self._plotAutovalues,
                'matplotType': self._viewMatplot,
                '3DmatplotType': self._get3DPlt

                # 'chimeraView': self._viewChimera,
                # 'thresholdMode': self._autovalueNumb,
                # 'selectPoints': self._selectPoints
                }


    def _loadPcaCoordinates(self):
        """ Check if the PCA data is generated and if so,
        read the data.
        *args and **kwargs will be passed to self._createPlot function.
        """
        matProjData = np.load(self.protocol._getExtraPath('Coordinates', 'matProj_splic.npy'))
        return matProjData

    def _loadPcaParticles(self):
        try:
            # filename = '/home/satinder/Desktop/NMA_MYSYS/splic_Tes_amrita.txt'
            filename = '/home/satinder/scipion_tesla_2.0/scipion-em-cryomethods/ortega_ribosome.txt'

            z_part = []
            with open(filename, 'r') as f:
                for y in f:
                    if y:
                        z_part.append(float(y.strip()))
            return z_part
        except ValueError:
            pass

    def _loadPcaEigenValue(self):
        eignValData = np.load(self.protocol._getExtraPath('EigenFile', 'eigenvalues.npy'))
        return eignValData



    def _autovalueNumb(self):
        # sCut = self.pcaCount.get()
        # thr = self.thr.get()
        dataValues = np.load(self.protocol._getExtraPath('EigenFile', 'matrix_vhDel.npy'))
        return dataValues

    def _plotAutovalues(self, paramName=None):
        temp_dir = '/extra'

        # ac= self.protocol._getExtraPath('EigenFile')
        autoVal= np.load(self.protocol._getExtraPath('EigenFile', 'eigenvalues.npy'))
        plt.plot(np.cumsum(autoVal))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

    def _interpolateType(self):
        if self.interpolateType.get() == linear:
            return self.linear.get()
        elif self.interpolateType.get() == nearest:
            return self.nearest.get()
        elif self.interpolateType.get() == slinear:
            return self.slinear.get()
        elif self.interpolateType.get() == quadratic:
            return self.quadratic.get()
        else:
            return 'cubic'


    def _viewMatplot(self,  paramName=None):
        kinds = ('nearest', 'zero', 'linear', 'slinear', 'quadratic', 'cubic')

        if self.heatMap.get() == MDS:
            nPCA = self.pcaCount.get()
            nBins = self.binSize.get()
            man = manifold.MDS(max_iter=100, n_init=1, random_state=0)
            mds = man.fit_transform(self._loadPcaCoordinates())
            counts, xedges, yedges = np.histogram2d(mds[:, 0], mds[:, 1],
                             weights=self._loadPcaParticles(), bins=nBins)
            countsExtended = np.zeros(
                (counts.shape[0] + 2, counts.shape[0] + 2))
            countsExtended[1:-1, 1:-1] = counts

            xedges = 0.5 * xedges[:-1] + 0.5 * xedges[1:]
            yedges = 0.5 * yedges[:-1] + 0.5 * yedges[1:]

            stepx = xedges[1] - xedges[0]
            stepy = yedges[1] - yedges[0]

            xedgesExtended = np.zeros(counts.shape[0] + 2)
            yedgesExtended = np.zeros(counts.shape[0] + 2)

            xedgesExtended[1:-1] = xedges
            xedgesExtended[0] = xedges[0] - stepx
            xedgesExtended[-1] = xedges[-1] + stepx

            yedgesExtended[1:-1] = yedges
            yedgesExtended[0] = yedges[0] - stepy
            yedgesExtended[-1] = yedges[-1] + stepy
            a = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                            num=countsExtended.shape[0])
            b = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                            num=countsExtended.shape[0])
            x, y = np.meshgrid(a, b, sparse=False, indexing='ij')

            a2 = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                             num=100)
            b2 = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                             num=100)
            grid_x, grid_y = np.meshgrid(a2, b2, sparse=False, indexing='ij')
            H2 = countsExtended.reshape(countsExtended.size)
            f = interpolate.interp2d(a, b, H2, kind='linear',
                                     bounds_error='True')
            znew = f(a2, b2)
            fig = plt.figure()
            CS = plt.contour(grid_x, grid_y, znew.T, 10, linewidths=1.5,
                             colors='k')
            CS = plt.contourf(grid_x, grid_y, znew.T, 20, cmap=plt.cm.hot,
                              vmax=(znew).max(), vmin=0)
            plt.colorbar()  # draw colorbar
            plt.show()

            fig_one = plt.figure()
            ax = fig_one.gca(projection='3d')
            ax.plot_surface(grid_x, grid_y, znew, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')
            plt.show()


        elif self.heatMap.get() == LocallyLinearEmbedding:
            nPCA = self.pcaCount.get()
            nBins = self.binSize.get()
            n_neighbors = self.neighbourCount.get() #self.protocol._inputVolLen()
            methods = ['standard', 'ltsa', 'hessian', 'modified']
            man = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2)
                                                  # eigen_solver='auto',
                                                  # method=methods[2],
                                                  # random_state=0)

            lle = man.fit_transform(self._loadPcaCoordinates()[:, 0:nPCA])
            counts, xedges, yedges = np.histogram2d(lle[:, 0], lle[:, 1],
                              weights=self._loadPcaParticles(), bins=nBins)
            countsExtended = np.zeros(
                (counts.shape[0] + 2, counts.shape[0] + 2))
            countsExtended[1:-1, 1:-1] = counts

            xedges = 0.5 * xedges[:-1] + 0.5 * xedges[1:]
            yedges = 0.5 * yedges[:-1] + 0.5 * yedges[1:]

            stepx = xedges[1] - xedges[0]
            stepy = yedges[1] - yedges[0]

            xedgesExtended = np.zeros(counts.shape[0] + 2)
            yedgesExtended = np.zeros(counts.shape[0] + 2)

            xedgesExtended[1:-1] = xedges
            xedgesExtended[0] = xedges[0] - stepx
            xedgesExtended[-1] = xedges[-1] + stepx

            yedgesExtended[1:-1] = yedges
            yedgesExtended[0] = yedges[0] - stepy
            yedgesExtended[-1] = yedges[-1] + stepy

            a = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                            num=countsExtended.shape[0])
            b = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                            num=countsExtended.shape[0])
            x, y = np.meshgrid(a, b, sparse=False, indexing='ij')

            a2 = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                             num=100)
            b2 = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                             num=100)
            grid_x, grid_y = np.meshgrid(a2, b2, sparse=False, indexing='ij')
            H2 = countsExtended.reshape(countsExtended.size)
            f = interpolate.interp2d(a, b, H2, kind= 'linear'
                                     ,bounds_error='True')
            znew = f(a2, b2)
            plt.figure()
            plt.contour(grid_x, grid_y, znew.T, 10, linewidths=1.5,
                             colors='k')
            plt.contourf(grid_x, grid_y, znew.T, 20, cmap=plt.cm.hot,
                              vmax=(znew).max(), vmin=0)
            plt.colorbar()  # draw colorbar
            plt.show()

        elif self.heatMap.get() == Isomap:

            nPCA = self.pcaCount.get()
            nBins = self.binSize.get()
            n_neighbors = self.neighbourCount.get()
            iso = manifold.Isomap(n_neighbors, n_components=2)

            isomap = iso.fit_transform(self._loadPcaCoordinates()[:, 0:nPCA])

            counts, xedges, yedges = np.histogram2d(isomap[:, 0], isomap[:, 1],
                                weights=self._loadPcaParticles(), bins=nBins)
            countsExtended = np.zeros(
                (counts.shape[0] + 2, counts.shape[0] + 2))
            countsExtended[1:-1, 1:-1] = counts


            xedges = 0.5 * xedges[:-1] + 0.5 * xedges[1:]
            yedges = 0.5 * yedges[:-1] + 0.5 * yedges[1:]

            stepx = xedges[1] - xedges[0]
            stepy = yedges[1] - yedges[0]

            xedgesExtended = np.zeros(counts.shape[0] + 2)
            yedgesExtended = np.zeros(counts.shape[0] + 2)

            xedgesExtended[1:-1] = xedges
            xedgesExtended[0] = xedges[0] - stepx
            xedgesExtended[-1] = xedges[-1] + stepx

            yedgesExtended[1:-1] = yedges
            yedgesExtended[0] = yedges[0] - stepy
            yedgesExtended[-1] = yedges[-1] + stepy

            a = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                            num=countsExtended.shape[0])
            b = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                            num=countsExtended.shape[0])
            x, y = np.meshgrid(a, b, sparse=False, indexing='ij')

            a2 = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                             num=100)
            b2 = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                             num=100)
            grid_x, grid_y = np.meshgrid(a2, b2, sparse=False, indexing='ij')
            H2 = countsExtended.reshape(countsExtended.size)
            f = interpolate.interp2d(a, b, H2, kind= 'linear',
                                     bounds_error='True')
            znew = f(a2, b2)
            fig = plt.figure()
            CS = plt.contour(grid_x, grid_y, znew.T, 10, linewidths=1.5,
                             colors='k')
            CS = plt.contourf(grid_x, grid_y, znew.T, 20, cmap=plt.cm.hot,
                              vmax=(znew).max(), vmin=0)
            plt.colorbar()  # draw colorbar
            plt.show()

        else:
            nPCA = self.pcaCount.get()
            nBins = self.binSize.get()
            man = manifold.TSNE(n_components=2, random_state=0)
            tsne = man.fit_transform(self._loadPcaCoordinates()[:, 0:nPCA])

            counts, xedges, yedges = np.histogram2d(tsne[:, 0], tsne[:, 1],
                                                    weights=self._loadPcaParticles(),
                                                    bins=nBins)

            countsExtended = np.zeros(
                (counts.shape[0] + 2, counts.shape[0] + 2))
            countsExtended[1:-1, 1:-1] = counts

            xedges = 0.5 * xedges[:-1] + 0.5 * xedges[1:]
            yedges = 0.5 * yedges[:-1] + 0.5 * yedges[1:]

            stepx = xedges[1] - xedges[0]
            stepy = yedges[1] - yedges[0]

            xedgesExtended = np.zeros(counts.shape[0] + 2)
            yedgesExtended = np.zeros(counts.shape[0] + 2)

            xedgesExtended[1:-1] = xedges
            xedgesExtended[0] = xedges[0] - stepx
            xedgesExtended[-1] = xedges[-1] + stepx

            yedgesExtended[1:-1] = yedges
            yedgesExtended[0] = yedges[0] - stepy
            yedgesExtended[-1] = yedges[-1] + stepy

            a = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                            num=countsExtended.shape[0])
            b = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                            num=countsExtended.shape[0])
            x, y = np.meshgrid(a, b, sparse=False, indexing='ij')

            a2 = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                             num=100)
            b2 = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                             num=100)
            grid_x, grid_y = np.meshgrid(a2, b2, sparse=False,
                                         indexing='ij')
            H2 = countsExtended.reshape(countsExtended.size)
            f = interpolate.interp2d(a, b, H2, kind= 'linear',
                                     bounds_error='True')
            znew = f(a2, b2)
            plt.figure()
            plt.contour(grid_x, grid_y, znew.T, 10, linewidths=1.5,
                             colors='k')
            plt.contourf(grid_x, grid_y, znew.T, 25, cmap=plt.cm.hot,
                              vmax=(znew).max(), vmin=0)
            plt.colorbar()  # draw colorbar
            plt.show()

    def _get3DPlt(self, paramName=None):
        if self.threeDMap.get == MDS:
            nPCA = self.pcaCount.get()
            nBins = self.binSize.get()
            man = manifold.MDS(max_iter=100, n_init=1, random_state=0)
            mds = man.fit_transform(self._loadPcaCoordinates())
            counts, xedges, yedges = np.histogram2d(mds[:, 0], mds[:, 1],
                                                    weights=self._loadPcaParticles(),
                                                    bins=nBins)
            countsExtended = np.zeros(
                (counts.shape[0] + 2, counts.shape[0] + 2))
            countsExtended[1:-1, 1:-1] = counts

            xedges = 0.5 * xedges[:-1] + 0.5 * xedges[1:]
            yedges = 0.5 * yedges[:-1] + 0.5 * yedges[1:]

            stepx = xedges[1] - xedges[0]
            stepy = yedges[1] - yedges[0]

            xedgesExtended = np.zeros(counts.shape[0] + 2)
            yedgesExtended = np.zeros(counts.shape[0] + 2)

            xedgesExtended[1:-1] = xedges
            xedgesExtended[0] = xedges[0] - stepx
            xedgesExtended[-1] = xedges[-1] + stepx

            yedgesExtended[1:-1] = yedges
            yedgesExtended[0] = yedges[0] - stepy
            yedgesExtended[-1] = yedges[-1] + stepy
            a = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                            num=countsExtended.shape[0])
            b = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                            num=countsExtended.shape[0])
            x, y = np.meshgrid(a, b, sparse=False, indexing='ij')

            a2 = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                             num=100)
            b2 = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                             num=100)
            grid_x, grid_y = np.meshgrid(a2, b2, sparse=False, indexing='ij')
            H2 = countsExtended.reshape(countsExtended.size)
            f = interpolate.interp2d(a, b, H2, kind=self._interpolateType(),
                                     bounds_error='True')
            znew = f(a2, b2)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(grid_x, grid_y, znew, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')
            plt.show()

        elif self.threeDMap.get == LocallyLinearEmbedding:
            nPCA = self.pcaCount.get()
            nBins = self.binSize.get()
            n_neighbors = self.neighbourCount.get()  # self.protocol._inputVolLen()
            methods = ['standard', 'ltsa', 'hessian', 'modified']
            man = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2)

            lle = man.fit_transform(self._loadPcaCoordinates()[:, 0:nPCA])
            counts, xedges, yedges = np.histogram2d(lle[:, 0], lle[:, 1],
                                                    weights=self._loadPcaParticles(),
                                                    bins=nBins)
            countsExtended = np.zeros(
                (counts.shape[0] + 2, counts.shape[0] + 2))
            countsExtended[1:-1, 1:-1] = counts

            xedges = 0.5 * xedges[:-1] + 0.5 * xedges[1:]
            yedges = 0.5 * yedges[:-1] + 0.5 * yedges[1:]

            stepx = xedges[1] - xedges[0]
            stepy = yedges[1] - yedges[0]

            xedgesExtended = np.zeros(counts.shape[0] + 2)
            yedgesExtended = np.zeros(counts.shape[0] + 2)

            xedgesExtended[1:-1] = xedges
            xedgesExtended[0] = xedges[0] - stepx
            xedgesExtended[-1] = xedges[-1] + stepx

            yedgesExtended[1:-1] = yedges
            yedgesExtended[0] = yedges[0] - stepy
            yedgesExtended[-1] = yedges[-1] + stepy

            a = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                            num=countsExtended.shape[0])
            b = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                            num=countsExtended.shape[0])
            x, y = np.meshgrid(a, b, sparse=False, indexing='ij')

            a2 = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                             num=100)
            b2 = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                             num=100)
            grid_x, grid_y = np.meshgrid(a2, b2, sparse=False, indexing='ij')
            H2 = countsExtended.reshape(countsExtended.size)
            f = interpolate.interp2d(a, b, H2, kind=self._interpolateType()
                                     , bounds_error='True')
            znew = f(a2, b2)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(grid_x, grid_y, znew, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')
            plt.show()

        elif self.threeDMap.get == Isomap:
            nPCA = self.pcaCount.get()
            nBins = self.binSize.get()
            n_neighbors = self.neighbourCount.get()
            iso = manifold.Isomap(n_neighbors, n_components=2)

            isomap = iso.fit_transform(self._loadPcaCoordinates()[:, 0:nPCA])

            counts, xedges, yedges = np.histogram2d(isomap[:, 0], isomap[:, 1],
                                                    weights=self._loadPcaParticles(),
                                                    bins=nBins)
            countsExtended = np.zeros(
                (counts.shape[0] + 2, counts.shape[0] + 2))
            countsExtended[1:-1, 1:-1] = counts

            xedges = 0.5 * xedges[:-1] + 0.5 * xedges[1:]
            yedges = 0.5 * yedges[:-1] + 0.5 * yedges[1:]

            stepx = xedges[1] - xedges[0]
            stepy = yedges[1] - yedges[0]

            xedgesExtended = np.zeros(counts.shape[0] + 2)
            yedgesExtended = np.zeros(counts.shape[0] + 2)

            xedgesExtended[1:-1] = xedges
            xedgesExtended[0] = xedges[0] - stepx
            xedgesExtended[-1] = xedges[-1] + stepx

            yedgesExtended[1:-1] = yedges
            yedgesExtended[0] = yedges[0] - stepy
            yedgesExtended[-1] = yedges[-1] + stepy

            a = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                            num=countsExtended.shape[0])
            b = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                            num=countsExtended.shape[0])
            x, y = np.meshgrid(a, b, sparse=False, indexing='ij')

            a2 = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                             num=100)
            b2 = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                             num=100)
            grid_x, grid_y = np.meshgrid(a2, b2, sparse=False, indexing='ij')
            H2 = countsExtended.reshape(countsExtended.size)
            f = interpolate.interp2d(a, b, H2, kind=self._interpolateType(),
                                     bounds_error='True')
            znew = f(a2, b2)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(grid_x, grid_y, znew, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')
            plt.show()

        else:
            nPCA = self.pcaCount.get()
            nBins = self.binSize.get()
            man = manifold.TSNE(n_components=2, random_state=0)
            tsne = man.fit_transform(self._loadPcaCoordinates()[:, 0:nPCA])

            counts, xedges, yedges = np.histogram2d(tsne[:, 0], tsne[:, 1],
                                                    weights=self._loadPcaParticles(),
                                                    bins=nBins)

            countsExtended = np.zeros(
                (counts.shape[0] + 2, counts.shape[0] + 2))
            countsExtended[1:-1, 1:-1] = counts

            xedges = 0.5 * xedges[:-1] + 0.5 * xedges[1:]
            yedges = 0.5 * yedges[:-1] + 0.5 * yedges[1:]

            stepx = xedges[1] - xedges[0]
            stepy = yedges[1] - yedges[0]

            xedgesExtended = np.zeros(counts.shape[0] + 2)
            yedgesExtended = np.zeros(counts.shape[0] + 2)

            xedgesExtended[1:-1] = xedges
            xedgesExtended[0] = xedges[0] - stepx
            xedgesExtended[-1] = xedges[-1] + stepx

            yedgesExtended[1:-1] = yedges
            yedgesExtended[0] = yedges[0] - stepy
            yedgesExtended[-1] = yedges[-1] + stepy

            a = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                            num=countsExtended.shape[0])
            b = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                            num=countsExtended.shape[0])
            x, y = np.meshgrid(a, b, sparse=False, indexing='ij')

            a2 = np.linspace(xedgesExtended.min(), xedgesExtended.max(),
                             num=100)
            b2 = np.linspace(yedgesExtended.min(), yedgesExtended.max(),
                             num=100)
            grid_x, grid_y = np.meshgrid(a2, b2, sparse=False,
                                         indexing='ij')
            H2 = countsExtended.reshape(countsExtended.size)
            f = interpolate.interp2d(a, b, H2, kind=self._interpolateType(),
                                     bounds_error='True')
            znew = f(a2, b2)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(grid_x, grid_y, znew, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')
            plt.show()












