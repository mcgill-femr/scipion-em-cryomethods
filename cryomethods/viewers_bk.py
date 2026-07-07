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
from pwem import viewers
from pwem.viewers import EmPlotter, ChimeraView, showj #ChimeraOldViewer,
from pyworkflow import gui
from sympy.abc import lamda

from cryomethods.functions import NumpyImgHandler

try:
    from itertools import izip
except ImportError:
    izip = zip
from os.path import exists
import numpy as np
import tkinter as tk
from shutil import copyfile

from sklearn import manifold
import scipy as sc
import matplotlib.pyplot as plt

#from cryomethods import Plugin #QUITAR
import pyworkflow.utils.properties as pwprop
from pyworkflow.gui.widgets import Button, HotButton
import pyworkflow.protocol.params as params
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)

from .protocols.protocol_volume_selector import ProtInitialVolumeSelector
from .protocols.protocol_ML_landscape import ProtLandscapePCA
from .protocols.protocol_loc_pdf import ProtLocPDF, PROB_DENSITY_FUNCT, ACC_MOMENTS
from .protocols.protocol_loc_pdf_classes import ProtLocPDF_classes
from glob import glob
from scipy.stats import johnsonsu, lognorm
#from cryomethods.j_johnson_M import f_johnson_M
import mrcfile
#from fitter import Fitter, get_common_distributions

RUN_LAST = 0
RUN_SELECTION = 1

VOLUME_SLICES = 0
VOLUME_CHIMERA = 1

CHIMERADATAVIEW = 0

PCA_COUNT = 1
PCA_THRESHOLD = 0
MDS = 0
LLEMBEDDING = 1
Isomap = 2
TSNE = 3
LINEAR = 0
CUBIC = 1

PROB_DENSITY_FUNCT = 0
ACC_MOMENTS = 1

FREQ_LABEL = 'frequency (1/A)'

class CryoMethodsPlotter(EmPlotter):
    """ Class to create several plots with Xmipp utilities"""
    def __init__(self, x=1, y=1, mainTitle="", **kwargs):
        EmPlotter.__init__(self, x, y, mainTitle, **kwargs)

    def plotHeatMap(self, img, xGrid, yGrid, boltzLaw, cmap='hot'):
        """ plot metadata columns mdLabelX and mdLabelY
            if nbins is in args then and histogram over y data is made
        """
        img.contour(xGrid, yGrid, boltzLaw.T, 10, linewidths=1.5, colors='k')
        img.contourf(xGrid, yGrid, boltzLaw.T, 20, cmap=cmap,
                          vmax=(boltzLaw).max(), vmin=(boltzLaw).min())
        return img


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
            view = ChimeraView(cmdFile)
        else:
            view = ChimeraView(volumes[0])

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
        return viewers.ObjectView(self._project, self.protocol.strId(),
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

#  --------------------------NMA_Landscape VIEWER-------------------------------

# class NmaLandscapeViewer(ProtocolViewer):
#     _label = 'viewer resolution3D'
#     _targets = [ProtLandscapeNMA]
#     _environments = [DESKTOP_TKINTER, WEB_DJANGO]
#
#     def _defineParams(self, form):
#         form.addSection(label='Results')
#         form.addParam('plotAutovalues', params.LabelParam,
#                       label="Display cumulative sum of eigenvalues")
#
#         group = form.addGroup('Landscape')
#         group.addParam('heatMap', params.EnumParam,
#                       choices=['MDS', 'LocallyLinearEmbedding',
#                                'Isomap', 'TSNE'],
#                       default=MDS,
#                       label='Non-linear Manifold embedding',
#                       help='select')
#         group.addParam('interpolateType', em.EnumParam,
#                        choices=['linear', 'cubic'],
#                        default=0,
#                        label="Interpolation Type")
#         group.addParam('binSize', params.IntParam, default=6,
#                        label="select bin size")
#         group.addParam('neighbourCount', params.IntParam, default=3,
#                        label="Select neighbour points",
#                        condition="heatMap==1 or heatMap==2")
#         group.addParam('pcaCount', params.IntParam, default=10,
#                        label="Select number of principal components")
#         group.addParam('points', params.IntParam, default=5,
#                        label="Select number of volumes you want to show")
#         group.addParam('plot', params.EnumParam,
#                        choices=['2D', '3D'],
#                        default=0,
#                        label='view 2D or 3D free-energy landscape.')
#         group.addParam('dimensionality', params.LabelParam,
#                        label='View trajectory in 2D free-energy landscape')
#         group.addParam('scatterPlot', params.LabelParam,
#                        label='scatter plot of 2d free-energy landscape')
#
#
#
#
#     def _getVisualizeDict(self):
#         visualizeDict = {'plotAutovalues': self._plotAutovalues,
#                          # 'dimensionality': self._viewHeatMap,
#                          'plot': self._viewPlot,
#                          # 'scatterPlot': self._scatterPlot
#                          }
#         return visualizeDict
#
#     def _showErrors(self, param=None):
#         views = []
#         self.errorList(self._errors, views)
#         return views
#
#     def _viewAll(self, *args):
#         pass
#
#     # ==========================================================================
#     # Show sum of eigenvalues
#     # ==========================================================================
#     def _plotAutovalues(self, paramName=None):
#         fn = self.protocol._getExtraPath('EigenFile', 'eigenvalues.npy')
#         autoVal = np.load(fn)
#         vals = (np.cumsum(autoVal))
#         plt.plot(vals)
#         plt.show()
#
#     def _viewPlot(self, paramName=None):
#         if self.plot.get() == 0:
#             self._view2DPlot()
#         else:
#             self._view3DHeatMap()
#
#     def _view2DPlot(self):
#         fn= self.protocol._getExtraPath("particles.npy")
#         weight = np.load(fn)
#         print (weight, "weight")
#         nBins = self.binSize.get()
#         coords = self._genralplot()
#         xedges, yedges, counts=self._getEdges(coords, nBins, weight)
#
#         a = np.linspace(xedges.min(), xedges.max(), num=counts.shape[0])
#         b = np.linspace(yedges.min(), yedges.max(), num=counts.shape[0])
#
#         a2 = np.linspace(xedges.min(), xedges.max(), num=100)
#         b2 = np.linspace(yedges.min(), yedges.max(), num=100)
#         H2 = counts.reshape(counts.size)
#         grid_x, grid_y = np.meshgrid(a2, b2, sparse=False, indexing='ij')
#         if self.interpolateType == LINEAR:
#             intType = 'linear'
#         else:
#             intType = 'cubic'
#         f = sc.interpolate.interp2d(a, b, H2, kind=intType,
#                                     bounds_error='True')
#         znew = f(a2, b2)
#         print (znew, "znew")
#         # ---------------------------finding maxima on 2d map-------------------
#         minima = znew.max()
#         print (minima, "minimaaa")
#         tempValue= -25
#         fac= np.true_divide(znew, minima)
#         boltzFac = tempValue * fac
#         boltzParts = self.protocol._getExtraPath('boltzFac')
#         np.save(boltzParts, boltzFac)
#         boltzLaw= np.load(self.protocol._getExtraPath('boltzFac.npy'))
#
#         # ---------------------------------------------------------------
#
#         plt.figure()
#         plt.contour(grid_x, grid_y, boltzLaw.T, 10, linewidths=1.5, colors='k')
#         plt.contourf(grid_x, grid_y, boltzLaw.T, 25, cmap=plt.cm.hot,
#                           vmax=(boltzLaw).max(), vmin=(boltzLaw).min())
#         # ----------showing x,y,z under cursor---------------------------
#         Xflat, Yflat, Zflat = grid_x.flatten(), grid_y.flatten(), boltzLaw.T.flatten()
#         def fmt(x, y):
#             # get closest point with known data
#             dist = np.linalg.norm(np.vstack([Xflat - x, Yflat - y]), axis=0)
#             idx = np.argmin(dist)
#             z = Zflat[idx]
#             return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)
#         # -------------------------------------------------------------------
#
#         plt.gca().format_coord = fmt
#         plt.colorbar()
#         savePlot = self.protocol._getExtraPath('2d_PLOT.png')
#         plt.savefig(savePlot)
#         # draw colorbar
#         plt.show()
#
#     def _genralplot(self, paramName=None):
#         nPCA = self.pcaCount.get()
#         matProj = self._loadPcaCoordinates()
#         if self.heatMap.get() == MDS:
#             man = manifold.MDS(max_iter=100, n_init=1, random_state=0)
#             coords = man.fit_transform(matProj[:, 0:nPCA])
#
#         elif self.heatMap.get() == LLEMBEDDING:
#             n_neighbors = self.neighbourCount.get()
#             man = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2)
#             coords = man.fit_transform(matProj[:, 0:nPCA])
#
#         elif self.heatMap.get() == Isomap:
#             n_neighbors = self.neighbourCount.get()
#             iso = manifold.Isomap(n_neighbors, n_components=2)
#             coords = iso.fit_transform(matProj[:, 0:nPCA])
#
#         else:
#             man = manifold.TSNE(n_components=2, random_state=0)
#             coords = man.fit_transform(matProj[:, 0:nPCA])
#         return coords
#
#
#
#     def _loadPcaCoordinates(self):
#         """ Check if the PCA data is generated and if so,
#         read the data.
#         *args and **kwargs will be passed to self._createPlot function.
#         """
#         fn = self.protocol._getExtraPath('Coordinates', 'matProj_splic.npy')
#         matProjData = np.load(fn)
#         return matProjData
#
#     def _getEdges(self, crds, nBins, weight):
#         counts, xedges, yedges = np.histogram2d(crds[:, 0], crds[:, 1],
#                                                 weights=weight,
#                                                 bins=nBins)
#         shapeCounts = counts.shape[0] + 2
#         countsExtended = np.zeros((shapeCounts, shapeCounts))
#         countsExtended[1:-1, 1:-1] = counts
#
#         def extendEdges(edges, shapeCounts):
#             xedges = 0.5 * edges[:-1] + 0.5 * edges[1:]
#             stepx = edges[1] - edges[0]
#             xedgesExtended = np.zeros(shapeCounts)
#             xedgesExtended[1:-1] = xedges
#             xedgesExtended[0] = xedges[0] - stepx
#             xedgesExtended[-1] = xedges[-1] + stepx
#             return xedgesExtended
#
#         xedgesExtended = extendEdges(xedges, shapeCounts)
#         yedgesExtended = extendEdges(yedges, shapeCounts)
#
#         return xedgesExtended, yedgesExtended, countsExtended


#  --------------------------ML_Landscape VIEWER--------------------------------

class PcaLandscapeViewer(ProtocolViewer):
    _label = 'viewer resolution3D'
    _targets = [ProtLandscapePCA]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Results')
        form.addParam('plotAutovalues', params.LabelParam,
                      label="Eigenvalues cumulative sum (%)")

        group = form.addGroup('Landscape')
        group.addParam('heatMap', params.EnumParam,
                       choices=['MDS', 'LocallyLinearEmbedding',
                                'Isomap', 'TSNE'],
                       default=MDS,
                       label='Non-linear Manifold embedding',
                       help='select')
        group.addParam('interpolateType', params.EnumParam,
                       choices=['linear', 'cubic'],
                       default=0,
                       label="Interpolation Type")
        group.addParam('pcaCount', params.IntParam, default=10,
                       label="Select number of principal components")
        group.addParam('binSize', params.IntParam, default=6,
                       label="select bin size")
        group.addParam('neighbourCount', params.IntParam, default=3,
                       label="Select neighbour points",
                       condition="heatMap==1 or heatMap==2")
        group.addParam('plot', params.EnumParam,
                       choices=['2D', '3D'],
                       default=0,
                       label='view 2D or 3D free-energy landscape.')
        group.addParam('points', params.IntParam, default=5,
                       label="Select number of volumes you want to show")
        group.addParam('trajectory', params.LabelParam,
                       label='View trajectory in 2D free-energy landscape')
        # group.addParam('scatterPlot', params.LabelParam,
        #                label='scatter plot of 2d free-energy landscape')

        group = form.addGroup('Guess PC value')
        group.addParam('volNumb', params.IntParam, default=1,
                       label="Select the volume to reconstruct")
        # group.addParam('reconstructVol', params.LabelParam,
        #                label="reconstruct map with selected PC value ")

    def _getVisualizeDict(self):
        visualizeDict = {'plotAutovalues': self._plotAutovalues,
                         'plot': self._viewPlot,
                         'trajectory': self._viewTraj,
                         # 'scatterPlot': self._scatterPlot
                         # 'reconstructVol': self._pcaReconstruction
                         }

        return visualizeDict

    def _showErrors(self, param=None):
        views = []
        self._errors = []
        self.errorList(self._errors, views)
        return views

    def _viewAll(self, *args):
        pass

    # ==========================================================================
    # Show sum of eigenvalues
    # ==========================================================================
    def _plotAutovalues(self, paramName=None):
        fn = self.protocol._getExtraPath('EigenFile', 'eigenvalues.npy')
        autoVal = np.load(fn)
        sumVals = np.sum(autoVal)
        vals = np.cumsum(autoVal/sumVals)
        # print("EigenVals and sum: ", autoVal, vals)
        # plt.plot(autoVal)
        plt.plot(vals)
        plt.show()

    def _viewPlot(self, paramName=None):
        if self.plot.get() == 0:
            self._view2DPlot()
        else:
            self._view3DPlot()

    def _view2DPlot(self):
        grid_x, grid_y, boltzLaw = self._getGridAndBoltzman()

        plt.figure()
        plt.contour(grid_x, grid_y, boltzLaw.T, 10, linewidths=1.5, colors='k')
        plt.contourf(grid_x, grid_y, boltzLaw.T, 25, cmap=plt.cm.hot,
                          vmax=(boltzLaw).max(), vmin=(boltzLaw).min())
        # ----------showing x,y,z under cursor---------------------------
        Xflat, Yflat, Zflat = grid_x.flatten(), grid_y.flatten(), boltzLaw.T.flatten()

        def fmt(x, y):
            # get closest point with known data
            dist = np.linalg.norm(np.vstack([Xflat - x, Yflat - y]), axis=0)
            idx = np.argmin(dist)
            z = Zflat[idx]
            return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)
        # -------------------------------------------------------------------

        plt.gca().format_coord = fmt
        plt.colorbar()
        savePlot = self.protocol._getExtraPath('2d_PLOT.png')
        plt.savefig(savePlot)
        # draw colorbar
        plt.show()

    def _view3DPlot(self):
        grid_x, grid_y, boltzLaw = self._getGridAndBoltzman()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(grid_x, grid_y, boltzLaw, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        # draw colorbar
        plt.show()

    def _viewTraj(self, paramName=None):
        coords = self._genralplot()
        grid_x, grid_y, boltzLaw = self._getGridAndBoltzman()

        win = self.tkWindow(HeatMapWindow,
                            title='Heat Map',
                            coords=coords,
                            callback=self._getMaps
                            )
        plotter = self._createPlot("Heat Map", "x", "y", grid_x, grid_y,
                                   boltzLaw, figure=win.figure)
        self.path = PointPath(plotter.getLastSubPlot(), self._getPoints)
        win.show()

    def _genralplot(self,paramName=None):
        nPCA = self.pcaCount.get()
        matProj = self._loadPcaCoordinates()
        if self.heatMap.get() == MDS:
            man = manifold.MDS(max_iter=100, n_init=1, random_state=0)
            coords = man.fit_transform(matProj[:, 0:nPCA])

        elif self.heatMap.get() == LLEMBEDDING:
            n_neighbors = self.neighbourCount.get()
            man = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2)
            coords = man.fit_transform(matProj[:, 0:nPCA])

        elif self.heatMap.get() == Isomap:
            n_neighbors = self.neighbourCount.get()
            iso = manifold.Isomap(n_neighbors, n_components=2)
            coords = iso.fit_transform(matProj[:, 0:nPCA])

        else:
            man = manifold.TSNE(n_components=2, random_state=0)
            coords = man.fit_transform(matProj[:, 0:nPCA])
        return coords

    def _getMaps(self, coords):
        from cryomethods import Plugin
        Plugin.setEnviron()
        coordMaps = self._getCoordMapFiles()
        f = open(coordMaps)
        for i, l in enumerate(f):
            fn = self.protocol._getPath("volume_%02d.mrc" %i)
            weigths = []
            for coord in coords:
                value = list(map(float, l.split()))
                weigths.append(self._getDistanceWeigth(value, coord))

            inputMaps = self.protocol._getMrcVolumes()
            for j, (v, w) in enumerate(izip(inputMaps, weigths)):

                npVol = NumpyImgHandler.loadMrc(v, False)
                if j == 0:
                    dType = npVol.dtype
                    newMap = np.zeros(npVol.shape)
                newMap += (w*npVol/sum(weigths))
            NumpyImgHandler.saveMrc(newMap.astype(dType), fn)

        f.close()

    def _getPoints(self, data):
        xData = data.getXData()
        yData = data.getYData()

        f = open(self._getCoordMapFiles(), 'w')
        for x, y in izip (xData, yData):
            print(x, y, end="\n", file=f)
        f.close()

    def _loadPcaCoordinates(self):
        """ Check if the PCA data is generated and if so,
        read the data.
        *args and **kwargs will be passed to self._createPlot function.
        """
        fn = self.protocol._getExtraPath('Coordinates', 'matProj_splic.npy')
        matProjData = np.load(fn)
        return matProjData

    def _loadPcaWeights(self):
        pass

    def _loadData(self):
        data = PathData(dim=2)
        return data

    def _loadPcaEigenValue(self):
        fn = self.protocol._getExtraPath('EigenFile', 'eigenvalues.npy')
        eignValData = np.load(fn)
        return eignValData

    def _getEdges(self, crds, nBins, weight):
        counts, xedges, yedges = np.histogram2d(crds[:, 0], crds[:, 1],
                                                weights=weight,
                                                bins=nBins)
        shapeCounts = counts.shape[0] + 2
        countsExtended = np.zeros((shapeCounts, shapeCounts))
        countsExtended[1:-1, 1:-1] = counts

        def extendEdges(edges, shapeCounts):
            xedges = 0.5 * edges[:-1] + 0.5 * edges[1:]
            stepx = edges[1] - edges[0]
            xedgesExtended = np.zeros(shapeCounts)
            xedgesExtended[1:-1] = xedges
            xedgesExtended[0] = xedges[0] - stepx
            xedgesExtended[-1] = xedges[-1] + stepx
            return xedgesExtended

        xedgesExtended = extendEdges(xedges, shapeCounts)
        yedgesExtended = extendEdges(yedges, shapeCounts)

        return xedgesExtended, yedgesExtended, countsExtended

    def _createPlot(self, title, xTitle, yTitle, x, y, boltzLaw, figure=None):
        xplotter = CryoMethodsPlotter(figure=figure)
        xplotter.plot_title_fontsize = 11
        img = xplotter.createSubPlot(title, xTitle, yTitle, 1, 1)
        xplotter.plotHeatMap(img, x, y, boltzLaw)
        return xplotter

    def _scatterPlot(self, paramName=None):
        weight = self._getWeights()
        matProj = self._loadPcaCoordinates()
        area = (50 * np.ones(117))  # 0 to 15 point radii
        plt.scatter(matProj[:, 0], matProj[:, 1], s=area, c=weight, alpha=0.5)
        plt.colorbar()
        plt.show()

    def _getCoordMapFiles(self):
        return self.protocol._getExtraPath('new_map_coordinates.txt')

    def __getCoordMapFiles(self):
        return self.protocol._getExtraPath('all_map_coordinates.txt')

    def _getDistanceWeigth(self, p1, p2):
        d = -1*((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
        w = np.exp(d)
        return w

    def _pcaReconstruction(self, paramName=None):
        from cryomethods import Plugin
        Plugin.setEnviron()
        if not os.path.exists(self.protocol._getExtraPath('Select_PC')):
            os.mkdir(self.protocol._getExtraPath('Select_PC'))
        nPCA = self.pcaCount.get()
        print (nPCA)
        avgVol = self.protocol._getPath('extramap_average.mrc')
        npAvgVol = NumpyImgHandler.loadMrc(avgVol, False)
        print ("average map is here")
        dType = npAvgVol.dtype
        fnIn = self.protocol._getMrcVolumes()
        volNum = self.volNumb.get()
        initVolNum = volNum - 1
        iniVolNp = NumpyImgHandler.loadMrc(fnIn[0], False)

        dim = iniVolNp.shape[0]
        print (len(iniVolNp), "iniVolNp")
        lenght = dim ** 3
        reshapeVol = iniVolNp.reshape(lenght)
        subsAvgVol= reshapeVol- npAvgVol.reshape(lenght)
        # -------------------------covariance matrix----------------------
        cov_matrix= np.load(
            self.protocol._getExtraPath('CovMatrix', 'covMatrix.npy'))
        print (len(cov_matrix), "cov_matrix")
        u, s, vh = np.linalg.svd(cov_matrix)
        sCut = int(self.pcaCount.get())
        print (sCut, "scut")
        vhDel = np.transpose(np.delete(vh, np.s_[sCut:vh.shape[1]], axis=0))
        # --------------------obatining base-----------------------------
        for eignRow in vhDel.T:
            base = np.zeros(lenght)
            # volSelect = self.protocol._getExtraPath('volume_id_%02d.mrc' % (self.volNumb.get()))
            volSelect= fnIn[initVolNum:volNum]
            print(volSelect, "volSelect")
            for (vol, eigenCoef) in izip(volSelect,eignRow):
                volInp = NumpyImgHandler.loadMrc(vol, False)
                volInpR = volInp.reshape(lenght)
                volSubs = volInpR - npAvgVol.reshape(lenght)
                base += volSubs * eigenCoef
                volBase = base.reshape((dim, dim, dim))
                # break
            break
        nameVol = 'reconstruct_base_%02d.mrc' % (self.volNumb.get())
        print('-------------saving map %s-----------------' % nameVol)
        NumpyImgHandler.saveMrc(volBase.astype(dType),self.protocol._getExtraPath('Select_PC',nameVol))
        #
        # # ----------------matproj-----------------------------------------
        matProj = []
        baseMrc = self.protocol._getExtraPath('Select_PC', 'reconstruct_base_??.mrc')
        baseMrcFile = sorted(glob(baseMrc))
        volSelect = fnIn[initVolNum:volNum]
        for vol in volSelect:
            volNp = NumpyImgHandler.loadMrc(vol, False)
            restNpVol = volNp.reshape(lenght) - npAvgVol.reshape(lenght)
            volRow = restNpVol.reshape(lenght)
            rowCoef = []
            for baseVol in baseMrcFile:
                npVol = NumpyImgHandler.loadMrc(baseVol, writable=False)
                baseVol_row = npVol.reshape(lenght)
                baseVol_col = baseVol_row.transpose()
                projCoef = np.dot(volRow, baseVol_col)
                rowCoef.append(projCoef)
        matProj.append(rowCoef)
        print (matProj, "matProj")
        print ("length of bese file", len(baseMrcFile))
        #
        # # obtaining volumes from coordinates-----------------------------------
        for projRow in matProj:
            vol = np.zeros((dim, dim, dim))
            for baseVol, proj in zip(baseMrcFile, projRow):
                volNpo = NumpyImgHandler.loadMrc(baseVol, False)
                vol += volNpo * proj
            finalVol = vol + npAvgVol
            nameRes = 'reconstruct_%02d.mrc' % (self.volNumb.get())
            print('-------------saving reconstruct_vols %s-----------------' % nameRes)
            NumpyImgHandler.saveMrc(finalVol.astype(dType),
                        self.protocol._getExtraPath('Select_PC', nameRes))
        finalVol= fnIn[volNum]

        orgVol = 'original_%02d.mrc' % (self.volNumb.get())
        dst = self.protocol._getExtraPath('Select_PC', orgVol)
        # NumpyImgHandler.saveMrc(finalVol.astype(dType),self.protocol._getExtraPath('Select_PC', orgVol))
        copyfile(finalVol, dst)

    def _getWeights(self):
        inputClasses = self.protocol.inputClasses.get()
        weightList = []
        for cls in inputClasses:
            size = cls.getSize()
            weightList.append(size)
        return weightList

    def _getGridAndBoltzman(self):
        weights = self._getWeights()
        nBins = self.binSize.get()
        coords = self._genralplot()
        xedges, yedges, counts=self._getEdges(coords, nBins, weights)

        a = np.linspace(xedges.min(), xedges.max(), num=counts.shape[0])
        b = np.linspace(yedges.min(), yedges.max(), num=counts.shape[0])

        a2 = np.linspace(xedges.min(), xedges.max(), num=100)
        b2 = np.linspace(yedges.min(), yedges.max(), num=100)
        H2 = counts.reshape(counts.size)
        grid_x, grid_y = np.meshgrid(a2, b2, sparse=False, indexing='ij')
        if self.interpolateType == LINEAR:
            intType = 'linear'
        else:
            intType = 'cubic'
        f = sc.interpolate.interp2d(a, b, H2, kind=intType,
                                    bounds_error='True')
        znew = f(a2, b2)
        print (znew, "znew")
        # ---------------------------finding maxima on 2d map-------------------
        minima = znew.max()
        print (minima, "minimaaa")
        tempValue= -25
        fac= np.true_divide(znew, minima)
        boltzFac = tempValue * fac
        boltzParts = self.protocol._getExtraPath('boltzFac')
        np.save(boltzParts, boltzFac)
        boltzLaw = np.load(self.protocol._getExtraPath('boltzFac.npy'))
        return grid_x, grid_y, boltzLaw

class PointPath():
        """ Graphical manager based on Matplotlib to handle mouse
        events to create a path of points.
        It also allow to modify the point positions on the path.
        """

        def __init__(self, ax, callback=None):
            self.ax = ax
            self.callback = callback
            self.dragIndex = None

            self.cidpress = ax.figure.canvas.mpl_connect('button_press_event',
                                                         self.onClick)
            self.cidrelease = ax.figure.canvas.mpl_connect(
                              'button_release_event', self.onRelease)

            self.pathData = PathData(dim=2)
            self.setState(0)
            self.path_line = None
            self.path_points = None

        def setState(self, state, notify=False):
            self.drawing = state

            if state == 0:
                self.ax.set_title('Click to add points.')
            else:
                raise Exception("Invalid PointPath state: %d" % state)

        def onClick(self, event):
            if event.inaxes != self.ax:
                return

            ex = event.xdata
            ey = event.ydata

            if self.drawing == 0:
                point = self.pathData.createEmptyPoint()
                point.setX(ex)
                point.setY(ey)
                self.pathData.addPoint(point)

                if self.pathData.getSize() == 1:  # first point is added
                    self.plotPath()
                else:
                    xs, ys = self.getXYData()
                    self.path_line.set_data(xs, ys)
                    self.path_points.set_data(xs, ys)

                self.ax.figure.canvas.draw()

            if self.callback:
                self.callback(self.pathData)

        def getXYData(self):
            xs = self.pathData.getXData()
            ys = self.pathData.getYData()
            return xs, ys

        def plotPath(self):
            xs, ys = self.getXYData()
            self.path_line, = self.ax.plot(xs, ys, alpha=0.75, color='blue')
            self.path_points, = self.ax.plot(xs, ys, 'o',
                                             color='red')

        def onMotion(self, event):
            if self.dragIndex is None or self.drawing < 2:
                return

            ex, ey = event.xdata, event.ydata
            point = self.pathData.getPoint(self.dragIndex)
            point.setX(ex)
            point.setY(ey)
            self.update()

        def onRelease(self, event):
            self.dragIndex = None
            self.update()

        def update(self):
            xs, ys = self.getXYData()
            self.path_line.set_data(xs, ys)
            self.path_points.set_data(xs, ys)
            self.ax.figure.canvas.draw()


class HeatMapWindow(gui.Window):
    """ This class creates a Window that will display Bfactor plot
    to adjust two points to fit B-factor.
    It will also contain a button to apply the B-factor to
    the volume and produce a new volumen that can be registered.
    """

    def __init__(self, **kwargs):
        gui.Window.__init__(self, **kwargs)

        self.coords = kwargs.get('coords')
        self.callback = kwargs.get('callback', None)
        self.plotter = None

        content = tk.Frame(self.root)
        self._createContent(content)
        content.grid(row=0, column=0, sticky='news')
        content.columnconfigure(0, weight=1)
        # content.rowconfigure(1, weight=1)

    def _createContent(self, content):
        self._createFigureBox(content)

    def _createFigureBox(self, content):
        from pyworkflow.gui.matplotlib_image import FigureFrame
        figFrame = FigureFrame(content, figsize=(6, 6))
        figFrame.grid(row=0, column=0, padx=5, columnspan=2)
        self.figure = figFrame.figure

        applyBtn = HotButton(content, text='Obtain Maps',
                             command=self._onMapEstimationClick)
        applyBtn.grid(row=1, column=0, sticky='ne', padx=5, pady=5)

        closeBtn = Button(content, text='Close',
                          imagePath=pwprop.Icon.ACTION_CLOSE,
                          command=self.close)
        closeBtn.grid(row=1, column=1, sticky='ne', padx=5, pady=5)

    def _onMapEstimationClick(self, e=None):
        gui.dialog.FlashMessage(self.root, "Calculating maps...",
                            func=self.callback(self.coords))

    def _onClosing(self):
        if self.plotter:
            self.plotter.close()
        gui.Window._onClosing(self)


class Point():
    """ Return x, y 2d coordinates and some other properties
    such as weight and state.
    """
    # Selection states
    DISCARDED = -1
    NORMAL = 0
    SELECTED = 1

    def __init__(self, pointId, data, weight, state=0):
        self._id = pointId
        self._data = data
        self._weight = weight
        self._state = state
        self._container = None

    def getId(self):
        return self._id

    def getX(self):
        return self._data[self._container.XIND]

    def setX(self, value):
        self._data[self._container.XIND] = value

    def getY(self):
        return self._data[self._container.YIND]

    def setY(self, value):
        self._data[self._container.YIND] = value

    def getZ(self):
        return self._data[self._container.ZIND]

    def setZ(self, value):
        self._data[self._container.ZIND] = value

    def getWeight(self):
        return self._weight

    def getState(self):
        return self._state

    def setState(self, newState):
        self._state = newState

    def eval(self, expression):
        localDict = {}
        for i, x in enumerate(self._data):
            localDict['x%d' % (i + 1)] = x
        return eval(expression, {"__builtins__": None}, localDict)

    def setSelected(self):
        self.setState(Point.SELECTED)

    def isSelected(self):
        return self.getState() == Point.SELECTED

    def setDiscarded(self):
        self.setState(Point.DISCARDED)

    def isDiscarded(self):
        return self.getState() == Point.DISCARDED

    def getData(self):
        return self._data


class Data():
    """ Store data points. """

    def __init__(self, **kwargs):
        # Indexes of data
        self._dim = kwargs.get('dim')  # The points dimensions
        self.clear()

    def addPoint(self, point, position=None):
        point._container = self
        if position is None:
            self._points.append(point)
        else:
            self._points.insert(position, point)

    def getPoint(self, index):
        return self._points[index]

    def __iter__(self):
        for point in self._points:
            if not point.isDiscarded():
                yield point

    def iterAll(self):
        """ Iterate over all points, including the discarded ones."""
        return iter(self._points)

    def getXData(self):
        return [p.getX() for p in self]

    def getYData(self):
        return [p.getY() for p in self]

    def getZData(self):
        return [p.getZ() for p in self]

    def getWeights(self):
        return [p.getWeight() for p in self]

    def getSize(self):
        return len(self._points)

    def getSelectedSize(self):
        return len([p for p in self if p.isSelected()])

    def getDiscardedSize(self):
        return len([p for p in self.iterAll() if p.isDiscarded()])

    def clear(self):
        self.XIND = 0
        self.YIND = 1
        self.ZIND = 2
        self._points = []


class PathData(Data):
    """ Just contains two list of x and y coordinates. """

    def __init__(self, **kwargs):
        Data.__init__(self, **kwargs)

    def createEmptyPoint(self):
        data = [0.] * self._dim  # create 0, 0...0 point
        point = Point(0, data, 0)
        point._container = self

        return point

    def removeLastPoint(self):
        del self._points[-1]


class CalculateHistogram(ProtocolViewer):
    _label = 'voxel histogram'
    _targets = [ProtLocPDF, ProtLocPDF_classes]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Results')
        # --------------------------- Method applied -------------------------------------
        group = form.addGroup('Method used')
        group.addParam('methodApplied', params.EnumParam,
                  choices=['PROB_DENSITY_FUNCT', 'ACC_MOMENTS'],
                  important=True,
                  label='Method applied earlier', display=params.EnumParam.DISPLAY_COMBO,
                  help='Methods appllied to particles.\n'
                       '1. PROB_DENSITY_FUNCT calculates the pdf of a selected voxel from '
                       'the volumes obtained in each range interval.\n'
                       '2. ACC_MOMENTS calculates the pdf of the 4 moments from a selected voxel.'
                      )

        # --------------------------- Reconstructed input volume -------------------------------------
        group.addParam('inputVolume', params.PointerParam,
                     condition='methodApplied==%d' % PROB_DENSITY_FUNCT,
                     pointerClass='Volume',  #SetOfParticles
                     label="Reconstructed input volume\n"
                           "(resolution Nyquist)",
                     help='Select the reconstructed input volume with resolution Nyquist\n'
                          'by default.')

        # --------------------------- Noise threshold -------------------------------------
        group.addParam('threshNoise', params.FloatParam,
                       condition="methodApplied==%d" % PROB_DENSITY_FUNCT,
                       label="Noise threshold",
                       help='Value of noise threshold.')
        # Value of noise threshold from which we can calculate the noise volumes of mean, std, skewness and kurtosis.'

        # --------------------------- Moments -------------------------------------
        #groupNoise = form.addGroup('Noise', condition="methodApplied==%d" % PROB_DENSITY_FUNCT)
        #groupNoise.addParam('threshNoise', params.FloatParam,
        #                    label="Noise threshold",
        #                    help='Value of noise threshold from which we can calculate the noise '
        #                         'volumes of mean, std, skewness and kurtosis.')


        # --------------------------- Voxel value -------------------------------------
        groupVoxel = form.addGroup('Voxel')
        groupVoxel.addParam('z_value', params.IntParam, default=191,
                       label="Z value",
                       help='Z coordinate of voxel')

        groupVoxel.addParam('y_value', params.IntParam, default=183,
                       label="Y value",
                       help='Y coordinate of voxel')

        groupVoxel.addParam('x_value', params.IntParam, default=200,
                       label="X value",
                       help='X coordinate of voxel')

        groupVoxel.addParam('histogram', params.LabelParam,
                       label='View histogram of the voxel')


    def _calculateHistogram(self, paramName=None):
        from cryomethods import Plugin
        Plugin.setEnviron()

        if self.methodApplied == PROB_DENSITY_FUNCT:

            self.rango = np.load(self.protocol._getExtraPath("rango.npy"))
            print(f'Range: \n{self.rango}')

            bin_centers = np.array([(self.rango[i] + self.rango[i + 1]) / 2.0 for i in range(len(self.rango) - 1)])
            print(f'Bin_centers: \n{bin_centers}')

            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n'
                  f'Noise threshold: {self.threshNoise.get()} '
                  f'\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            bins_noise = bin_centers[bin_centers <= self.threshNoise.get()]
            print(f'bines correspondientes al ruido {bins_noise}')
            bins_prot = bin_centers[bin_centers > self.threshNoise.get()]
            print(f'bines correspondientes a proteina {bins_prot}')

            print(f'---------------------------------------')
            indices_noise = np.where(bin_centers <= self.threshNoise.get())[0]
            print(f'Índices de bins ruido: {indices_noise}')
            indices_prot = np.where(bin_centers > self.threshNoise.get())[0]
            print(f'Índices de bins proteina: {indices_prot}')
            print(f'---------------------------------------')

            range_volumes = []
            for i in range(1, self.protocol.numBins.get() +1):
                volume = self.protocol._getExtraPath("rangeVol_%s.mrc" % i)
                self.voxel_size = mrcfile.open(volume).voxel_size
                range_volumes.append(NumpyImgHandler.loadMrc(volume))


            weighted_mean = NumpyImgHandler.loadMrc(self.protocol._getExtraPath("weighted_mean.mrc"))
            weighted_std = NumpyImgHandler.loadMrc(self.protocol._getExtraPath("weighted_std.mrc"))
            weighted_skewness = NumpyImgHandler.loadMrc(self.protocol._getExtraPath("weighted_skewness.mrc"))
            weighted_kurtosis = NumpyImgHandler.loadMrc(self.protocol._getExtraPath("weighted_kurtosis.mrc"))
            most_probable_bin = NumpyImgHandler.loadMrc(self.protocol._getExtraPath("most_probable_bin.mrc"))


            voxel_values = np.array([])
            for i in range(len(range_volumes)):
                voxel_values = np.append(voxel_values, range_volumes[i][self.z_value.get(), self.y_value.get(), self.x_value.get()])
                #voxel_values.append(range_volumes[i][self.z_value.get(), self.y_value.get(), self.x_value.get()])


            print("Voxel values:", voxel_values)
            print(f'---------------------------------------')
            voxel_noise = voxel_values[bin_centers <= self.threshNoise.get()]
            voxel_prot = voxel_values[bin_centers > self.threshNoise.get()]
            print(f'Voxel values referred to noise {voxel_noise}')
            print(f'Voxel values referred to protein {voxel_prot}')
            print(f'---------------------------------------')

            valueVox_inputVol = NumpyImgHandler.loadMrc(self.inputVolume.get().getFileName())[self.z_value.get(), self.y_value.get(), self.x_value.get()]
            print(f'Real value of the voxel {self.z_value.get(), self.y_value.get(), self.x_value.get()}: {valueVox_inputVol}')

            mean_x = weighted_mean[self.z_value.get(), self.y_value.get(), self.x_value.get()]
            std_dev_x = weighted_std[self.z_value.get(), self.y_value.get(), self.x_value.get()]
            skew_x = weighted_skewness[self.z_value.get(), self.y_value.get(), self.x_value.get()]
            kurt_x = weighted_kurtosis[self.z_value.get(), self.y_value.get(), self.x_value.get()]
            max_bin_x = most_probable_bin[self.z_value.get(), self.y_value.get(), self.x_value.get()]


            print(f"Weighted mean in voxel {self.z_value.get(), self.y_value.get(), self.x_value.get()}: {mean_x}")
            print(f"Weighted standard deviation in voxel {self.z_value.get(), self.y_value.get(), self.x_value.get()}: {std_dev_x}")
            print(f"Weighted skewness in voxel {self.z_value.get(), self.y_value.get(), self.x_value.get()}: {skew_x}")
            print(f"Weighted kurtosis in voxel {self.z_value.get(), self.y_value.get(), self.x_value.get()}: {kurt_x}")
            print(f"Most probable bin in voxel {self.z_value.get(), self.y_value.get(), self.x_value.get()}: {max_bin_x}")
            print('-----------------------------------------------')
            print('-----------------------------------------------')

            #mrcfile.write(os.path.join(self.protocol._getExtraPath(), "noise_weighted_mean.mrc"), weighted_mean_noise.astype(np.float32), voxel_size=self.voxel_size)
            #mrcfile.write(output_max_freq, most_probable_freq.astype(np.float32), voxel_size=self.voxel_size)
            #mrcfile.write(output_max_bin, bin_values.astype(np.float32), voxel_size=self.voxel_size)
            #mrcfile.write(os.path.join(self.protocol._getExtraPath(), "noise_weighted_std.mrc"), weighted_std_noise.astype(np.float32), voxel_size=self.voxel_size)
            #mrcfile.write(os.path.join(self.protocol._getExtraPath(), "noise_weighted_skewness.mrc"), weighted_skewness_noise.astype(np.float32), voxel_size=self.voxel_size)
            #mrcfile.write(os.path.join(self.protocol._getExtraPath(), "noise_weighted_kurtosis.mrc"), weighted_kurtosis_noise.astype(np.float32), voxel_size=self.voxel_size)


            #noise_mean_voxel = noise_mean[self.z_value.get(), self.y_value.get(), self.x_value.get()]
            #noise_std_voxel = noise_std[self.z_value.get(), self.y_value.get(), self.x_value.get()]
            #noise_skewness_voxel = noise_skew[self.z_value.get(), self.y_value.get(), self.x_value.get()]
            #noise_kurtosis_voxel = noise_kurt[self.z_value.get(), self.y_value.get(), self.x_value.get()]

            #print(f"Noise weighted mean in voxel {self.z_value.get(), self.y_value.get(), self.x_value.get()}: {noise_mean_voxel}")
            #print(f"Noise weighted standard deviation in voxel {self.z_value.get(), self.y_value.get(), self.x_value.get()}: {noise_std_voxel}")
            #print(f"Noise weighted skewness in voxel {self.z_value.get(), self.y_value.get(), self.x_value.get()}: {noise_skewness_voxel}")
            #print(f"Noise weighted kurtosis in voxel {self.z_value.get(), self.y_value.get(), self.x_value.get()}: {noise_kurtosis_voxel}")
            #print('-----------------------------------------------')
            #print('-----------------------------------------------')





            #x = np.linspace(min(bin_centers), max(bin_centers), 4000) #self.protocol.numBins.get()
            #x = np.linspace(mean_x - 4 * std_dev_x, mean_x + 4 * std_dev_x, 1000)

            #kurt_x = (kurt_x + 3)
            #print(f_johnson_M(mean_x, std_dev_x, skew_x, kurt_x))
            #coef, _, _ = f_johnson_M(mean_x, std_dev_x, skew_x, kurt_x)
            #gamma, delta, xi, lambd = coef
            #print('valores de gamma, delta, xi y lamda', gamma, delta, xi, lambd)

            ## si es SL --> sin el kurt_x + 3 en el 0,0,0 salia algo

            #sigma = 1 / delta
            #scale = np.exp(xi + lambd)
            #print(f'sigma {sigma} y scale {scale}')

            #x_min = np.exp(xi + lambd - 3 * sigma)  # Ajusta el rango basado en la lognormal
            #x_max = np.exp(xi + lambd + 3 * sigma)
            #x = np.linspace(x_min, x_max, 40) #1000


            #y_johnson = lognorm.pdf(x, sigma, loc=0, scale=scale)

            #y_johnson = johnsonsu.pdf(x, gamma, delta, loc=xi, scale=abs(lambd))
            #print('VALOR DE JOHNSON', y_johnson)
            #y_johnson_scaled = y_johnson * np.sum(voxel_values) * (bin_centers[1] - bin_centers[0])

            #y_johnson *= np.max(voxel_values) / np.max(y_johnson)

            #f = Fitter(voxel_values, distributions=get_common_distributions())
            #f.fit()
            #f.summary()

            # ------------------------------- Full histogram (noise + protein) ------------------------------------
            plt.figure(figsize=(8, 6))
            plt.bar(bin_centers, voxel_values, width=0.00001, edgecolor="black", align="edge", color="black",
                    alpha=0.7)
            plt.plot(bin_centers, voxel_values, 'o', color='black')
            plt.plot(bin_centers, voxel_values, color='orange')

            # Weighted mean, weighted standard deviation and real value in the graphic
            plt.axvline(mean_x, color='red', linestyle='--', label=f"Weighted mean: {mean_x:.6f}")
            plt.axvline(mean_x - std_dev_x, color='green', linestyle='--', label=f"-1σ: {mean_x - std_dev_x:.6f}")
            plt.axvline(mean_x + std_dev_x, color='green', linestyle='--', label=f"+1σ: {mean_x + std_dev_x:.6f}")
            plt.axvline(valueVox_inputVol, color='blueviolet', linestyle='--', label=f"Voxel real_value: {valueVox_inputVol:.6f}")
            plt.axvline(max_bin_x, color='orchid', linestyle='--', label=f"Most probable bin: {max_bin_x:.6f}")

            plt.xlabel("Voxel intensity", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title(f"Histogram of voxel intensities {self.z_value.get(), self.y_value.get(), self.x_value.get()}", fontsize=14)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.legend()


            # -------------------------------------- Noise histogram ------------------------------------------
            plt.figure(figsize=(8, 6))
            plt.bar(bins_noise, voxel_noise, width=0.00001, edgecolor="black", align="edge", color="black",
                    alpha=0.7)
            plt.plot(bins_noise, voxel_noise, 'o', color='black')
            plt.plot(bins_noise, voxel_noise, color='orange')
            plt.title(f"Noise voxel intensities {self.z_value.get(), self.y_value.get(), self.x_value.get()} "
                      f"below threshold {self.threshNoise.get()}",
                      fontsize=14)

            # ------------------------------------- Protein histogram -----------------------------------------
            plt.figure(figsize=(8, 6))
            plt.bar(bins_prot, voxel_prot, width=0.00001, edgecolor="black", align="edge", color="black",
                    alpha=0.7)
            plt.plot(bins_prot, voxel_prot, 'o', color='black')
            plt.plot(bins_prot, voxel_prot, color='orange')
            plt.title(f"Protein voxel intensities {self.z_value.get(), self.y_value.get(), self.x_value.get()} "
                      f"above threshold {self.threshNoise.get()}",
                      fontsize=14)
            plt.show()

            ##plt.figure(figsize=(8, 6))
            #plt.plot(bin_centers, y_johnson, color='blue', linewidth=2, linestyle="solid", label="Johnson SU")
            #plt.show()


        elif self.methodApplied == ACC_MOMENTS:

            mean = NumpyImgHandler.loadMrc(self.protocol._getExtraPath("1_mean.mrc"))
            variance = NumpyImgHandler.loadMrc(self.protocol._getExtraPath("2_variance.mrc"))
            skewness = NumpyImgHandler.loadMrc(self.protocol._getExtraPath("3_skewness.mrc"))
            kurtosis = NumpyImgHandler.loadMrc(self.protocol._getExtraPath("4_kurtosis.mrc"))

            moments = [mean, variance, skewness, kurtosis]

            voxel_values = []
            for i in range(len(moments)):
                voxel_values.append(moments[i][self.z_value.get(), self.y_value.get(), self.x_value.get()])


            voxel_values[1] = np.sqrt(voxel_values[1])
            voxel_values[3] = voxel_values[3] + 3 #np.sqrt(voxel_values[3] + 3)
            print("Voxel values:", voxel_values)


            x = np.linspace(-1, 1, 500)
            '''revisar el tema del johnsonsu ya que los parámetros a, b, loc y scale de scipy.stats.johnsonsu
            no se corresponden directamente con la media, desviación típica, asimetría ni curtosis. No garantiza
            que la distribucion tenga la media, desviación, asimetría o curtosis determinadas, es decir, el
            resultado será alguna distribución Johnson, pero no tendrá los momentos que se quieren
            '''
            y_johnson = johnsonsu.pdf(x, voxel_values[2], voxel_values[3], loc=voxel_values[0], scale=voxel_values[1])

            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.1, bottom=0.35)

            ax.plot(x, y_johnson, label="Johnson SU")
            ax.legend()
            ax.set_title(f"Histogram of voxel intensities {self.z_value.get(), self.y_value.get(), self.x_value.get()}", fontsize=14)
            ax.set_xlabel("x")
            ax.set_ylabel("Density")
            plt.show()


            #hist, bin_edges = np.histogram(voxel_values, bins=10)
            #plt.figure(figsize=(8, 6))
            #plt.bar(bin_edges[:-1], hist, width=0.0001, edgecolor="black", align="edge", color="blue",
            #        alpha=0.7)
            #plt.xlabel("Voxel intensity", fontsize=12)
            #plt.ylabel("Frequency", fontsize=12)
            #plt.title("Histogram of voxel intensities", fontsize=14)
            #plt.grid(axis="y", linestyle="--", alpha=0.7)
            #plt.show()




    def _getVisualizeDict(self):
        visualizeDict = {'histogram': self._calculateHistogram}

        return visualizeDict
