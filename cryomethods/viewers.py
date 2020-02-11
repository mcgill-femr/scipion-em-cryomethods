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
from itertools import izip
from os.path import exists
import numpy as np
import Tkinter as tk

from django.contrib.localflavor import si
from sklearn import manifold
import scipy as sc
import matplotlib.pyplot as plt

from cryomethods import Plugin
import pyworkflow.utils.properties as pwprop
from pyworkflow.gui.widgets import Button, HotButton
import pyworkflow.em as em
import pyworkflow.gui as gui
import pyworkflow.em.viewers.showj as showj
from pyworkflow.em.viewers.plotter import EmPlotter
import pyworkflow.protocol.params as params
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)

from .protocols.protocol_volume_selector import ProtInitialVolumeSelector
from .protocols.protocol_ML_landscape import ProtLandscapePCA
from .protocols.protocol_NMA_landscape import ProtLandscapeNMA
from convert import loadMrc, saveMrc
import matplotlib as mpl
import math
from scipy.signal import argrelextrema
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import scipy



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

#  --------------------------NMA_Landscape VIEWER-------------------------------

class NmaLandscapeViewer(ProtocolViewer):
    _label = 'viewer resolution3D'
    _targets = [ProtLandscapeNMA]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Results')
        form.addParam('plotAutovalues', params.LabelParam,
                      label="Display cumulative sum of eigenvalues")

        group = form.addGroup('Landscape')
        group.addParam('heatMap', params.EnumParam,
                      choices=['MDS', 'LocallyLinearEmbedding',
                               'Isomap', 'TSNE'],
                      default=MDS,
                      label='Non-linear Manifold embedding',
                      help='select')
        group.addParam('interpolateType', em.EnumParam,
                       choices=['linear', 'cubic'],
                       default=0,
                       label="Interpolation Type")
        group.addParam('binSize', params.IntParam, default=6,
                       label="select bin size")
        group.addParam('neighbourCount', params.IntParam, default=3,
                       label="Select neighbour points",
                       condition="heatMap==1 or heatMap==2")
        group.addParam('pcaCount', params.IntParam, default=10,
                       label="Select number of principal components")
        group.addParam('points', params.IntParam, default=5,
                       label="Select number of volumes you want to show")
        group.addParam('plot', params.EnumParam,
                       choices=['2D', '3D'],
                       default=0,
                       label='view 2D or 3D free-energy landscape.')
        group.addParam('dimensionality', params.LabelParam,
                       label='View trajectory in 2D free-energy landscape')
        group.addParam('scatterPlot', params.LabelParam,
                       label='scatter plot of 2d free-energy landscape')




    def _getVisualizeDict(self):
        visualizeDict = {'plotAutovalues': self._plotAutovalues,
                         # 'dimensionality': self._viewHeatMap,
                         'plot': self._viewPlot,
                         # 'scatterPlot': self._scatterPlot
                         }
        return visualizeDict

    def _showErrors(self, param=None):
        views = []
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
        vals = (np.cumsum(autoVal))
        plt.plot(vals)
        plt.show()

    def _viewPlot(self, paramName=None):
        if self.plot.get() == 0:
            self._view2DPlot()
        # else:
        #     self._view3DHeatMap()

    def _genralplot(self, paramName=None):
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

    def _view2DPlot(self):
        fn = self.protocol._getExtraPath('Particle_Weights.npy')
        weight = np.load(fn)
        print (weight, "weight")
        nBins = self.binSize.get()
        coords = self._genralplot()
        xedges, yedges, counts = self._getEdges(coords, nBins, weight)

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
        tempValue = -25
        fac = np.true_divide(znew, minima)
        boltzFac = tempValue * fac
        boltzParts = self.protocol._getExtraPath('boltzFac')
        np.save(boltzParts, boltzFac)
        boltzLaw = np.load(self.protocol._getExtraPath('boltzFac.npy'))

        # ---------------------------------------------------------------

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

    def _loadPcaCoordinates(self):
        """ Check if the PCA data is generated and if so,
        read the data.
        *args and **kwargs will be passed to self._createPlot function.
        """
        fn = self.protocol._getExtraPath('Coordinates', 'matProj_splic.npy')
        matProjData = np.load(fn)
        return matProjData

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


#  --------------------------ML_Landscape VIEWER--------------------------------

class PcaLandscapeViewer(ProtocolViewer):
    _label = 'viewer resolution3D'
    _targets = [ProtLandscapePCA]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Results')
        form.addParam('plotAutovalues', params.LabelParam,
                      label="Display cumulative sum of eigenvalues")

        group = form.addGroup('Landscape')
        group.addParam('heatMap', params.EnumParam,
                      choices=['MDS', 'LocallyLinearEmbedding',
                               'Isomap', 'TSNE'],
                      default=MDS,
                      label='Non-linear Manifold embedding',
                      help='select')
        group.addParam('interpolateType', em.EnumParam,
                       choices=['linear', 'cubic'],
                       default=0,
                       label="Interpolation Type")
        group.addParam('binSize', params.IntParam, default=6,
                       label="select bin size")
        group.addParam('neighbourCount', params.IntParam, default=3,
                       label="Select neighbour points",
                       condition="heatMap==1 or heatMap==2")
        group.addParam('pcaCount', params.IntParam, default=10,
                       label="Select number of principal components")
        group.addParam('points', params.IntParam, default=5,
                       label="Select number of volumes you want to show")
        group.addParam('addWeights', params.FileParam, label="Weight File path",
                       allowsNull=True,
                       help='Specify a path to weights for volumes.')
        group.addParam('plot', params.EnumParam,
                       choices=['2D', '3D'],
                       default=0,
                       label='view 2D or 3D free-energy landscape.')
        group.addParam('dimensionality', params.LabelParam,
                       label='View trajectory in 2D free-energy landscape')
        group.addParam('scatterPlot', params.LabelParam,
                       label='scatter plot of 2d free-energy landscape')




    def _getVisualizeDict(self):
        visualizeDict = {'plotAutovalues': self._plotAutovalues,
                         'dimensionality': self._viewHeatMap,
                         'plot': self._viewPlot,
                         'scatterPlot': self._scatterPlot
                         }
        return visualizeDict

    def _showErrors(self, param=None):
        views = []
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
        vals = (np.cumsum(autoVal))
        plt.plot(vals)
        plt.show()

    def _getParticles(self):
        weightPath= self.addWeights.get()

        with open(weightPath) as f:
            lines = f.readlines()
        parts= np.loadtxt(lines, delimiter=', ', unpack=True)

        particleArray = self.protocol._getExtraPath('Particle_Weights')
        partWeights = np.save(particleArray, parts)
        return partWeights



    def _viewPlot(self, paramName=None):
        if self.plot.get() == 0:
            self._view2DPlot()
        else:
            self._view3DHeatMap()


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


    def _viewHeatMap(self, paramName=None):
        self._getParticles()
        fn = self.protocol._getExtraPath('Particle_Weights.npy')
        weight = np.load(fn)
        nPCA = self.pcaCount.get()
        nBins = self.binSize.get()
        coords= self._genralplot()
        xedges, yedges, counts = self._getEdges(coords, nBins, weight)

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
        # ---------------------------finding maxima on 2d map-------------------
        minima = znew.max()
        print (minima, "minimaaa")
        tempValue = -(25)
        fac= np.true_divide(znew, minima)
        boltzFac = tempValue * fac
        boltzParts = self.protocol._getExtraPath('boltzFac')
        np.save(boltzParts, boltzFac)
        boltzLaw = np.load(self.protocol._getExtraPath('boltzFac.npy'))

        # ---------------------------------------------------------------

        win = self.tkWindow(HeatMapWindow,
                            title='Heat Map',
                            coords=coords,
                            callback=self._getMaps
                            )
        plotter = self._createPlot("Heat Map", "x", "y", grid_x, grid_y,
                                   boltzLaw, figure=win.figure)
        self.path = PointPath(plotter.getLastSubPlot(), self._getPoints)
        win.show()

    def _getMaps(self, coords):
        Plugin.setEnviron()
        coordMaps = self._getCoordMapFiles()
        f = open(coordMaps)
        for i, l in enumerate(f):
            fn = self.protocol._getPath("volume_%02d.mrc" %i)
            weigths = []
            for coord in coords:
                value = map(float, l.split())
                weigths.append(self._getDistanceWeigth(value, coord))

            inputMaps = self.protocol._getMrcVolumes()
            for j, (v, w) in enumerate(izip(inputMaps, weigths)):

                npVol = loadMrc(v, False)
                if j == 0:
                    dType = npVol.dtype
                    newMap = np.zeros(npVol.shape)
                newMap += (w*npVol/sum(weigths))
            saveMrc(newMap.astype(dType), fn)

        f.close()
        # volSet = self.protocol._createSetOfVolumes()
        # volSet.setSamplingRate(pixelSize)
        # newVol = vol.clone()
        # newVol.setObjId(None)
        # newVol.setLocation(volOut)
        # volSet.append(newVol)
        # volSet.write()
        #
        # self.objectView(volSet).show()

    def _getPoints(self, data):
        xData = data.getXData()
        yData = data.getYData()

        f = open(self._getCoordMapFiles(), 'w')
        for x, y in izip (xData, yData):
            print >> f, x, y
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

    def _view2DPlot(self):
        self._getParticles()
        fn = self.protocol._getExtraPath('Particle_Weights.npy')
        weight = np.load(fn)
        print (weight, "weight")
        nBins = self.binSize.get()
        coords = self._genralplot()
        xedges, yedges, counts=self._getEdges(coords, nBins, weight)

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
        boltzLaw= np.load(self.protocol._getExtraPath('boltzFac.npy'))

        # ---------------------------------------------------------------

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

    def _scatterPlot(self, paramName=None):
        self._getParticles()
        fn = self.protocol._getExtraPath('Particle_Weights.npy')
        weight = np.load(fn)
        print (weight, "weight")
        matProj = self._loadPcaCoordinates()
        area = (50 * np.ones(117))  # 0 to 15 point radii
        colors = weight
        plt.scatter(matProj[:, 0], matProj[:, 1], s=area, c=colors, alpha=0.5)
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

    def _view3DHeatMap(self):
        self._getParticles()
        fn = self.protocol._getExtraPath('Particle_Weights.npy')
        weight = np.load(fn)
        nBins = self.binSize.get()
        coords = self._genralplot()
        xedges, yedges, counts = self._getEdges(coords, nBins, weight)

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
        # ---------------------------finding maxima on 2d map-------------------
        minima = znew.max()
        print (minima, "minimaaa")
        tempValue = -25
        fac = np.true_divide(znew, minima)
        boltzFac = tempValue * fac
        boltzParts = self.protocol._getExtraPath('boltzFac')
        np.save(boltzParts, boltzFac)
        boltzLaw = np.load(self.protocol._getExtraPath('boltzFac.npy'))

        # ---------------------------------------------------------------

        fig= plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(grid_x, grid_y, boltzLaw, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        # draw colorbar
        plt.show()





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
        figFrame = FigureFrame(content, figsize=(13, 13))
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
