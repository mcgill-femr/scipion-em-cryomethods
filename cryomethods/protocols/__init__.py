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

from .protocol_2d_auto_classifier import Prot2DAutoClassifier
from .protocol_3d_auto_classifier import Prot3DAutoClassifier
from .protocol_volume_selector import ProtInitialVolumeSelector
from .protocol_directional_pruning import ProtDirectionalPruning
from .protocol_directional_ransac import ProtClass3DRansac
from .protocol_volume_clustering import ProtVolClustering
from .protocol_correction import ProtocolMapCorrector
# from .protocol_NMA_landscape import ProtLandscapeNMA
from .protocol_ML_landscape import ProtLandscapePCA
from .protocol_CNN import ProtSCNN
from .protocol_CTF import ProtDCTF