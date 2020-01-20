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
"""
This sub-package contains cryoMethods protocols and tools.
"""
import os, sys
import pyworkflow.em
import pyworkflow.utils as pwutils

from .constants import RELION_CRYOMETHODS_HOME, CRYOMETHODS_HOME, V3_0

# from bibtex import _bibtex # Load bibtex dict with references
_logo = "cryomethods_logo.png"
_references = []


class Plugin(pyworkflow.em.Plugin):
    _homeVar = CRYOMETHODS_HOME
    _pathVars = [CRYOMETHODS_HOME, RELION_CRYOMETHODS_HOME]
    _supportedVersions = []

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(CRYOMETHODS_HOME, 'cryomethods-0.1')
        cls._defineEmVar(RELION_CRYOMETHODS_HOME, 'relion-3.0')


    @classmethod
    def getEnviron(cls):
        """ Setup the environment variables needed to launch Relion. """

        environ = cls.getRelionEnviron()
        pythonPath = [cls.getHome('imageLib'),
                      cls.getHome('alignLib'),
                      cls.getHome('alignLib/frm/swig'),
                      cls.getHome('alignLib/tompy')]

        libPath = [cls.getHome('alignLib/frm/swig'),
                   cls.getHome('alignLib/SpharmonicKit27')]

        # for lPath in libPath:
        #     if not lPath in os.environ['LD_LIBRARY_PATH']:
        #        environ.update({'LD_LIBRARY_PATH': lPath},
        #                       position=pwutils.Environ.BEGIN)

        for pPath in pythonPath:
            if not pPath in os.environ['PYTHONPATH']:
                environ.update({'PYTHONPATH': pPath},
                               position=pwutils.Environ.BEGIN)
        return environ


    @classmethod
    def __getRelionHome(cls, *paths):
        """ Return a path from the "home" of the package
         if the _homeVar is defined in the plugin. """
        home = cls.getVar(RELION_CRYOMETHODS_HOME)
        return os.path.join(home, *paths) if home else ''


    @classmethod
    def setEnviron(cls):
        environ = cls.getEnviron()
        pythonPath = [cls.getHome('imageLib'),
                      cls.getHome('alignLib'),
                      cls.getHome('alignLib/frm/swig'),
                      cls.getHome('alignLib/tompy')]

        for path in pythonPath:
            if not path in sys.path:
                sys.path.append(path)
        os.environ.update(cls.getEnviron())


    @classmethod
    def getRelionEnviron(cls):
        """ Setup the environment variables needed to launch Relion. """

        environ = pwutils.Environ(os.environ)
        binPath = cls.__getRelionHome('bin')
        libPath = cls.__getRelionHome('lib') + ":" + cls.__getRelionHome('lib64')

        if not binPath in environ['PATH']:
            environ.update({'PATH': binPath,
                            'LD_LIBRARY_PATH': libPath,
                            'SCIPION_MPI_FLAGS': os.environ.get('RELION_MPI_FLAGS', ''),
                            }, position=pwutils.Environ.BEGIN)

        # Take Scipion CUDA library path
        cudaLib = environ.getFirst(('RELION_CUDA_LIB', 'CUDA_LIB'))
        environ.addLibrary(cudaLib)

        return environ

    @classmethod
    def getActiveRelionVersion(cls):
        """ Return the version of the Relion binaries that is currently active.
        In the current implementation it will be inferred from the RELION_HOME
        variable, so it should contain the version number in it. """
        home = cls.__getRelionHome()
        for v in cls.getSupportedRelionVersions():
            if v in home:
                return v
        return ''

    @classmethod
    def getActiveVersion(cls):
        """ Return the version of the Relion binaries that is currently active.
        In the current implementation it will be inferred from the RELION_HOME
        variable, so it should contain the version number in it. """
        home = cls.getHome()
        for v in cls.getSupportedVersions():
            if v in home:
                return v
        return ''

    @classmethod
    def getSupportedRelionVersions(cls):
        """ Return the list of supported binary versions. """
        return [V3_0]

    @classmethod
    def getSupportedVersions(cls):
        """ Return the list of supported binary versions. """
        return ['0.1']

    @classmethod
    def defineBinaries(cls, env):
        libSphPath = cls.getHome('alignLib/SpharmonicKit27/libsphkit.so')
        libFrmPath = cls.getHome('alignLib/frm/swig/_swig_frm.so')
        commands = ('python alignLib/compile.py ; ln -sf %s ../../lib/ '
                    '; ln -sf %s ../../lib/' %(libSphPath, libFrmPath))
        target = libFrmPath
        url= 'https://github.com/mcgill-femr/cryomethods/archive/v0.1.tar.gz'
        env.addPackage('cryomethods', version='0.1',
                       url=url,
                       commands=[(commands, target)])
        ## PIP PACKAGES ##
        def addPipModule(moduleName, *args, **kwargs):
            """ To try to add certain pipModule.
                If it fails due to it is already add by other plugin or Scipion,
                  just returns its name to use it as a dependency.
                Raise the exception if unknown error is gotten.
            """
            try:
                return env.addPipModule(moduleName, *args, **kwargs)._name
            except Exception as e:
                if "Duplicated target '%s'" % moduleName == str(e):
                    return moduleName
                else:
                    raise Exception(e)
        #
        # joblib = addPipModule('joblib', '0.11', target='joblib*')
        #
        # ## --- DEEP LEARNING TOOLKIT --- ##
        scipy = addPipModule('scipy', '0.14.0', default=False,
                                deps=['lapack', 'matplotlib'])
        cython = addPipModule('cython', '0.22', target='Cython-0.22*',
                                 default=False)
        scikit_learn = addPipModule('scikit-learn', '0.20.0',
                                       target='scikit_learn*',
                                       default=True, deps=[scipy, cython])
        # unittest2 = addPipModule('unittest2', '0.5.1', target='unittest2*',
        #                             default=False)
        # h5py = addPipModule('h5py', '2.8.0rc1', target='h5py*',
        #                        default=False, deps=[unittest2])
        # cv2 = addPipModule('opencv-python', "3.4.2.17",
        #                       target="cv2", default=False)

pyworkflow.em.Domain.registerPlugin(__name__)
