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

from .constants import RELION_HOME, CRYOMETHODS_HOME, V2_0, V2_1, V3_0

# from bibtex import _bibtex # Load bibtex dict with references
_logo = "cryomethods_logo.png"
_references = []


class Plugin(pyworkflow.em.Plugin):
    _homeVar = CRYOMETHODS_HOME
    _pathVars = [CRYOMETHODS_HOME]
    _supportedVersions = []

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(CRYOMETHODS_HOME, 'cryomethods-0.1')
        cls._defineEmVar(RELION_HOME, 'relion-2.1')


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

        for lPath in libPath:
            if not lPath in os.environ['LD_LIBRARY_PATH']:
               environ.update({'LD_LIBRARY_PATH': lPath},
                              position=pwutils.Environ.BEGIN)

        for pPath in pythonPath:
            if not pPath in os.environ['PYTHONPATH']:
                environ.update({'PYTHONPATH': pPath},
                               position=pwutils.Environ.BEGIN)
        return environ


    @classmethod
    def __getRelionHome(cls, *paths):
        """ Return the binary home path and possible some subfolders. """
        return os.path.join(os.environ[RELION_HOME], *paths)


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
        home = cls.__getHome()
        for v in cls.getSupportedVersions():
            if v in home:
                return v
        return ''

    @classmethod
    def isVersion2Relion(cls):
        return cls.getActiveRelionVersion().startswith("2.")


    @classmethod
    def getSupportedRelionVersions(cls):
        """ Return the list of supported binary versions. """
        return [V2_0, V2_1]

    @classmethod
    def getSupportedVersions(cls):
        """ Return the list of supported binary versions. """
        return ['0.1']

    @classmethod
    def defineBinaries(cls, env):
        relion_commands = [('./INSTALL.sh -j %d' % env.getProcessors(),
                            ['relion_build.log',
                             'bin/relion_refine'])]

        env.addPackage('relion', version='1.4',
                       tar='relion-1.4.tgz',
                       commands=relion_commands)

        env.addPackage('relion', version='1.4f',
                       tar='relion-1.4_float.tgz',
                       commands=relion_commands)

        # Define FFTW3 path variables
        relion_vars = {'FFTW_LIB': env.getLibFolder(),
                       'FFTW_INCLUDE': env.getIncludeFolder()}

        relion2_commands = [('cmake -DGUI=OFF -DCMAKE_INSTALL_PREFIX=./ .', []),
                            ('make -j %d' % env.getProcessors(),
                             ['bin/relion_refine'])]

        env.addPackage('relion', version='2.0',
                       tar='relion-2.0.4.tgz',
                       commands=relion2_commands,
                       updateCuda=True,
                       vars=relion_vars)

        env.addPackage('relion', version='2.1',
                       tar='relion-2.1.tgz',
                       commands=relion2_commands,
                       updateCuda=True,
                       vars=relion_vars,
                       default=True)

        ## PIP PACKAGES ##
        def tryAddPipModule(moduleName, *args, **kwargs):
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

        joblib = tryAddPipModule('joblib', '0.11', target='joblib*')

        ## --- DEEP LEARNING TOOLKIT --- ##
        scipy = tryAddPipModule('scipy', '0.14.0', default=False,
                                deps=['lapack', 'matplotlib'])
        cython = tryAddPipModule('cython', '0.22', target='Cython-0.22*',
                                 default=False)
        scikit_learn = tryAddPipModule('scikit-learn', '0.19.1',
                                       target='scikit_learn*',
                                       default=False, deps=[scipy, cython])
        unittest2 = tryAddPipModule('unittest2', '0.5.1', target='unittest2*',
                                    default=False)
        h5py = tryAddPipModule('h5py', '2.8.0rc1', target='h5py*',
                               default=False, deps=[unittest2])
        cv2 = tryAddPipModule('opencv-python', "3.4.2.17",
                              target="cv2", default=False)

pyworkflow.em.Domain.registerPlugin(__name__)
