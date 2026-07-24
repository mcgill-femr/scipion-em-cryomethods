"""Build the native alignLib extension used by CryoMethods.

This builder deliberately does not depend on the historical ``compile.py``:
that script silently ignored make/linker failures and assumed a system Python
installation.  All paths are discovered from the Python interpreter running
the build (the Scipion interpreter during plugin installation).
"""

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def _run(command, cwd, env=None):
    print("[cryomethods]", " ".join(str(x) for x in command), flush=True)
    subprocess.check_call(command, cwd=str(cwd), env=env)


def _find_file(name, directories):
    for directory in directories:
        if not directory:
            continue
        candidate = Path(directory) / name
        if candidate.exists():
            return candidate.resolve()
    return None


def build(root):
    root = Path(root).resolve()
    align = root / "alignLib"
    sph = align / "SpharmonicKit27"
    frm = align / "frm"
    swig = frm / "swig"
    src = frm / "src"

    required = [sph / "Makefile", src / "Makefile", swig / "frm.i"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("alignLib source is incomplete: %s" % ", ".join(missing))

    for tool in ("make", "gcc"):
        if shutil.which(tool) is None:
            raise RuntimeError("Required build tool not found: %s" % tool)
    swig_bin = shutil.which("swig")
    if swig_bin is None:
        candidate = Path(sys.executable).parent / "swig"
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            swig_bin = str(candidate)
    if swig_bin is None:
        raise RuntimeError("Required build tool not found: swig (install Python package 'swig')")

    try:
        import numpy
        numpy_include = Path(numpy.get_include())
    except ImportError as ex:
        raise RuntimeError("Numpy is required to build alignLib") from ex

    python_include = Path(sysconfig.get_paths()["include"])
    fftw_include = _find_file(
        "fftw3.h",
        [os.environ.get("FFTW_INCLUDE_PATH"),
         os.environ.get("SCIPION_HOME", "") + "/software/include",
         "/usr/include", "/usr/local/include"])
    if fftw_include is None:
        raise RuntimeError("FFTW3 headers not found (install FFTW3 development files)")

    libdirs = []
    for value in (os.environ.get("FFTW_LIB_PATH"),
                  sysconfig.get_config_var("LIBDIR"),
                  os.environ.get("SCIPION_HOME", "") + "/software/lib",
                  "/usr/lib", "/usr/lib64", "/usr/local/lib"):
        if value and Path(value).is_dir() and value not in libdirs:
            libdirs.append(value)
    fftw_lib = next((Path(d) / "libfftw3.so" for d in libdirs
                     if (Path(d) / "libfftw3.so").exists()), None)
    if fftw_lib is None:
        # Debian-like systems often expose only the versioned development link.
        fftw_lib = next((p for d in libdirs
                         for p in Path(d).glob("libfftw3.so*")
                         if p.is_file()), None)
    if fftw_lib is None:
        raise RuntimeError("FFTW3 library not found (install FFTW3 development files)")
    fftw_dir = fftw_lib.parent

    py_libdir = sysconfig.get_config_var("LIBDIR") or ""
    py_ldlibrary = sysconfig.get_config_var("LDLIBRARY") or ""
    py_lib = _find_file(py_ldlibrary, [py_libdir]) if py_ldlibrary else None
    if py_lib is None:
        # The extension does not need Python symbols at link time on Linux,
        # but retain the path when a shared libpython is available.
        py_libdir = ""

    env = os.environ.copy()
    env.update({
        "PYTHON_INCLUDE_PATH": str(python_include),
        "NUMPY_INCLUDE_PATH": str(numpy_include),
        "FFTW_INCLUDE_PATH": str(fftw_include.parent),
        "FFTW_LIB_PATH": str(fftw_dir),
    })

    _run(["make", "-C", str(sph), "all"], root, env)
    _run(["make", "-C", str(src), "lib"], root, env)

    _run([swig_bin, "-python", "frm.i"], swig, env)
    _run(["gcc", "-O3", "-fPIC", "-c", "frm.c", "frm_wrap.c",
          "-I" + str(src), "-I" + str(python_include),
          "-I" + str(numpy_include), "-I" + str(fftw_include.parent),
          "-I" + str(sph),
          # SWIG 4 still emits these Python-2 checks for this legacy
          # interface.  They are only used for optional type diagnostics;
          # defining them as false keeps the wrapper loadable on Python 3.
          "-DPyFile_Check(x)=0", "-DPyInstance_Check(x)=0"], swig, env)

    link = ["gcc", "-shared", "frm.o", "frm_wrap.o",
            "-L" + str(fftw_dir), "-lfftw3", "-lm",
            "../src/lib_vio.o", "../src/lib_pio.o", "../src/lib_std.o",
            "../src/lib_eul.o", "../src/lib_pwk.o", "../src/lib_vec.o",
            "../src/lib_vwk.o", "../src/lib_tim.o",
            "-L" + str(sph), "-lsphkit",
            "-Wl,-rpath,$ORIGIN/../../SpharmonicKit27",
            "-o", "_swig_frm.so"]
    if py_libdir:
        link[2:2] = ["-L" + py_libdir]
    _run(link, swig, env)

    extension = swig / "_swig_frm.so"
    if not extension.exists():
        raise RuntimeError("alignLib build produced no %s" % extension)
    test_env = env.copy()
    test_env["PYTHONPATH"] = os.pathsep.join(
        [str(align), str(swig), test_env.get("PYTHONPATH", "")])
    test_env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(sph), str(fftw_dir), test_env.get("LD_LIBRARY_PATH", "")])
    _run([sys.executable, "-c", "import frm; import swig_frm"],
         root, test_env)
    print("[cryomethods] alignLib compiled successfully: %s" % extension)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.getcwd(),
                        help="cryomethods package root (default: current directory)")
    args = parser.parse_args()
    try:
        build(args.root)
    except Exception as ex:
        print("[cryomethods] alignLib build failed: %s" % ex, file=sys.stderr)
        raise
