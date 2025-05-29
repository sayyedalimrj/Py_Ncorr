# ─────────────────────────────────────────────────────────────────────────────
#  setup.py – builds the two C++/pybind11 extensions in ncorr_python_webapp
#  * Adds  cpp_src/shims  (where our dummy mex.h lives) to the include path
#  * No other functional changes
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import subprocess

# pybind11 might not be pre-installed in build isolation; fall back gracefully
try:
    import pybind11           # noqa: E402
except ModuleNotFoundError:    # < pip --no-build-isolation > path
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=2.12"])
    import pybind11           # noqa: E402


class BuildExt(build_ext):
    """Inject compiler flags for C++17 & OpenMP across platforms."""

    c_opts = {
        "msvc": ["/std:c++17", "/openmp", "/O2"],
        "unix": ["-std=c++17", "-O3", "-fopenmp", "-DNCORR_OPENMP"],
    }
    l_opts = {
        "msvc": [],
        "unix": ["-fopenmp"],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link = self.l_opts.get(ct, [])
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link
        super().build_extensions()


ROOT = Path(__file__).parent.resolve()

include_dirs = [
    pybind11.get_include(),
    (ROOT / "cpp_src" / "ncorr_lib").as_posix(),
    (ROOT / "cpp_src" / "ncorr_alg").as_posix(),
    (ROOT / "cpp_src" / "shims").as_posix(),        #  ← NEW: dummy mex.h lives here
]

# ─── Shared library (.cpp) list -------------------------------------------
lib_src = [
    "cpp_src/ncorr_lib/standard_datatypes.cpp",
    "cpp_src/ncorr_lib/ncorr_datatypes.cpp",
    "cpp_src/ncorr_lib/ncorr_lib.cpp",
]

# ─── Core extension --------------------------------------------------------
core_sources = ["bindings/py_ncorr_core_bindings.cpp"] + lib_src

core_ext = Extension(
    "ncorr_app._ext._ncorr_cpp_core",
    sources=core_sources,
    include_dirs=include_dirs,
    language="c++",
)

# ─── Algorithms extension --------------------------------------------------
alg_sources = (
    ["bindings/py_ncorr_algorithms_bindings.cpp"]
    + [f"cpp_src/ncorr_alg/{name}" for name in (
        "ncorr_alg_formmask.cpp",
        "ncorr_alg_formregions.cpp",
        "ncorr_alg_formboundary.cpp",
        "ncorr_alg_formthreaddiagram.cpp",
        "ncorr_alg_formunion.cpp",
        "ncorr_alg_extrapdata.cpp",
        "ncorr_alg_adddisp.cpp",
        "ncorr_alg_convert.cpp",
        "ncorr_alg_dispgrad.cpp",
        "ncorr_alg_testopenmp.cpp",
        "ncorr_alg_calcseeds.cpp",
        "ncorr_alg_rgdic.cpp",
    )]
    + lib_src                      # algorithms also need core lib sources
)

alg_ext = Extension(
    "ncorr_app._ext._ncorr_cpp_algs",
    sources=alg_sources,
    include_dirs=include_dirs,
    language="c++",
)

# ─── Setup call ------------------------------------------------------------
setup(
    name="ncorr_python_webapp",
    version="0.1.0",
    description="Python + C++ port of Ncorr 2-D DIC with web front-end",
    packages=find_packages(where="."),
    package_data={"ncorr_app": ["_ext/*"]},
    ext_modules=[core_ext, alg_ext],
    cmdclass={"build_ext": BuildExt},
    install_requires=Path("requirements.txt").read_text().splitlines(),
    python_requires=">=3.9",
)
