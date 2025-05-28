from setuptools import setup, Extension, find_packages
import pybind11
import sys

cpp_args = ['-std=c++11', '-O3'] # Or -std=c++17 etc.
linker_args = []

# Platform specific OpenMP flags
if sys.platform == 'darwin': # macOS
    cpp_args.append('-mmacosx-version-min=10.14')
    # Check if compiler is clang or gcc, as flags differ
    # For Apple clang, OpenMP might need specific handling (e.g. libomp from brew)
    # Assuming a compiler that supports -fopenmp (like modern clang or gcc from brew)
    cpp_args.extend(['-Xpreprocessor', '-fopenmp'])
    linker_args.extend(['-lomp'])
elif sys.platform == 'linux': # Linux
    cpp_args.append('-fopenmp')
    linker_args.append('-fopenmp') # For GCC, -fopenmp handles both compilation and linking
elif sys.platform == 'win32': # Windows (MSVC)
    cpp_args.append('/openmp')
    # MSVC linker handles OpenMP automatically when /openmp is used for compilation

ext_modules = [
    Extension(
        'ncorr_app._ext._ncorr_cpp_test_module',
        ['bindings/test_binding.cpp'],
        include_dirs=[
            pybind11.get_include()
        ],
        language='c++',
        extra_compile_args=cpp_args,
        extra_link_args=linker_args, # Might not be needed for test_module if no OpenMP
    ),
    Extension(
        'ncorr_app._ext._ncorr_cpp_core',
        sources=[
            'bindings/py_ncorr_core_bindings.cpp',
            'cpp_src/ncorr_lib/standard_datatypes.cpp',
            'cpp_src/ncorr_lib/ncorr_datatypes.cpp',
            'cpp_src/ncorr_lib/ncorr_lib.cpp',
        ],
        include_dirs=[
            pybind11.get_include(),
            'cpp_src/ncorr_lib/',
        ],
        language='c++',
        extra_compile_args=cpp_args, # Core lib might not need OpenMP compile flags itself
        extra_link_args=linker_args, # Core lib might not need OpenMP link flags itself
    ),
    Extension(
        'ncorr_app._ext._ncorr_cpp_algs',  # Output module name
        sources=[
            'bindings/py_ncorr_algorithms_bindings.cpp',
            # Include all algorithm .cpp files
            'cpp_src/ncorr_alg/ncorr_alg_formmask.cpp',
            'cpp_src/ncorr_alg/ncorr_alg_formregions.cpp',
            'cpp_src/ncorr_alg/ncorr_alg_formboundary.cpp',
            'cpp_src/ncorr_alg/ncorr_alg_formthreaddiagram.cpp',
            'cpp_src/ncorr_alg/ncorr_alg_formunion.cpp',
            'cpp_src/ncorr_alg/ncorr_alg_extrapdata.cpp',
            'cpp_src/ncorr_alg/ncorr_alg_adddisp.cpp',
            'cpp_src/ncorr_alg/ncorr_alg_convert.cpp',
            'cpp_src/ncorr_alg/ncorr_alg_dispgrad.cpp',
            'cpp_src/ncorr_alg/ncorr_alg_testopenmp.cpp',
            'cpp_src/ncorr_alg/ncorr_alg_calcseeds.cpp',
            'cpp_src/ncorr_alg/ncorr_alg_rgdic.cpp',
            # Also include dependent library .cpp files if not linked as a separate library
            'cpp_src/ncorr_lib/standard_datatypes.cpp',
            'cpp_src/ncorr_lib/ncorr_datatypes.cpp',
            'cpp_src/ncorr_lib/ncorr_lib.cpp',
        ],
        include_dirs=[
            pybind11.get_include(),
            'cpp_src/ncorr_lib/', # For standard_datatypes.h, ncorr_datatypes.h, ncorr_lib.h
            'cpp_src/ncorr_alg/'  # If algorithm files have their own .h files (currently they seem self-contained .cpp)
        ],
        language='c++',
        extra_compile_args=cpp_args + ['-DNCORR_OPENMP'], # Add OpenMP definition for these files
        extra_link_args=linker_args,
    )
]

setup(
    name='ncorr_app',
    version='0.1.0',
    author='Your Name/Team Name',
    author_email='your.email@example.com',
    description='Python port of Ncorr for 2D Digital Image Correlation with a web interface.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='<your_project_url_here>',
    packages=find_packages(exclude=['tests*']),
    ext_modules=ext_modules,
    install_requires=[
        'numpy',
        'scipy',
        'opencv-python',
        'Pillow',
        'pybind11>=2.6',
        'Flask',
        'Flask-RESTful',
        'gunicorn',
        'celery',
        'redis',
        'PyYAML',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.8',
    zip_safe=False,
)