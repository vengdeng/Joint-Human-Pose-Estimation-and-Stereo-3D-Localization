from setuptools import setup

import numpy
from Cython.Build import cythonize


# extract version from __init__.py
with open('openpifpaf/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]


setup(
    name='openpifpaf',
    version=VERSION,
    packages=[
        'openpifpaf',
        'openpifpaf.decoder',
        'openpifpaf.encoder',
        'openpifpaf.network',
    ],
    description='Joint Human Pose Estimation and Stereo Localization',
    ext_modules=cythonize('openpifpaf/functional.pyx',
                          include_path=[numpy.get_include()],
                          annotate=True,
                          compiler_directives={'language_level': 3}),

    install_requires=[
        'matplotlib',
        'pysparkling',  # for log analysis
        'python-json-logger',
        'scipy',
        'torch>=1.0.0',
        'torchvision',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
        'train': [
            'pycocotools',  # pre-install cython
        ],
    },
)
