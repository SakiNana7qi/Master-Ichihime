from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import os
from pathlib import Path

ROOT = Path(__file__).parent
SRC = str(ROOT / "fastops" / "fastops.pyx")

setup(
    name="mahjong_fastops",
    ext_modules=cythonize(
        [SRC],
        compiler_directives={"language_level": 3, "boundscheck": False, "wraparound": False},
    ),
    include_dirs=[np.get_include()],
)


