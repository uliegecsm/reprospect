from setuptools import setup

from mypyc.build import mypycify

setup(
    ext_modules = mypycify(
        [
            'reprospect/tools/sass.py',
            'reprospect/utils/cmake.py',
            'reprospect/utils/detect.py',
            'reprospect/utils/ldd.py',
        ],
        verbose = True,
    ),
)
