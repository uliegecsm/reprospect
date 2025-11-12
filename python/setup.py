from setuptools import setup

from mypyc.build import mypycify

setup(
    ext_modules = mypycify(
        [
            'reprospect/test/matchers.py',
            'reprospect/test/sass.py',
            'reprospect/tools/architecture.py',
            'reprospect/tools/binaries/cuobjdump.py',
            'reprospect/tools/binaries/demangle.py',
            'reprospect/tools/ncu.py',
            'reprospect/tools/sass.py',
            'reprospect/utils/cmake.py',
            'reprospect/utils/detect.py',
            'reprospect/utils/ldd.py',
            'reprospect/utils/subprocess_helpers.py',
        ],
        verbose = True,
    ),
)
