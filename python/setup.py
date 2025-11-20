from setuptools import setup

from mypyc.build import mypycify

setup(
    ext_modules = mypycify(
        [
            'reprospect/test/matchers.py',
            'reprospect/test/matchers_impl.py',
            'reprospect/test/sass.py',
            'reprospect/tools/architecture.py',
            'reprospect/tools/binaries/cuobjdump.py',
            'reprospect/tools/binaries/demangle.py',
            'reprospect/tools/binaries/elf.py',
            'reprospect/tools/ncu.py',
            'reprospect/tools/sass/decode.py',
            'reprospect/utils/cmake.py',
            'reprospect/utils/detect.py',
            'reprospect/utils/ldd.py',
            'reprospect/utils/nvcc.py',
            'reprospect/utils/subprocess_helpers.py',
        ],
        verbose = True,
        strict_dunder_typing = True,
    ),
)
