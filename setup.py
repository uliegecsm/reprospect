import logging
import pprint

from setuptools import setup
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


def enable_mypyc(dist: Distribution) -> None:
    """
    Enable `mypyc` as extension modules.

    References:

    * https://mypyc.readthedocs.io/en/latest/getting_started.html#using-setup-py
    """
    from mypyc.build import mypycify

    ext_modules = mypycify(
        [
            'reprospect/test/features.py',
            'reprospect/test/sass/composite_impl.py',
            'reprospect/test/sass/composite.py',
            'reprospect/test/sass/controlflow/block.py',
            'reprospect/test/sass/instruction/address.py',
            'reprospect/test/sass/instruction/constant.py',
            'reprospect/test/sass/instruction/half.py',
            'reprospect/test/sass/instruction/instruction.py',
            'reprospect/test/sass/instruction/integer.py',
            'reprospect/test/sass/instruction/memory.py',
            'reprospect/test/sass/instruction/operand.py',
            'reprospect/test/sass/instruction/pattern.py',
            'reprospect/test/sass/instruction/register.py',
            'reprospect/test/sass/matchers/add_int128.py',
            'reprospect/test/sass/matchers/add_int32.py',
            'reprospect/test/sass/matchers/convert_fp32_to_fp16.py',
            'reprospect/test/sass/matchers/move32.py',
            'reprospect/tools/architecture.py',
            'reprospect/tools/binaries/cuobjdump.py',
            'reprospect/tools/binaries/demangle.py',
            'reprospect/tools/binaries/elf.py',
            'reprospect/tools/binaries/nvdisasm.py',
            'reprospect/tools/binaries/symtab.py',
            'reprospect/tools/ncu/cacher.py',
            'reprospect/tools/ncu/metrics.py',
            'reprospect/tools/ncu/report.py',
            'reprospect/tools/ncu/session.py',
            'reprospect/tools/nsys.py',
            'reprospect/tools/sass/controlflow.py',
            'reprospect/tools/sass/decode.py',
            'reprospect/utils/cmake.py',
            'reprospect/utils/detect.py',
            'reprospect/utils/ldd.py',
            'reprospect/utils/nvcc.py',
            'reprospect/utils/subprocess_helpers.py',
        ],
        verbose=True,
        strict_dunder_typing=True,
    )
    logging.info(f'The following mypyc extension modules will be used:\n{pprint.pformat(ext_modules)}')
    dist.ext_modules = ext_modules

class bdist_wheel(_bdist_wheel):
    """
    Compile with `mypyc` when building wheels.
    """
    def finalize_options(self) -> None:
        logging.info('Building a built distribution (bdist).')
        enable_mypyc(self.distribution)
        super().finalize_options()
        assert self.root_is_pure is False

setup(
    cmdclass={
        "bdist_wheel": bdist_wheel,
    },
)
