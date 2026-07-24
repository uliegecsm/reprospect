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
            'reprospect/testing/features.py',
            'reprospect/testing/binaries/sass/controlflow/block.py',
            'reprospect/testing/binaries/sass/instruction/address.py',
            'reprospect/testing/binaries/sass/instruction/atomic.py',
            'reprospect/testing/binaries/sass/instruction/branch.py',
            'reprospect/testing/binaries/sass/instruction/constant.py',
            'reprospect/testing/binaries/sass/instruction/floating.py',
            'reprospect/testing/binaries/sass/instruction/half.py',
            'reprospect/testing/binaries/sass/instruction/immediate.py',
            'reprospect/testing/binaries/sass/instruction/instruction.py',
            'reprospect/testing/binaries/sass/instruction/integer.py',
            'reprospect/testing/binaries/sass/instruction/load.py',
            'reprospect/testing/binaries/sass/instruction/memory.py',
            'reprospect/testing/binaries/sass/instruction/operand.py',
            'reprospect/testing/binaries/sass/instruction/pattern.py',
            'reprospect/testing/binaries/sass/instruction/register.py',
            'reprospect/testing/binaries/sass/instruction/store.py',
            'reprospect/testing/binaries/sass/instruction/validate.py',
            'reprospect/testing/binaries/sass/operation/add_int128.py',
            'reprospect/testing/binaries/sass/operation/add_int32.py',
            'reprospect/testing/binaries/sass/operation/cas.py',
            'reprospect/testing/binaries/sass/operation/convert_fp_to_int.py',
            'reprospect/testing/binaries/sass/operation/convert_fp32_to_fp16.py',
            'reprospect/testing/binaries/sass/operation/convert_int_to_fp.py',
            'reprospect/testing/binaries/sass/operation/move32.py',
            'reprospect/testing/binaries/sass/sequence/sequence.py',
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
            'reprospect/tools/nsys/cacher.py',
            'reprospect/tools/nsys/report.py',
            'reprospect/tools/nsys/session.py',
            'reprospect/tools/binaries/sass/controlflow.py',
            'reprospect/tools/binaries/sass/decoder.py',
            'reprospect/utils/cmake.py',
            'reprospect/utils/compile_command.py',
            'reprospect/utils/detect.py',
            'reprospect/utils/ldd.py',
            'reprospect/utils/nvcc.py',
            'reprospect/utils/rich_helpers.py',
            'reprospect/utils/subprocess_helpers.py',
            'reprospect/utils/types.py',
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
