import pathlib
import subprocess
import typing

import pytest

from reprospect.test.features import PTX
from reprospect.test.sass.instruction.half import Fp16MulMatcher, Fp16FusedMulAddMatcher
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.sass import Decoder

from tests.python.parameters import Parameters, PARAMETERS

class TestFp16FusedMulAddMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.half.Fp16FusedMulAddMatcher`.
    """
    PTX : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'hfma2.ptx'

    def test_individual(self) -> None:
        matcher = Fp16FusedMulAddMatcher(packed=False)

        assert (matched := matcher.match('HFMA2 R22, R22.H0_H0, R24.H0_H0, R27.H0_H0')) is not None
        assert matched.operands == ('R22', 'R22.H0_H0', 'R24.H0_H0', 'R27.H0_H0')

        assert (matched := matcher.match('HFMA2 R0, R0.H0_H0, R2.H0_H0, 3, 3')) is not None
        assert matched.operands == ('R0', 'R0.H0_H0', 'R2.H0_H0', '3', '3')

    def test_packed(self) -> None:
        matcher = Fp16FusedMulAddMatcher(packed=True)

        assert (matched := matcher.match(inst = 'HFMA2 R7, R2, R2, -RZ')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R7'
        assert matched.operands == ('R7', 'R2', 'R2', '-RZ')

        assert matcher.match('HFMA2 R19, -RZ, RZ, 0, 0') is not None
        assert matcher.match('HFMA2.MMA R57, -RZ, RZ, 0, 0') is not None
        assert matcher.match('HFMA2.MMA R20, -RZ, RZ, 1.875, 0') is not None

        assert (matched := matcher.match('HFMA2 R24, -RZ, RZ, 0, 1.370906829833984375e-06')) is not None
        assert matched.operands == ('R24', '-RZ', 'RZ', '0', '1.370906829833984375e-06')

    def test_any(self) -> None:
        matcher = Fp16FusedMulAddMatcher(packed=None)

        assert matcher.match('HFMA2 R22, R22.H0_H0, R24.H0_H0, R27.H0_H0') is not None
        assert matcher.match('HFMA2 R19, -RZ, RZ, 0, 0') is not None

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str, scope = 'class')
    def test_from_ptx(self, request, workdir : pathlib.Path, parameters : Parameters) -> None:
        """
        Compile the PTX from :py:attr:`PTX`.
        """
        cubin = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cubin'
        ptx   = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.ptx'

        ptx_isa_version = PTX(arch = parameters.arch).min_isa_version
        ptx.write_text(self.PTX.read_text().format(
            version = f'{ptx_isa_version.major}.{ptx_isa_version.minor}',
            cc = parameters.arch.compute_capability.as_int,
        ))

        subprocess.check_call(('ptxas', '--verbose', f'-arch={parameters.arch.as_sm}', ptx, '-o', cubin))

        cuobjdump = CuObjDump(file = cubin, arch = parameters.arch)

        decoder = Decoder(code = cuobjdump.functions['hfma2'].code)

        matcher = Fp16FusedMulAddMatcher(packed=False)
        matched = tuple(instruction for instruction in decoder.instructions if matcher.match(instruction.instruction))
        counter = sum(1 for instruction in decoder.instructions if instruction.instruction.startswith('HFMA2'))
        assert 2 == len(matched) <= counter

class TestFp16MulMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.half.Fp16MulMatcher`.
    """
    def test_individual(self) -> None:
        assert (matched := Fp16MulMatcher(packed = False).match(inst = 'HMUL2 R0, R2.H0_H0, R3.H0_H0')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R0'
        assert matched.operands[-1] == 'R3.H0_H0'

    def test_packed(self) -> None:
        assert (matched := Fp16MulMatcher(packed = True).match(inst = 'HMUL2 R0, R2, R3')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R0'
        assert matched.operands[-1] == 'R3'

    def test_any(self) -> None:
        assert (matched := Fp16MulMatcher().match(inst = 'HMUL2 R0, R2.H1_H1, R3.H0_H0')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R0'
        assert matched.operands == ('R0', 'R2.H1_H1', 'R3.H0_H0')

        assert (matched := Fp16MulMatcher().match(inst = 'HMUL2 R0, R2, R3')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R0'
        assert matched.operands == ('R0', 'R2', 'R3')
