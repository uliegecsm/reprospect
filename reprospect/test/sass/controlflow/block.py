import abc
import sys
import typing

import mypy_extensions

from reprospect.test.sass.composite_impl import InSequenceMatcher, SequenceMatcher
from reprospect.test.sass.instruction import InstructionMatch, InstructionMatcher
from reprospect.tools.sass.controlflow import BasicBlock, Graph

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@mypy_extensions.mypyc_attr(allow_interpreted_subclasses=False)
class BasicBlockMatcherBase(abc.ABC):
    """
    Base class for matchers that find blocks in a :py:class:`reprospect.tools.sass.controlflow.Graph`.
    """
    @abc.abstractmethod
    def match(self, cfg: Graph) -> tuple[BasicBlock, list[InstructionMatch]] | None:
        ...

    def explain(self, cfg: Graph) -> str:
        return f'{self!r} did not match any block of {cfg!r}.'

    @typing.final
    def assert_matches(self, cfg: Graph) -> tuple[BasicBlock, list[InstructionMatch]]:
        if (matched := self.match(cfg=cfg)) is None:
            raise RuntimeError(self.explain(cfg=cfg))
        return matched

class BasicBlockMatcher(BasicBlockMatcherBase):
    """
    Match the first block that matches :py:attr:`matcher`.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher',)

    def __init__(self, matcher: InstructionMatcher|SequenceMatcher) -> None:
        self.matcher: typing.Final[InSequenceMatcher] = matcher if isinstance(matcher, InSequenceMatcher) else InSequenceMatcher(matcher)

    @override
    def match(self, cfg: Graph) -> tuple[BasicBlock, list[InstructionMatch]] | None:
        for block in cfg.blocks:
            if (matched := self.matcher.match(instructions=block.instructions)) is not None:
                return block, matched
        return None

    @override
    def explain(self, cfg: Graph) -> str:
        return f'{self.matcher.matcher!r} did not match any block of {cfg!r}.'

class BasicBlockWithParentMatcher(BasicBlockMatcherBase):
    """
    Match the first block of a :py:class:`reprospect.tools.sass.controlflow.Graph` that
    matches :py:attr:`matcher` and has an incoming edge from :py:attr:`parent`.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher', 'parent')

    def __init__(self, matcher: InstructionMatcher|SequenceMatcher, parent: BasicBlock) -> None:
        self.matcher: typing.Final[InSequenceMatcher] = matcher if isinstance(matcher, InSequenceMatcher) else InSequenceMatcher(matcher)
        self.parent: typing.Final[BasicBlock] = parent

    @override
    def match(self, cfg: Graph) -> tuple[BasicBlock, list[InstructionMatch]] | None:
        for block in cfg.edges[self.parent]:
            if (matched := self.matcher.match(instructions=block.instructions)) is not None:
                return block, matched
        return None

    @override
    def explain(self, cfg: Graph) -> str:
        return f'{self.matcher.matcher!r} did not match any child of {self.parent!r}.'
