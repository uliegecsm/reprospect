import attrs

from reprospect.test.sass.composite_impl import InSequenceMatcher
from reprospect.test.sass.instruction import InstructionMatch
from reprospect.tools.sass.controlflow import BasicBlock, Graph

@attrs.define(frozen=True, slots=True)
class BlockMatcher:
    """
    Match the first block of a :py:class:`reprospect.tools.sass.controlflow.Graph` that
    matches :py:attr:`matcher`.
    """
    matcher: InSequenceMatcher = attrs.field(converter=lambda x: x if isinstance(x, InSequenceMatcher) else InSequenceMatcher(x))

    def match(self, cfg: Graph) -> tuple[BasicBlock, list[InstructionMatch]] | None:
        for block in cfg.blocks:
            if (matched := self.matcher.match(instructions=block.instructions)) is not None:
                return block, matched
        return None

    def explain(self, cfg: Graph) -> str:
        return f'{self.matcher} did not match any block of {cfg}.'

    def assert_matches(self, cfg: Graph) -> tuple[BasicBlock, list[InstructionMatch]]:
        if (matched := self.match(cfg=cfg)) is None:
            raise RuntimeError(self.explain(cfg=cfg))
        return matched
