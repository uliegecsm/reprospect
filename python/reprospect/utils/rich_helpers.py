import abc

import rich.console
import rich.tree

class TreeMixin(metaclass = abc.ABCMeta):
    """
    Define :py:meth:`__str__` based on the :py:class:`rich.tree.Tree` representation from :py:meth:`to_tree`.
    """
    @abc.abstractmethod
    def to_tree(self) -> rich.tree.Tree:
        """
        Convert to a :py:class:`rich.tree.Tree`.
        """

    def __str__(self) -> str:
        """
        Use :py:class:`rich.console.Console` in capture mode.
        """
        with rich.console.Console() as console, console.capture() as capture:
            console.print(self.to_tree(), no_wrap = True)
        return capture.get()
