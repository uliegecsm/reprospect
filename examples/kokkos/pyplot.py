import sys

import matplotlib.artist
import matplotlib.legend
import matplotlib.legend_handler
import matplotlib.text
import matplotlib.transforms

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class HandlerText(matplotlib.legend_handler.HandlerBase):
    @override
    def create_artists(self,
        legend: matplotlib.legend.Legend,
        orig_handle: matplotlib.artist.Artist,
        xdescent: float, ydescent: float,
        width: float, height: float,
        fontsize: float, trans: matplotlib.transforms.Transform,
    ) -> list[matplotlib.artist.Artist]:
        if not isinstance(orig_handle, matplotlib.text.Text):
            raise TypeError('Wrong usage.')
        orig_handle.set_transform(trans)
        orig_handle.set_position((xdescent, ydescent))
        orig_handle.set_fontsize(fontsize)
        return [orig_handle]
