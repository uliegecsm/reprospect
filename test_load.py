import argparse
import logging
import pathlib

import numpy
import matplotlib.pyplot

def main(colors: pathlib.Path, iterations: pathlib.Path, output:pathlib.Path) -> None:
    """
    Read data from `colors` and `iterations`, assuming they contain 2 sizes of type
    :py:class:numpy.uint64` and flattened values of type :py:class:`numpy.uint32`.
    """
    with colors.open('rb') as fin:
        dim_0 = numpy.fromfile(fin, dtype=numpy.uint64, count=1)[0]
        dim_1 = numpy.fromfile(fin, dtype=numpy.uint64, count=1)[0]
        colors_data = numpy.fromfile(fin, dtype=numpy.uint32).reshape((dim_1, dim_0))

    with iterations.open('rb') as fin:
        dim_0 = numpy.fromfile(fin, dtype=numpy.uint64, count=1)[0]
        dim_0 = numpy.fromfile(fin, dtype=numpy.uint64, count=1)[0]
        iterations_data = numpy.fromfile(fin, dtype=numpy.uint32).reshape((dim_1, dim_0))

    assert colors_data.shape == iterations_data.shape

    # Convert to float for proper normalization
    iterations = iterations.astype(np.float64)

    # ---------- Plot ----------
    fig, axes = plt.subplots(nrows=1, ncols=2,
                            figsize=(20, 10),
                            constrained_layout=True)

    fig.patch.set_facecolor('black')

    # First image (categorical colors)
    im0 = axes[0].imshow(
        data,
        origin='lower',
        extent=[-2, 2, -2, 2]
    )

    axes[0].set_title("Root Index", color="white")

    # Add colorbar for first image
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar0.ax.get_yticklabels(), color='white')


    # Second image (iterations)
    from matplotlib.colors import LogNorm
    bands = 24
    periodic = np.mod(iterations.astype(float), bands)
    im1 = axes[1].imshow(
        periodic,
        # iterations,
        origin='lower',
        extent=[-2, 2, -2, 2],
        cmap="turbo",
        interpolation="bilinear",
        # cmap="twilight",
        # norm=LogNorm(vmin=1, vmax=np.max(iterations))
    )

    axes[1].set_title("Iterations", color="white")

    # Add colorbar for second image
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar1.ax.get_yticklabels(), color='white')

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors='white')

    plt.savefig(
        'newton_fractal.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='black'
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--colors', type=pathlib.Path, required=True)
    parser.add_argument('--interations', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(**vars(parse_args()))
