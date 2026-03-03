import numpy as np
from matplotlib import pyplot as plt

colors_f = 'colors.bin'
iterations_f = 'iterations.bin'

# ---------- Load colors ----------
with open(colors_f, "rb") as f:
    dim0 = np.fromfile(f, dtype=np.uint64, count=1)[0]
    dim1 = np.fromfile(f, dtype=np.uint64, count=1)[0]
    data = np.fromfile(f, dtype=np.uint32)

data = data.reshape((dim1, dim0))  # assumes LayoutRight

# ---------- Load iterations ----------
with open(iterations_f, "rb") as f:
    dim0_i = np.fromfile(f, dtype=np.uint64, count=1)[0]
    dim1_i = np.fromfile(f, dtype=np.uint64, count=1)[0]
    iterations = np.fromfile(f, dtype=np.uint32)

iterations = iterations.reshape((dim1_i, dim0_i))

print(np.min(iterations))
print(np.max(iterations))

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