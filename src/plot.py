import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib as mpl

result = np.load(
    "/data2/david3684/2024_arithmetic/disentanglement_errors_0.5.npy"
)

print(result)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(
    result,
    extent=[-2, 2, -2, 2],
    origin="lower",
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
)

# plot a square x: -1, 1 y: -1, 1, red dashed line, no fill color
ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "r--", linewidth=1)
# plot a x at the origin
ax.plot([0], [0], "rx", linewidth=1)

ax.set_title("DTD - SUN397")
ax.set_aspect("equal")
ax.set_xlabel(r"$\alpha_1$")
ax.set_ylabel(r"$\alpha_2$")
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-2, -1, 0, 1, 2])

fig.colorbar(im, ax=ax)
fig.tight_layout()
plt.show()
plt.savefig("disentanglement_errors_rank0.5.png")