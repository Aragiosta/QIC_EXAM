import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors

np.random.seed(19680801)

datasets = [
    (i+1)/10 * np.random.rand(10, 20)
    for i in range(4)
]

fig, axs = plt.subplots(1, 2)
fig.suptitle('Correlation matrices, 14 topological sites')

# create a single norm to be shared across all images
norm = colors.Normalize(vmin=-1, vmax=1)

images = []
for ax, data in zip(axs.flat, [np.real(zz_corr), np.real(xx_corr)]):
    images.append(
        ax.imshow(
            data,
            norm=norm,
            origin='lower',
            cmap='coolwarm'
        )
    )

axs[0].set_title('$\\langle Z_i Z_j \\rangle - \\langle Z_i \\rangle\\langle Z_j \\rangle$')
axs[1].set_title('$\\langle X_i X_j \\rangle - \\langle X_i \\rangle\\langle X_j \\rangle$')
axs[0].set_xticks(list(range(14)), np.arange(1, 15))
axs[0].set_yticks(list(range(14)), np.arange(1, 15))
axs[1].set_xticks(list(range(14)), np.arange(1, 15))
axs[1].set_yticks(list(range(14)), np.arange(1, 15))

fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, cmap='coolwarm', label='Value')

plt.show()