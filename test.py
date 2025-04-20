import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors

fig, axs = plt.subplots(1, 2)
fig.suptitle('Correlation matrices, 28 trivial sites')

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
# axs[0].set_xticks(list(range(28)), np.arange(1, 29))
# axs[0].set_yticks(list(range(28)), np.arange(1, 29))
# axs[1].set_xticks(list(range(28)), np.arange(1, 29))
# axs[1].set_yticks(list(range(28)), np.arange(1, 29))

fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, cmap='coolwarm', label='Value')

plt.show()