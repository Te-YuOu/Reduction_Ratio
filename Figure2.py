# Imported packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns

time = np.linspace(0, 24, 100)
max_radiation = 1000 * np.sin(np.pi * (time - 6) / 12)**2
max_radiation[(time<6)|(time>18)] = 0
mean_radiation = 500 * np.sin(np.pi * (time - 6) / 12)**2
mean_radiation[(time<6)|(time>18)] = 0

# Figure 2
def Fig2(dpi, figure_path):
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(6, 5), dpi=100, constrained_layout=True)
    gs = gridspec.GridSpec(1, 1)

    diurnal_ax = fig.add_subplot(gs[0])

    # Plot the max and obs radiation
    diurnal_ax.plot(time, max_radiation, color='k', linestyle = ':', zorder = 2, lw = 3)
    diurnal_ax.plot(time, mean_radiation, color='k', linestyle = '-', zorder = 2, lw = 3)

    diurnal_ax.fill_between(time, 0, max_radiation, interpolate=True,
                    facecolor='none', edgecolor='darkred', hatch='o', zorder = 1)
    diurnal_ax.fill_between(time, mean_radiation, max_radiation, where=(max_radiation >= mean_radiation), interpolate=True,
                    facecolor='none', edgecolor='mediumblue', hatch='x', zorder = 1)

    # Labels and limits
    diurnal_ax.set_xticks(np.arange(0, 27, 3))
    diurnal_ax.set_xticks(np.arange(0, 25, 1), minor = True)
    diurnal_ax.set_yticks(np.arange(0, 1600, 200))
    diurnal_ax.set_yticks(np.arange(0, 1600, 100), minor = True)
    diurnal_ax.set_xlabel('Time [hour]', weight = 'bold')
    diurnal_ax.set_ylabel('Incoming Solar Radiation [Wm$^{-2}$]', weight = 'bold')
    diurnal_ax.set_ylim(0, 1100)
    diurnal_ax.set_xlim(3, 21)
    diurnal_ax.grid(alpha = 0.5, zorder = 0)

    legend_elements = [
        Line2D([0], [0], linestyle = '-', color='k', label='Observed', lw=2),
        Line2D([0], [0], linestyle = ':', color='k', label='Maximum', lw=2)
    ]
    diurnal_ax.legend(handles=legend_elements, loc='upper left', ncol=1)

    ax_for_legend2 = diurnal_ax.twinx()
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='mediumblue', hatch='xx', label='$\mathrm{Reduction_{Total}}$'),
        mpatches.Patch(facecolor='none', edgecolor='darkred', hatch='oo', label='$\mathrm{Maximum_{Total}}$')
    ]
    ax_for_legend2.legend(handles=legend_elements, loc='upper right', ncol=1)
    ax_for_legend2.set_yticks([])

    # Show plot
    plt.savefig(figure_path+'/Figure2.png',bbox_inches='tight', dpi = dpi)
    plt.show()
