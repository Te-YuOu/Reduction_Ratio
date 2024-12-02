# Imported packages
import numpy as np
import netCDF4 as nc
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Figure S1 and S2
def FigS1_S2(dpi, figure_path, data_path, shp_path):
    ## Coastline
    taiwan_coastline = gpd.read_file(shp_path+'/Taiwan_WGS84.shp')
    taiwan_coastline = taiwan_coastline.set_geometry('geometry')
    ## Terrain
    taiwan_dem20 = gpd.read_file(shp_path+'/Taiwan_2000m_WGS84.shp')
    taiwan_dem20 = taiwan_dem20.set_geometry('geometry')
    dem20 = nc.Dataset(data_path + '/dem20_TCCIPInsolation.nc')
    lon = dem20.variables['lon'][:]
    lat = dem20.variables['lat'][:]
    ele = dem20.variables['dem20'][:, :]
    Landmask = ~np.isnan(ele)

    # Load data
    MonthHour_max = np.load(data_path + '/MonthHour_max_20152019_Before.npy')
    # Set Seaborn theme
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(16, 12), dpi=50, constrained_layout=True)
    gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1], wspace=0.05, hspace=0.15)
    month_axes = {1:['Jan', fig.add_subplot(gs[0, 0])], 2:['Feb', fig.add_subplot(gs[1, 0])], 3:['Mar', fig.add_subplot(gs[2, 0])], 
                4:['Apr', fig.add_subplot(gs[0, 1])], 5:['May', fig.add_subplot(gs[1, 1])], 6:['Jun', fig.add_subplot(gs[2, 1])], 
                7:['Jul', fig.add_subplot(gs[0, 2])], 8:['Aug', fig.add_subplot(gs[1, 2])], 9:['Sep', fig.add_subplot(gs[2, 2])], 
                10:['Oct', fig.add_subplot(gs[0, 3])], 11:['Nov', fig.add_subplot(gs[1, 3])], 12:['Dec', fig.add_subplot(gs[2, 3])]}

    # Plot each month’s boxplot
    for mm in range(1, 13):
        ax = month_axes[mm][1]
        boxDiurnal = [MonthHour_max[mm-1, hh-6, :, :][Landmask] for hh in range(6, 20)]
        sns.boxplot(data=boxDiurnal, ax=ax, color='silver', showfliers=True, showmeans=True, width=1,
                    meanprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'red', 'markeredgecolor': 'red', 'alpha': 1},
                    flierprops={'marker': 'x', 'markersize': 3, 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'linestyle': 'none', 'alpha': 0.5})
        ax.set_title(month_axes[mm][0], weight='bold')
        ax.set_xticks(np.arange(0, 15, 3))
        ax.set_xticks(np.arange(0, 15, 1), minor=True)
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0, 1400, 200))
        ax.set_yticks(np.arange(0, 1300, 100), minor=True)
        ax.set_yticklabels([])
        ax.set_xlim(-0.5, 13.5)
        ax.set_ylim(-20, 1200)
        ax.grid()

    # Customize specific y-axis and x-axis tick labels
    for mm in [1, 2, 3]:
        month_axes[mm][1].set_yticklabels(np.arange(0, 1400, 200))
    for mm in [3, 6, 9, 12]:
        month_axes[mm][1].set_xticks(np.arange(0, 15, 3), labels=np.arange(6, 21, 3))

    # Labels
    month_axes[2][1].set_ylabel('Maximum Incoming Solar Radiation [Wm$^{-2}$]', weight='bold', fontsize=14)
    month_axes[6][1].text(12, -250, 'Time [hour]', weight='bold', fontsize=14)

    # Subfigure labels
    sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)']
    for mm in range(1, 13):
        ax = month_axes[mm][1]
        ax.text(0.04, 1.035, sub_labels[mm-1], transform=ax.transAxes, fontsize=12, va='center', ha='center')

    # Save and show plot
    fig.savefig(figure_path + '/FigureS1.png', bbox_inches='tight', dpi=dpi)
    plt.show()

    # Set figure and GridSpec
    fig = plt.figure(figsize=(16, 12), dpi=100, constrained_layout=True)
    gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1], wspace=-0.6, hspace=0.1)

    # Set up the month axes with WGS84 projection and grey background
    month_axes = {1:['Jan', fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())], 
                2:['Feb', fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())], 
                3:['Mar', fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())], 
                4:['Apr', fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())], 
                5:['May', fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())], 
                6:['Jun', fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())], 
                7:['Jul', fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())], 
                8:['Aug', fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())], 
                9:['Sep', fig.add_subplot(gs[2, 2], projection=ccrs.PlateCarree())], 
                10:['Oct', fig.add_subplot(gs[0, 3], projection=ccrs.PlateCarree())], 
                11:['Nov', fig.add_subplot(gs[1, 3], projection=ccrs.PlateCarree())], 
                12:['Dec', fig.add_subplot(gs[2, 3], projection=ccrs.PlateCarree())]}

    # Set background color for each subplot to grey
    for mm, (label, ax) in month_axes.items():
    #    ax.set_facecolor('lightgrey')  # Set the background to grey
        ax.set_title(label, weight='bold')
        ax.set_extent([119.9, 122.1, 21.8, 25.4])

        # Customizations for each month
        if mm in [3, 6, 9, 12]:
            ax.tick_params(labelsize='small')
            ax.set_xticks(np.arange(119, 122.5, 1), crs=ccrs.PlateCarree())
            ax.set_xticks(np.arange(119, 122.5, 0.5), minor=True, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())

        if mm in [1, 2, 3]:
            ax.tick_params(labelsize='small')
            ax.set_yticks(np.arange(21, 26, 1), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(21, 26, 0.5), minor=True, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
    # Function to darken a colormap by applying a gamma correction
    def darken_cmap(cmap_name, gamma):
        cmap = plt.get_cmap(cmap_name)
        cmap_colors = cmap(np.linspace(0, 1, cmap.N))
        darkened_colors = mcolors.ListedColormap(cmap_colors ** gamma)
        return darkened_colors

    # Use the darkened version of the colormaps
    dark_blues = darken_cmap('Blues', gamma=3)
    dark_reds = darken_cmap('Reds', gamma=3)
    for mm in range(1, 13):
        ax = month_axes[mm][1]
        ax.set_title(month_axes[mm][0], weight='bold')
        ax.set_extent([119.9, 122.1, 21.8, 25.4])

        taiwan_coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.3)
        taiwan_dem20.plot(ax=ax, facecolor='none', edgecolor='grey', linewidth=0.2, linestyle = '-')
        high_outlier = np.full((525, 575), 0)
        low_outlier = np.full((525, 575), 0)
        for hh in range(12, 17):
            Plot_data = MonthHour_max[mm-1, hh-6, :, :]
            maskPlot_data = np.ma.array(Plot_data, mask=~Landmask)
            h_outlier_criteria = np.nanquantile(Plot_data[Landmask], 0.75)+1.5*(np.nanstd(Plot_data[Landmask]))
            l_outlier_criteria = np.nanquantile(Plot_data[Landmask], 0.25)-1.5*(np.nanstd(Plot_data[Landmask]))
            
            high_outlier[maskPlot_data>h_outlier_criteria] = high_outlier[maskPlot_data>h_outlier_criteria] + 1
            low_outlier[maskPlot_data<l_outlier_criteria] = low_outlier[maskPlot_data<l_outlier_criteria] + 1      
        maskhigh_outlier = np.ma.array(high_outlier, mask=~(Landmask&(high_outlier!=0)))
        masklow_outlier = np.ma.array(low_outlier, mask=~(Landmask&(low_outlier!=0)))
        PLOT_low = ax.pcolormesh(lon, lat, masklow_outlier, cmap=dark_blues, shading='auto', vmin=0, vmax=5)
        PLOT_high = ax.pcolormesh(lon, lat, maskhigh_outlier, cmap=dark_reds, shading='auto', vmin=0, vmax=5)

    # Subfigure labels
    sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)']
    for mm in range(1, 13):
        ax = month_axes[mm][1]
        ax.text(0.04, 1.035, sub_labels[mm-1], transform=ax.transAxes, fontsize=12, va='center', ha='center')

    cax_low = fig.add_axes([month_axes[3][1].get_position().x0, month_axes[3][1].get_position().y0 - 0.05, 
                        month_axes[3][1].get_position().x1 - month_axes[3][1].get_position().x0, 0.015])
    cbar_low = plt.colorbar(PLOT_low, cax=cax_low, ticks=np.arange(0, 6), orientation='horizontal', extend='max')
    cbar_low.set_label('Low Outliers Hour Count', weight='bold', fontsize = 10)

    cax_high = fig.add_axes([month_axes[6][1].get_position().x0, month_axes[6][1].get_position().y0 - 0.05, 
                        month_axes[6][1].get_position().x1 - month_axes[6][1].get_position().x0, 0.015])
    cbar_high = plt.colorbar(PLOT_high, cax=cax_high, ticks=np.arange(0, 6), orientation='horizontal', extend='max')
    cbar_high.set_label('High Outliers Hour Count', weight='bold', fontsize = 10)

    legend_elements = [
        Line2D([0], [0], linestyle = '-', color='grey', label='2000m isotope', lw=2)
    ]
    month_axes[9][1].legend(handles=legend_elements, loc='center', frameon=False, ncol=1,
                            bbox_to_anchor=(0.5, -0.185), fontsize = 10)

    # Save and show plot
    fig.savefig(figure_path + '/FigureS2.png', bbox_inches='tight', dpi=dpi)
    plt.show()