# Imported packages
import numpy as np
import netCDF4 as nc
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# %% Figure 4 and S4
def Fig4_S4(dpi, figure_path, data_path):
    ## Coastline
    taiwan_coastline = gpd.read_file(data_path+'/Taiwan_WGS84.shp')
    taiwan_coastline = taiwan_coastline.set_geometry('geometry')
    ## Terrain
    dem20 = nc.Dataset(data_path+'/dem20_TCCIPInsolation.nc')
    lon = dem20.variables['lon'][:]
    lat = dem20.variables['lat'][:]
    ele = dem20.variables['dem20'][:,:]
    Landmask = ~np.isnan(ele) ## mask of land areas

    ## Mask for MCF, non-MCF, non-Forest 3 categories
    MCFfraction = np.load(data_path+'/MCFfraction_TCCIP.npy')
    nonmcf = (MCFfraction[:, :, 1] == 25)
    mcf = (MCFfraction[:, :, 2] == 25)
    nonforest = (MCFfraction[:, :, 0] == 25)

    # Load your shapefile 
    Forest = gpd.read_file(data_path+'/Forest2017.shp')
    Forest = Forest.set_geometry('geometry')

    ## Reduction Ratio without rainy data
    ReductionRatio = np.load(data_path+'/ReductionRatio_daily_20152019.npy')
    Rain = nc.Dataset(data_path+'/rain.20112019.daily.1km-grid-v2.nc')
    Rain = Rain.variables['rain'][:,:,:]
    ReductionRatio_norain = np.ma.array(ReductionRatio, mask = Rain[-1639:] <= 0)
    ReductionRatio_df = pd.DataFrame({'yyyymmdd': pd.date_range(start='2015-07-07',end='2019-12-31')})
    ReductionRatio_df['month'] = ReductionRatio_df['yyyymmdd'].apply(lambda t: t.month)

    # Taiwan Atmospheric Event Database from Su et al.(2018)(https://osf.io/4zutj/)
    weather_events = pd.read_csv(data_path+'/TAD_v2022_20220601.csv')
    weather_events['yyyymmdd'] = pd.to_datetime(weather_events['yyyymmdd'].map(str), format = '%Y%m%d')
    weather_events = weather_events[(weather_events['yyyymmdd'] >= '2015-07-07') & (weather_events['yyyymmdd'] <= '2019-12-31')]
    for iweather in ['CS', 'TYW', 'TC100', 'TC200', 'TC300', 'TC500', 'TC1000', 'NWPTY', 'FT', 'NE', 'SNE', 'SWF', 'SSWF']:
        ReductionRatio_df[iweather] = weather_events[iweather].values
    
    fig = plt.figure(figsize=(8, 4), dpi=100)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.2)

    map_ax = fig.add_subplot(gs[0, 0])

    # Plot the areas with specified colors

    map_ax.tick_params(labelsize='small')
    map_ax.set_xticks(np.arange(120, 122.5, 1), labels = ['120°E', '121°E', '122°E'])
    map_ax.set_xticks(np.arange(120, 122.5, 0.5), minor=True)
    map_ax.set_yticks(np.arange(22, 25.5, 1), labels = ['22°N', '23°N', '24°N', '25°N'])
    map_ax.set_yticks(np.arange(22, 26, 0.5), minor=True)
    map_ax.set_xlim(119.9, 122.1)
    map_ax.set_ylim(21.8, 25.4)

    period_parameters = {
        'NE': (((ReductionRatio_df['month'] == 12) | (ReductionRatio_df['month'] == 1) | (ReductionRatio_df['month'] == 2)) & 
                ~((ReductionRatio_df['NE'] == 0) )), 
        'None': (((ReductionRatio_df['month'] == 12) | (ReductionRatio_df['month'] == 1) | (ReductionRatio_df['month'] == 2)) & 
                ((ReductionRatio_df['NE'] == 0) & (ReductionRatio_df['FT'] == 0) & (ReductionRatio_df['CS'] == 0)))}
    Plot_data_NE = np.nanmean(ReductionRatio_norain[period_parameters['NE'], :, :], axis=0) * 100
    Plot_data_None = np.nanmean(ReductionRatio_norain[period_parameters['None'], :, :], axis=0) * 100
    maskPlot_data = np.ma.array(Plot_data_NE - Plot_data_None, mask=~Landmask)

    PLOT = map_ax.contourf(lon, lat, maskPlot_data, levels=np.arange(0, 32.5, 5), cmap='Reds', extend='both', alpha=1)
    cax = fig.add_axes([map_ax.get_position().x0-0.02, map_ax.get_position().y0, 
                        0.01, map_ax.get_position().height])
    cbar = plt.colorbar(PLOT, cax=cax, ticks=np.arange(0, 32.5, 5), orientation='vertical', extend='both')
    cbar.set_label('NE-induced Reduction Ratio [%]', weight='bold')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('left')

    taiwan_coastline.plot(ax=map_ax, facecolor='none', edgecolor='black', linewidth=0.5)
    Forest.plot(ax=map_ax, facecolor='none', hatch='xxxxxx', linewidth=0.01, edgecolor='black', alpha=0.5)
 
    legend_elements = [
        mpatches.Patch(facecolor='none', hatch='xxxxx', linewidth=0.001, edgecolor=(0.2, 0.2, 0.2), label='forest region')]
    map_ax.legend(handles=legend_elements, loc='upper left', ncol=1, fontsize=7)

    scatter_ax = fig.add_subplot(gs[0, 1])

    cmap = plt.cm.Reds  # Adjust as needed to match your color scheme
    norm = plt.Normalize(vmin=0, vmax=30)  # Ensure this matches the right figure's color bar

    Plot_data_DJF = np.nanmean(ReductionRatio_norain[((ReductionRatio_df['month'] == 12) | (ReductionRatio_df['month'] == 1) | (ReductionRatio_df['month'] == 2)), :, :], axis=0) * 100

    reduction_ratio_no_forest = Plot_data_DJF[nonforest]
    reduction_ratio_forest = Plot_data_DJF[nonmcf | mcf]

    Plot_data = Plot_data_NE - Plot_data_None
    reduction_by_ne_no_forest = Plot_data[nonforest]
    reduction_by_ne_forest = Plot_data[nonmcf | mcf]

    scatter_ax.scatter(reduction_ratio_forest, reduction_by_ne_forest, color=(0, 0.4, 0), s=0.1, alpha=0.7, label='forest')
    r, p = pearsonr(reduction_ratio_forest, reduction_by_ne_forest)
    scatter_ax.text(scatter_ax.get_position().x0-0.5, scatter_ax.get_position().y1, '$\mathrm{r_{forest}}\:\:\:\:\:\:$= '+str(round(r, 2))+'***', transform=scatter_ax.transAxes, 
                    color=(0, 0.3, 0))

    scatter_ax.scatter(reduction_ratio_no_forest, reduction_by_ne_no_forest, color=(0.2, 0.2, 0.2), s=0.1, alpha=0.8, label='no forest')
    r, p = pearsonr(reduction_ratio_no_forest, reduction_by_ne_no_forest)
    scatter_ax.text(scatter_ax.get_position().x0-0.5, scatter_ax.get_position().y1+0.06, '$\mathrm{r_{no\:forest}}\:$= '+str(round(r, 2))+'***', transform=scatter_ax.transAxes, 
                    color=(0.03, 0.03, 0.03))

    scatter_ax.plot(np.linspace(30, 85, 10), np.linspace(0, 35, 10), linestyle='--', color='k', linewidth=1)
    scatter_ax.set_xticks(np.arange(10, 90, 10))
    scatter_ax.set_xticks(np.arange(15, 90, 5), minor=True)
    scatter_ax.set_xlim(30, 85)
    scatter_ax.set_xlabel('DJF-mean Reduction Ratio [%]', weight='bold')
    scatter_ax.set_yticks(np.arange(0, 80, 5))
    scatter_ax.set_yticks(np.arange(0, 80, 2.5), minor=True)
    scatter_ax.set_ylim(0, 35)
    scatter_ax.set_ylabel('NE-induced Reduction Ratio [%]', weight='bold')
    scatter_ax.grid(alpha=0.5)

    # Add subfigure labels
    for ax, label in zip([map_ax, scatter_ax], ['(a)', '(b)']):
        ax.text(0.02, 1.05, label, transform=ax.transAxes, fontsize=14, va='center', ha='center')

    fig.savefig(figure_path+'/Figure4.png', bbox_inches='tight', dpi=dpi)
    plt.show()

    period_parameters = {
        'NE': (((ReductionRatio_df['month'] == 12) | (ReductionRatio_df['month'] == 1) | (ReductionRatio_df['month'] == 2)) & (ReductionRatio_df['NE'] == 1)), 
        'None': (((ReductionRatio_df['month'] == 12) | (ReductionRatio_df['month'] == 1) | (ReductionRatio_df['month'] == 2)) & 
                ((ReductionRatio_df['NE'] == 0) & (ReductionRatio_df['FT'] == 0) & (ReductionRatio_df['CS'] == 0)))
    }
    # Define reduction ratio map function
    def plot_reduction_map(ax, Plot_data, lon, lat, Landmask, extent=[119.9, 122.1, 21.8, 25.4]):
        maskPlot_data = np.ma.array(Plot_data, mask=~Landmask)
        ax.tick_params(labelsize='small')
        ax.set_xticks(np.arange(119, 122.5, 1), crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(119, 122.5, 0.5), minor=True, crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(21, 26, 1), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(21, 26, 0.5), minor=True, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_extent(extent)
        ax.gridlines(crs=ccrs.PlateCarree(), color='black', linestyle='dotted', 
                    xlocs=np.arange(118, 123, 0.5), ylocs=np.arange(21, 26, 0.5), alpha=0.3)
        return ax.contourf(lon, lat, maskPlot_data, levels=np.arange(10, 82.5, 2.5), cmap='Blues', extend='both', alpha=1)

    # Seaborn theme and figure initialization
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(6, 4), dpi=100, constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0, hspace=0.3)

    # Subplot map and scatter plot initialization
    map_axes = [fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()), fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())]

    # Plot maps and scatter plots for JJA and DJF
    for i, iperiod in enumerate(period_parameters.keys()):
        Plot_data = np.nanmean(ReductionRatio_norain[period_parameters[iperiod], :, :], axis=0) * 100
        proportion = round(100*np.sum(period_parameters[iperiod]) / np.sum(((ReductionRatio_df['month'] == 12) | (ReductionRatio_df['month'] == 1) | (ReductionRatio_df['month'] == 2))), 2)
        # Plot Reduction Ratio Maps
        PLOT = plot_reduction_map(map_axes[i], Plot_data, lon, lat, Landmask)
        map_axes[i].set_title(iperiod +'('+ str(proportion)+'%)', weight = 'bold', loc = 'right')
        taiwan_coastline.plot(ax=map_axes[i], facecolor='none', edgecolor='black', linewidth=0.5)

    cax = fig.add_axes([map_axes[1].get_position().x1 + 0.02, map_axes[1].get_position().y0, 
                        0.015, map_axes[1].get_position().height])
    cbar = plt.colorbar(PLOT, cax=cax, ticks=np.arange(10, 81, 10), orientation='vertical', extend='both')
    cbar.set_label('Reduction Ratio [%]', weight='bold')

    map_axes[1].set_xticklabels([])

    # Add subfigure labels
    for ax, label in zip([map_axes[0], map_axes[1]], ['(a)', '(b)']):
        ax.text(0.02, 1.04, label, transform=ax.transAxes, fontsize=14, va='center', ha='center')

    fig.savefig(figure_path+'/FigureS4.png', bbox_inches='tight', dpi=dpi)
    plt.show()