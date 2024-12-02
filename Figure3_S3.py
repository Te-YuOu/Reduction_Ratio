# Imported packages
import numpy as np
import netCDF4 as nc
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Figure 3 and S3
def Fig3_S3(dpi, figure_path, data_path):
    # Data Preparation
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

    ## Reduction Ratio without rainy data
    ReductionRatio = np.load(data_path+'/ReductionRatio_daily_20152019.npy')
    Rain = nc.Dataset(data_path+'/rain.20112019.daily.1km-grid-v2.nc')
    Rain = Rain.variables['rain'][:,:,:]
    ReductionRatio_norain = np.ma.array(ReductionRatio, mask = Rain[-1639:] <= 0)
    ReductionRatio_df = pd.DataFrame({'yyyymmdd': pd.date_range(start='2015-07-07',end='2019-12-31')})
    ReductionRatio_df['month'] = ReductionRatio_df['yyyymmdd'].apply(lambda t: t.month)
    
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
        #ax.gridlines(crs=ccrs.PlateCarree(), color='black', linestyle='dotted', 
        #            xlocs=np.arange(118, 123, 0.5), ylocs=np.arange(21, 26, 0.5), alpha=0.3)
        return ax.contourf(lon, lat, maskPlot_data, levels=np.arange(20, 82.5, 2.5), cmap='Blues', extend='both', alpha=1)

    # Define scatter plot function
    def plot_scatter(ax, Plot_data, ele, nonmcf, mcf, nonforest):
        ax.scatter(Plot_data[nonmcf], ele[nonmcf], s=0.1, color='darkred', label='other forest')
        ax.scatter(Plot_data[mcf], ele[mcf], s=0.1, color='navy', label='MCF')
        ax.scatter(Plot_data[nonforest], ele[nonforest], s=0.1, color='grey', label='no forest')
        ax.set_xticks(np.arange(10, 90, 10))
        ax.set_xticks(np.arange(20, 85, 5), minor=True)
        ax.set_yticks(np.arange(0, 4000, 500))
        ax.set_yticks(np.arange(0, 4000, 250), minor=True)
        ax.set_xlim(20, 80)
        ax.set_ylim(0, 3750)
        ax.set_xlabel('Reduction Ratio [%]', weight='bold')
        ax.set_ylabel('Elevation [m]', weight='bold')
        ax.grid(alpha=0.5)

    # Define boxplot function
    def plot_boxplot(ax, maskPlot_data, nonmcf, mcf, nonforest):
        data = {
            'Category': ['no forest'] * maskPlot_data[nonforest].size +
                        ['other forest'] * maskPlot_data[nonmcf].size +
                        ['MCF'] * maskPlot_data[mcf].size,
            'Reduction Ratio': np.concatenate([maskPlot_data[nonforest].compressed(),
                                            maskPlot_data[nonmcf].compressed(),
                                            maskPlot_data[mcf].compressed()])
        }
        df = pd.DataFrame(data)
        boxplot = sns.boxplot(x='Category', y='Reduction Ratio', data=df, ax=ax, showmeans=True, 
                            palette=sns.color_palette(['silver', 'darkred', 'navy']), width=0.6, showfliers=True, 
                            meanprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'alpha': 1},
                            flierprops={'marker': 'x', 'color': (0.1,0.1,0.1), 'markersize': 4, 'linestyle': 'none', 'alpha': 0.6})
        # Set box alpha by modifying the boxes after they are created
        for patch in boxplot.patches:
            patch.set_alpha(0.5)  # Change this value to adjust the transparency
        ax.set_xlabel('')
        ax.set_xticklabels(['no forest', 'other forest', 'MCF'], weight='bold', fontsize=10)
        ax.set_yticks(np.arange(20, 95, 10))
        ax.set_yticks(np.arange(20, 95, 5), minor=True)
        ax.set_ylim(20, 90)
        ax.set_ylabel('Reduction Ratio [%]', weight='bold')
        ax.grid(alpha=0.5)

        # Display p-values on the plot
        y_max = df['Reduction Ratio'].max() * 1.03  # Set y_max for annotation height
        y_offset = 3.2  # Offset for each p-value line

        t, p = stats.ttest_ind(maskPlot_data[nonmcf], maskPlot_data[nonforest], equal_var=False, alternative='greater')
        print('Ha: Reduction in other forest is greater than in no forest')
        print(f't-statistic:{t}\np value:{p}\n')
        x1, x2 = 0, 1
        y = y_max + 0 * y_offset  # Adjust y position for each line
        ax.plot([x1, x1, x2, x2], [y - y_offset / 5, y, y, y - y_offset / 5], color="black", linewidth=1)        
        if p < 10e-3:
            ax.text((x1 + x2) * 0.5, y-0.7, '***', ha='center', color="black")
        else:
            ax.text((x1 + x2) * 0.5, y+0.7, 'ns', ha='center', color="black", fontsize=10)  
                    
        t, p = stats.ttest_ind(maskPlot_data[mcf], maskPlot_data[nonforest], equal_var=False, alternative='greater')
        print('Ha: Reduction in MCF is greater than in no forest')
        print(f't-statistic:{t}\np value:{p}\n')
        x1, x2 = 0, 2
        y = y_max + 2 * y_offset  # Adjust y position for each line
        ax.plot([x1, x1, x2, x2], [y - y_offset / 5, y, y, y - y_offset / 5], color="black", linewidth=1)        
        if p < 10e-3:
            ax.text((x1 + x2) * 0.5, y-0.7, '***', ha='center', color="black")
        else:
            ax.text((x1 + x2) * 0.5, y+0.7, 'ns', ha='center', color="black", fontsize=10)       
        
        t, p = stats.ttest_ind(maskPlot_data[mcf], maskPlot_data[nonmcf], equal_var=False, alternative='greater')
        print('Ha: Reduction in MCF is greater than in other forest')
        print(f't-statistic:{t}\np value:{p}\n')
        x1, x2 = 1, 2
        y = y_max + 1 * y_offset  # Adjust y position for each line
        ax.plot([x1, x1, x2, x2], [y - y_offset / 5, y, y, y - y_offset / 5], color="black", linewidth=1)        
        if p < 10e-3:
            ax.text((x1 + x2) * 0.5, y-0.7, '***', ha='center', color="black")
        else:
            ax.text((x1 + x2) * 0.5, y+0.7, 'ns', ha='center', color="black", fontsize=10)  
        
        


    # JJA, DJF 
    period_parameters = {
        'JJA': ((ReductionRatio_df['month'] == 6) | (ReductionRatio_df['month'] == 7) | (ReductionRatio_df['month'] == 8)), 
        'DJF': ((ReductionRatio_df['month'] == 12) | (ReductionRatio_df['month'] == 1) | (ReductionRatio_df['month'] == 2))
    }
    # Seaborn theme and figure initialization
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(12, 8), dpi=100, constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, width_ratios=[2, 2, 2.5], height_ratios=[1, 1], wspace=0.3, hspace=0.3)

    # Subplot map and scatter plot initialization
    map_axes = [fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
                fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())]
    scatter_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    box_axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])]

    # Plot maps and scatter plots for JJA and DJF
    for i, iperiod in enumerate(period_parameters.keys()):
        print(iperiod)
        Plot_data = np.nanmean(ReductionRatio_norain[period_parameters[iperiod], :, :], axis=0) * 100
        
        # Plot Reduction Ratio Maps
        PLOT = plot_reduction_map(map_axes[i], Plot_data, lon, lat, Landmask)
        taiwan_coastline.plot(ax=map_axes[i], facecolor='none', edgecolor='black', linewidth=0.5)
        
        # Plot Scatter Plots
        plot_scatter(scatter_axes[i], Plot_data, ele, nonmcf, mcf, nonforest)
        
        # Plot Boxplots
        maskPlot_data = np.ma.array(Plot_data, mask=~Landmask)
        plot_boxplot(box_axes[i], maskPlot_data, nonmcf, mcf, nonforest)

    # Add colorbar, legend, and labels
    cax = fig.add_axes([map_axes[1].get_position().x0, map_axes[1].get_position().y0 - 0.07, 
                        map_axes[1].get_position().width, 0.02])
    cbar = plt.colorbar(PLOT, cax=cax, ticks=np.arange(20, 82.5, 20), orientation='horizontal', extend='both')
    cbar.set_label('Reduction Ratio [%]', weight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='no forest', markerfacecolor='grey', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='other forest', markerfacecolor='darkred', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='MCF', markerfacecolor='navy', markersize=6)
    ]
    scatter_axes[1].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3)

    # Hide x-axis labels for the top row
    for ax in [scatter_axes[0]]:
        ax.set_xlabel('')
        
    # Add subfigure labels
    for ax, label in zip([map_axes[0], scatter_axes[0], box_axes[0], 
                        map_axes[1], scatter_axes[1], box_axes[1]], ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
        ax.text(0.05, 1.05, label, transform=ax.transAxes, fontsize=14, va='center', ha='center')
    fig.savefig(figure_path+'/Figure3.png',bbox_inches='tight', dpi = 600)
    plt.show()
    # MAM, SOP
    period_parameters = {
        'MAM': ((ReductionRatio_df['month'] == 3) | (ReductionRatio_df['month'] == 4) | (ReductionRatio_df['month'] == 5)), 
        'SOP': ((ReductionRatio_df['month'] == 9) | (ReductionRatio_df['month'] == 10) | (ReductionRatio_df['month'] == 11))
    }
    # Seaborn theme and figure initialization
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(12, 8), dpi=100, constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, width_ratios=[2, 2, 2.5], height_ratios=[1, 1], wspace=0.3, hspace=0.3)

    # Subplot map and scatter plot initialization
    map_axes = [fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
                fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())]
    scatter_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    box_axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])]

    # Plot maps and scatter plots for MAM and SOP
    for i, iperiod in enumerate(period_parameters.keys()):
        print(iperiod)
        Plot_data = np.nanmean(ReductionRatio_norain[period_parameters[iperiod], :, :], axis=0) * 100
        
        # Plot Reduction Ratio Maps
        PLOT = plot_reduction_map(map_axes[i], Plot_data, lon, lat, Landmask)
        taiwan_coastline.plot(ax=map_axes[i], facecolor='none', edgecolor='black', linewidth=0.5)
        # Plot Scatter Plots
        plot_scatter(scatter_axes[i], Plot_data, ele, nonmcf, mcf, nonforest)
        
        # Plot Boxplots
        maskPlot_data = np.ma.array(Plot_data, mask=~Landmask)
        plot_boxplot(box_axes[i], maskPlot_data, nonmcf, mcf, nonforest)

    # Add colorbar, legend, and labels
    cax = fig.add_axes([map_axes[1].get_position().x0, map_axes[1].get_position().y0 - 0.07, 
                        map_axes[1].get_position().width, 0.02])
    cbar = plt.colorbar(PLOT, cax=cax, ticks=np.arange(20, 82.5, 20), orientation='horizontal', extend='both')
    cbar.set_label('Reduction Ratio [%]', weight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='no forest', markerfacecolor='grey', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='other forest', markerfacecolor='darkred', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='MCF', markerfacecolor='navy', markersize=6)
    ]
    scatter_axes[1].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3)

    # Hide x-axis labels for the top row
    for ax in [scatter_axes[0]]:
        ax.set_xlabel('')
        
    # Add subfigure labels
    for ax, label in zip([map_axes[0], scatter_axes[0], box_axes[0], 
                        map_axes[1], scatter_axes[1], box_axes[1]], ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
        ax.text(0.05, 1.05, label, transform=ax.transAxes, fontsize=14, va='center', ha='center')
    fig.savefig(figure_path+'/FigureS3.png',bbox_inches='tight', dpi = 600)
    plt.show()