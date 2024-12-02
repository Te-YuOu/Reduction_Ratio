# Imported packages
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import contextily as ctx
import cartopy.crs as ccrs
from shapely.geometry import Point

# In-situ data preparation
def CWA_StationData(istno, data_path):
    # Load data
    file_name = '/'+istno+'_19800101_20201201.csv'
    df = pd.read_csv(data_path+file_name)

    # Time variable
    df['yyyymmddhh'] = pd.to_datetime(df['yyyymmddhh'], errors='coerce')
    df['hour'] = df['yyyymmddhh'].apply(lambda t: t.hour)
    df['month'] = df['yyyymmddhh'].apply(lambda t: t.month)
    df['date'] = df['yyyymmddhh'].apply(lambda t: t.date())
    
    # Raw data 
    ## Extract valid data
    for ivar in ['SS02','VS01','CD11']: ## Required variable: Solar Radiaiton, Visibility, Total Cloud Amount, Obscurity
        df[ivar].loc[df[ivar]<0]=np.nan
    df['CD11'].loc[df['CD11']>10]=np.nan
    df['VS01'].loc[-((df['hour'] == 2)|(df['hour'] == 5)|(df['hour'] == 8)|(df['hour'] == 9)|(df['hour'] == 11)|(df['hour'] == 14)|(df['hour'] == 17)|(df['hour'] == 20)|(df['hour'] == 21))] = np.nan
    df['CD11'].loc[-((df['hour'] == 2)|(df['hour'] == 5)|(df['hour'] == 8)|(df['hour'] == 9)|(df['hour'] == 11)|(df['hour'] == 14)|(df['hour'] == 17)|(df['hour'] == 20)|(df['hour'] == 21))] = np.nan
    ## Transfer unit to W/m^2
    df["SS02"] = df["SS02"]/0.0036

    # Season classificatoin
    df['season'] = np.full(len(df), np.nan)
    df['season'].loc[(df['month'] == 12)|(df['month'] == 1)|(df['month'] == 2)] = 'DJF'
    df['season'].loc[(df['month'] == 3)|(df['month'] == 4)|(df['month'] == 5)] = 'MAM'
    df['season'].loc[(df['month'] == 6)|(df['month'] == 7)|(df['month'] == 8)] = 'JJA'
    df['season'].loc[(df['month'] == 9)|(df['month'] == 10)|(df['month'] == 11)] = 'SON'

    # Extract required variable
    df = df[(['yyyymmddhh', 'date', 'hour','season', 'month', 'SS02', 'VS01', 'CD11', 'PP01'])]

    # Remove rainy data
    df = df[df['PP01'] == 0]
    return df

# %% Figure 1
def Fig1(dpi, figure_path, data_path):
    CWAStation = {'Alishan':{'data':CWA_StationData('467530', data_path), 'color':'navy', 'type':'MCF', 'lon':120.813242, 'lat':23.508208},
                  'Chiayi':{'data':CWA_StationData('467480', data_path), 'color':'darkred', 'type':'no forest', 'lon':120.432906, 'lat':23.495925}}
    
    # Create a figure with two subplots using GridSpec
    fig = plt.figure(figsize=(10, 7), dpi=100)
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1], wspace=0.4, hspace=0.3)

    # Main plot for data (ax1 and ax2)
    ax1 = fig.add_subplot(gs[0, 0])

    # for legend
    legend_elements = [
        Line2D([0], [0], linestyle = '-', color='k', label='Mean', lw=2),
        Line2D([0], [0], linestyle = ':', color='k', label='Maximum', lw=2),
        Line2D([0], [0], linestyle = '-', color='grey', label='Cloud-Fog', lw=6, alpha = 0.7)
    ]
    ax1.legend(handles=legend_elements, loc='upper left', ncol=1)
    
    # Move ax1 y-axis to the right side
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax1.set_yticks(np.arange(0, 80, 10))
    ax1.set_yticks(np.arange(0, 80, 5), minor = True)
    ax1.set_ylim(0, 60)
    ax1.set_ylabel('Cloud-Fog Proportion [%]', weight='bold')
    ax1.set_xticks(np.arange(0, 27, 3))
    ax1.set_xticks(np.arange(0, 24, 1), minor=True)
    ax1.set_xlim(0, 24)
    ax1.set_xlabel('Time [hour]', weight='bold')
    ax1.grid(zorder = 0, alpha = 0.5)

    # Plot bars (proportion) on ax1 for Chiayi
    df = CWAStation['Chiayi']['data']
    obs = (df['hour'] == 8) | (df['hour'] == 11) | (df['hour'] == 14) | (df['hour'] == 17)
    dis_all = df.groupby(df.hour).apply(lambda g: g.count())

    idf = df[((df['CD11'] == 10)|(df['VS01'] <= 1)) & obs]
    idis = idf.groupby(idf.hour).apply(lambda g: g.count())
    Dis_percent = idis / dis_all

    bar_width = 0.8
    r1 = Dis_percent['CD11'].index-0.4  # Positions for Chiayi bars
    r2 = [x + bar_width for x in r1]  # Positions for Alishan bars

    ax1.bar(r1, Dis_percent['CD11'].values * 100, color='indianred', width=bar_width, alpha=0.7, zorder=1)

    # Alishan data
    df = CWAStation['Alishan']['data']
    obs = (df['hour'] == 8) | (df['hour'] == 11) | (df['hour'] == 14) | (df['hour'] == 17)
    dis_all = df.groupby(df.hour).apply(lambda g: g.count())

    idf = df[((df['CD11'] == 10)|(df['VS01'] <= 1)) & obs]
    idis = idf.groupby(idf.hour).apply(lambda g: g.count())
    Dis_percent = idis / dis_all
    ax1.bar(r2, Dis_percent['CD11'].values * 100, color='steelblue', width=bar_width, alpha=0.7, zorder=1)

    # Now, plot the lines for mean and Q99 on ax2
    ax2 = ax1.twinx()
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()
    ax2.spines["left"].set_position(("outward", 0))
    ax2.set_ylabel('Incoming Solar Radiation [Wm$^{-2}$]', weight='bold')
    ax2.set_ylim(0, 1200)
    ax2.set_yticks(np.arange(0, 1400, 200))
    ax2.set_yticks(np.arange(0, 1400, 100), minor = True)

    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()

    legend_elements = [
        Line2D([0], [0], linestyle = '-', color='darkred', label='Chiayi', lw=2),
        Line2D([0], [0], linestyle = '-', color='navy', label='Alishan', lw=2)
    ]
    ax2.legend(handles=legend_elements, loc='upper right', ncol=1)
    
    time = np.arange(0, 24)
    for istation in ['Alishan', 'Chiayi']:
        df = CWAStation[istation]['data']
        mean = df['SS02'].groupby(df.hour).apply(lambda g: g.mean(skipna=True))
        maximum = df['SS02'].groupby(df.hour).apply(lambda g: g.quantile(0.99))
        hours = time[:] 
        ax2.plot(hours, maximum, color=CWAStation[istation]['color'], linestyle=':', lw=3)
        ax2.plot(hours, mean, color=CWAStation[istation]['color'], linestyle='-', lw=3)
    
    relationship_parameters = {'CLD':['CD11', 'Total Cloud Amount', '[out of 10]'], 'VIS':['VS01', 'Visibility', '[km]']}
    itime = 11
    icriteria = 7
    iseason = 'JJA'
    idf = CWAStation['Alishan']['data']
    idf = idf[(idf['season'] == iseason)]
    idf_itime = idf[(idf['hour'] == itime)] 

    ax3 = fig.add_subplot(gs[0, 1])
    iGroup = 'CLD'
    ivar = relationship_parameters[iGroup][0]
    XY = pd.concat([idf_itime['SS02'], idf_itime[ivar]], axis=1)
    sample_size = XY.groupby(ivar).apply(lambda g: g.count())['SS02']
    Mean = XY.groupby(ivar).apply(lambda g: g.mean(skipna=True))['SS02']
    Median = XY.groupby(ivar).apply(lambda g: g.median(skipna=True))['SS02']
    Q25 = XY.groupby(ivar).apply(lambda g: g.quantile(0.25))['SS02']
    Q75 = XY.groupby(ivar).apply(lambda g: g.quantile(0.75))['SS02']
    idx_nan = np.isnan(idf_itime['SS02'].values) | np.isnan(idf_itime[ivar].values)
    r, p = pearsonr(idf_itime['SS02'].values[~idx_nan], idf_itime[ivar].values[~idx_nan])
    ax3.text(ax3.get_position().x0-0.6, ax3.get_position().y0-0.5, 'r = -'+str(np.abs(round(r, 2)))+' ***', transform=ax3.transAxes)
    
    CRITERIA = sample_size > icriteria

    ax3.hlines(Mean.index[CRITERIA], Q25[CRITERIA], Q75[CRITERIA], color = 'grey', label = "IQR")
    ax3.plot(Median[CRITERIA], Mean.index[CRITERIA], linestyle = '', marker = "o", markerfacecolor = "w", markeredgecolor = 'grey', label = "median", markersize = 5)
    ax3.plot(Mean[CRITERIA], Mean.index[CRITERIA], linestyle = '', marker = "o", markerfacecolor = "k", markeredgecolor = 'k', label = "mean", markersize = 5)
    ax3.set_xticks(np.arange(0, 1400, 200))
    ax3.set_xticks(np.arange(0, 1400, 100), minor=True)
    ax3.set_xlim(100, 1100)
    ax3.set_yticks(np.arange(0, 11))
    ax3.set_ylim(-0.5, 10.5)
    ax3.set_ylabel('Total Cloud Amount [out of 10]', weight = 'bold')
    ax3.grid(alpha = 0.5)

    ax4 = fig.add_subplot(gs[1, 1])
    iGroup = 'VIS'
    ivar = relationship_parameters[iGroup][0]
    XY = pd.concat([idf_itime['SS02'], idf_itime[ivar]], axis=1)
    sample_size = XY.groupby(ivar).apply(lambda g: g.count())['SS02']
    Mean = XY.groupby(ivar).apply(lambda g: g.mean(skipna=True))['SS02']
    Median = XY.groupby(ivar).apply(lambda g: g.median(skipna=True))['SS02']
    Q25 = XY.groupby(ivar).apply(lambda g: g.quantile(0.25))['SS02']
    Q75 = XY.groupby(ivar).apply(lambda g: g.quantile(0.75))['SS02']
    idx_nan = np.isnan(idf_itime['SS02'].values) | np.isnan(idf_itime[ivar].values)
    r, p = pearsonr(idf_itime['SS02'].values[~idx_nan], idf_itime[ivar].values[~idx_nan])
    ax4.text(ax4.get_position().x1-0.45, ax4.get_position().y0-0.05, 'r = +'+str(round(r, 2))+' ***', transform=ax4.transAxes)

    CRITERIA = sample_size > icriteria
    index = [0, 0.1, 0.2, 0.5, 1, 2.5, 3, 5, 10, 12, 15, 20]

    ax4.hlines(index, Q25[index], Q75[index], color = 'grey', label = "IQR")
    ax4.plot(Median[index], index, linestyle = '', marker = "o", markerfacecolor = "w", markeredgecolor = 'grey', label = "median", markersize = 5)
    ax4.plot(Mean[index], index, linestyle = '', marker = "o", markerfacecolor = "k", markeredgecolor = 'k', label = "mean", markersize = 5)
    ax4.set_xticks(np.arange(0, 1400, 200))
    ax4.set_xticks(np.arange(0, 1400, 100), minor=True)
    ax4.set_xlim(100, 1100)
    ax4.set_xlabel('Incoming Solar Radiation [Wm$^{-2}$]', weight = 'bold')
    ax4.set_yscale('log') #log sale for x-axis in visibility
    ax4.set_yticks([0.1, 0.5, 1, 5, 10, 20], labels=[0.1, 0.5, 1, 5, 10, 20])
    ax4.set_ylabel('Visibility [km]', weight = 'bold')
    ax4.grid(alpha = 0.5)

    legend_elements = [
        Line2D([0], [0], linestyle = '-', color = 'grey', label = "IQR", lw=2),
        Line2D([0], [0], linestyle = '', marker = "o", markerfacecolor = "w", markeredgecolor = 'grey', label = "median", markersize = 5),
        Line2D([0], [0], linestyle = '', marker = "o", markerfacecolor = "k", markeredgecolor = 'k', label = "mean", markersize = 5)
    ]
    ax4.legend(handles=legend_elements, loc='lower center', 
            bbox_to_anchor=(0.5, -0.35), 
            ncol=3)

    # Load shapefile 
    MCF = gpd.read_file(data_path+'/MCF2017.shp')
    nonMCF = gpd.read_file(data_path+'/nonMCF2017.shp')
    MCF = MCF.set_geometry('geometry')
    nonMCF = nonMCF.set_geometry('geometry')

    # Coordinates of CWA stations 
    Chiayi_station = gpd.GeoDataFrame(
        {'geometry': [Point(CWAStation['Chiayi']['lon'], CWAStation['Chiayi']['lat'])], 'name': ['Chiayi']}
    )
    Alishan_station = gpd.GeoDataFrame(
        {'geometry': [Point(CWAStation['Alishan']['lon'], CWAStation['Alishan']['lat'])], 'name': ['Alishan']}
    )
    Chiayi_station = Chiayi_station.set_geometry('geometry')
    Alishan_station = Alishan_station.set_geometry('geometry')

    taiwan_coastline = gpd.read_file(data_path+'/Taiwan_WGS84.shp')
    taiwan_coastline = taiwan_coastline.set_geometry('geometry')

    ax = fig.add_subplot(gs[1, 0])

    # Main map setup
    ax.tick_params(labelsize='small')
    ax.set_xticks(np.arange(120.3, 121, 0.1), labels = ['120.3°E', '120.4°E', '120.5°E', '120.6°E', '120.7°E', '120.8°E', '120.9°E', '121°E'])
    ax.set_xticks(np.arange(120.3, 121, 0.05), minor=True)
    ax.set_yticks(np.arange(23.2, 23.8, 0.1), labels = ['23.2°N', '23.3°N', '23.4°N', '23.5°N', '23.6°N', '23.7°N', '23.8°N'])
    ax.set_yticks(np.arange(23.2, 23.8, 0.05), minor=True)
    ax.set_xlim(120.3, 120.9)
    ax.set_ylim(23.35, 23.65)
    ax.grid(alpha=0.5)

    # Plot MCF and other forests
    MCF.plot(ax=ax, color='darkgreen', alpha=0.7)
    nonMCF.plot(ax=ax, color='palegreen', alpha=0.7)

    # Plot CWA stations
    Chiayi_station.plot(ax=ax, color='black', marker='o', markersize=20)
    Alishan_station.plot(ax=ax, color='black', marker='o', markersize=20)

    # Create custom legend handles for MCF and non-MCF
    mcf_patch = mpatches.Patch(color='darkgreen', alpha=0.7, label='MCF')
    nonmcf_patch = mpatches.Patch(color='palegreen', alpha=0.7, label='other forest')
    cwa = Line2D([0], [0], marker='o', color='w', label='CWA station', markerfacecolor='k', markersize=6)

    # Add the legend with custom handles
    ax.legend(handles=[mcf_patch, nonmcf_patch, cwa], loc='upper left')

    # Add labels to the stations
    for x, y, label in zip(Chiayi_station.geometry.x, Chiayi_station.geometry.y, Chiayi_station['name']):
        ax.text(x+0.008, y, label, fontsize=10, ha='left', weight='bold', color='black')
    for x, y, label in zip(Alishan_station.geometry.x, Alishan_station.geometry.y, Alishan_station['name']):
        ax.text(x+0.008, y, label, fontsize=10, ha='left', weight='bold', color='black')

    # Adding contextily base map for better visuals in the main map
    ctx.add_basemap(ax, crs=MCF.crs.to_string(), source=ctx.providers.CartoDB.Positron, zoom=12, attribution=False)

    # Add an inset map
    ax_inset = inset_axes(ax, width="30%", height="30%", 
                        bbox_to_anchor=(-0.2, -0.05, 1.6, 1.6),  # (x0, y0, width, height)
                        loc='lower left', 
                        bbox_transform=ax.transAxes, 
                        borderpad=2)
    ax_inset.set_xlim(119.9, 122.1)
    ax_inset.set_ylim(21.8, 25.4)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    # Plot the overview map content
    taiwan_coastline.plot(ax=ax_inset, color='w', edgecolor='black', linewidth=0.5)
    MCF.plot(ax=ax_inset, color='darkgreen', alpha=0.7)
    nonMCF.plot(ax=ax_inset, color='palegreen', alpha=0.7)

    # Draw a black rectangle in the inset to show the main map area
    main_area_rect = mpatches.Rectangle((120.3, 23.4), 0.6, 0.2, linewidth=0.5, edgecolor='black', facecolor='none')
    ax_inset.add_patch(main_area_rect)

    # Add subfigure labels
    for ax, label in zip([ax1, ax, ax3, ax4], ['(a)', '(b)', '(c)', '(d)']):
        ax.text(0.05, 1.05, label, transform=ax.transAxes, fontsize=14, va='center', ha='center')

    plt.savefig(figure_path+'/Figure1.png',bbox_inches='tight', dpi = dpi)
    plt.show()
