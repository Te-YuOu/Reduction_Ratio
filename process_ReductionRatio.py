# %%
import os
import re
import numpy as np
from datetime import datetime, timedelta
import time
import netCDF4 as nc

# Paths and data loading
Himawari_path = "/nas1/TCCIP_satellite/Himawari"
output_path = '/work6/L.Teyu/ReductionRatio'

data_path = '/work6/L.Teyu/ReductionRatio'
figure_path = '/work/home/L.Teyu/ReductionRatio'
shp_path = '/work6/L.Teyu/Schulz2017/'

dem20 = nc.Dataset(data_path + '/dem20_TCCIPInsolation.nc')
lon = dem20.variables['lon'][:]
lat = dem20.variables['lat'][:]
ele = dem20.variables['dem20'][:, :]
Landmask = ~np.isnan(ele)

# Load example file to get lat/lon
ncData_ex = nc.Dataset(Himawari_path + '/insotwf1h_2016010106.nc')
lon, lat = ncData_ex.variables['longitude'][:], ncData_ex.variables['latitude'][:]

# Pre-allocate arrays
MonthHour_max = np.full((12, 14, 525, 575), np.nan)
MonthHour_mean = np.full((12, 14, 525, 575), np.nan)

# Process data for each month-hour combination
for mm in range(1, 13):
    for hh in range(6, 20):
        MonthHour_data = np.full((1500, 525, 575), np.nan)  
        index_dd = 0
        if mm == 1: 
            for imm in [12, 1, 2]:
                for yyyy in range(2015, 2020):
                    for dd in range(1, 32):
                        try:
                            filepath = Himawari_path + f'/insotwf1h_{yyyy:04}{imm:02}{dd:02}{hh:02}.nc'
                            ncData = nc.Dataset(filepath)
                            MonthHour_data[index_dd, :, :] = ncData.variables['data'][:, :] / 0.0036
                            index_dd += 1
                        except FileNotFoundError:
                            print(f'File missing for: {yyyy}-{imm:02}-{dd:02} {hh:02}')
                            continue            
        elif mm == 12:
            for imm in [11, 12, 1]:
                for yyyy in range(2015, 2020):
                    for dd in range(1, 32):
                        try:
                            filepath = Himawari_path + f'/insotwf1h_{yyyy:04}{imm:02}{dd:02}{hh:02}.nc'
                            ncData = nc.Dataset(filepath)
                            MonthHour_data[index_dd, :, :] = ncData.variables['data'][:, :] / 0.0036
                            index_dd += 1
                        except FileNotFoundError:
                            print(f'File missing for: {yyyy}-{imm:02}-{dd:02} {hh:02}')
                            continue
        else:
            for imm in range(mm-1, mm+2):
                for yyyy in range(2015, 2020):
                    for dd in range(1, 32):
                        try:
                            filepath = Himawari_path + f'/insotwf1h_{yyyy:04}{imm:02}{dd:02}{hh:02}.nc'
                            ncData = nc.Dataset(filepath)
                            MonthHour_data[index_dd, :, :] = ncData.variables['data'][:, :] / 0.0036
                            index_dd += 1
                        except FileNotFoundError:
                            print(f'File missing for: {yyyy}-{imm:02}-{dd:02} {hh:02}')
                            continue

        # Calculate max and mean while excluding NaNs
        MonthHour_max[mm-1, hh-6, :, :] = np.nanmax(MonthHour_data, axis=0)
        MonthHour_mean[mm-1, hh-6, :, :] = np.nanmean(MonthHour_data, axis=0)
        
        print(f'Processed Month: {mm}, Hour: {hh}')

# Save results
np.save(output_path+'/MonthHour_max_20152019_Before.npy', MonthHour_max)
np.save(output_path+'/MonthHour_mean_20152019_Before.npy', MonthHour_mean)

# %%
'''
MonthHour_max = np.load(output_path+'/MonthHour_max_20152019_Before.npy')
MonthHour_max[7, 2, :, :] = MonthHour_max_before[6, 2, :, :]
MonthHour_max[11, 2, :, :] = MonthHour_max_before[0, 2, :, :]
MonthHour_mean = np.load(output_path+'/MonthHour_mean_20152019_Before.npy')
'''
#%%
ReductionRatio_daily = []
for yyyy in range(2015, 2020): 
    for mm in range(1, 13):
        maximum = MonthHour_max[mm-1, :, :, :]
        for dd in range(1, 32):
            diurnal = np.tile(np.nan, (14, 525, 575))
            for hh in range(6, 20):
                try:
                    filepath = Himawari_path + f'/insotwf1h_{yyyy:04}{mm:02}{dd:02}{hh:02}.nc'
                    ncData = nc.Dataset(filepath)
                    diurnal[(hh-6),:,:] = ncData.variables['data'][:,:]/0.0036  
                except FileNotFoundError:
                    print(f'File missing for: {yyyy}-{mm:02}-{dd:02} {hh:02}')
                    break
            if np.nansum(diurnal) != 0:
                ReductionRatio = (np.nansum(maximum, axis = 0) - np.nansum(diurnal, axis = 0))/np.nansum(maximum, axis = 0)
                ReductionRatio_daily.append(ReductionRatio)
            else:
                if ((yyyy == 2019) & (dd == 1)):
                    ReductionRatio = np.full((525, 575), np.nan)
                    ReductionRatio_daily.append(ReductionRatio)
                else:
                    pass
print(np.array(ReductionRatio_daily).shape)
np.save(output_path+'/ReductionRatio_daily_20152019.npy', np.array(ReductionRatio_daily))
#%%