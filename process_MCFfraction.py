import numpy as np
import netCDF4 as nc
import rasterio
import xarray as xr

# Paths to the input files
tif_path = '/work6/L.Teyu/Schulz2017/MCF_WGS84.tif'
nc_path = '/work6/L.Teyu/ReductionRatio/dem20_TCCIPInsolation.nc'

with rasterio.open(tif_path) as dataset:
    tif_data = dataset.read(1)
    # Get the affine transform and the shape (rows, cols)
    transform = dataset.transform
    width = dataset.width
    height = dataset.height
    # Create arrays of pixel coordinates (rows, cols)
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    # Convert pixel coordinates to geographic coordinates
    lon, lat = rasterio.transform.xy(transform, rows, cols, offset='center')
lon_MCF = np.array(lon)[0,:]
lat_MCF = np.array(lat)[:,0]

ncData = nc.Dataset(nc_path)
lon_TCCIP = ncData.variables['lon'][:]
lat_TCCIP = ncData.variables['lat'][:]
llon, llat = np.meshgrid(lon_TCCIP, lat_TCCIP)

dem20 = ncData.variables['dem20'][:,:]
Landmask = ~np.isnan(dem20) ## mask of land areas
maskllon = llon[Landmask]
maskllat = llat[Landmask]

mcf_TCCIP = np.empty((525, 575, 4))
for i in range(len(maskllon)):
    ilon_MCF = np.argmin(np.abs(lon_MCF - maskllon[i]))
    ilat_MCF = np.argmin(np.abs(lat_MCF - maskllat[i]))
    array = tif_data[int(ilat_MCF-2):int(ilat_MCF+3), int(ilon_MCF-2):int(ilon_MCF+3)]
    array[np.where(array == 3)] = 0
    counts = [np.count_nonzero(array == num) for num in range(4)]
    ilon_TCCIP = np.argmin(np.abs(lon_TCCIP - maskllon[i]))
    ilat_TCCIP = np.argmin(np.abs(lat_TCCIP - maskllat[i]))
    print(i)
    mcf_TCCIP[ilat_TCCIP, ilon_TCCIP, 0] = counts[0]
    mcf_TCCIP[ilat_TCCIP, ilon_TCCIP, 1] = counts[1]
    mcf_TCCIP[ilat_TCCIP, ilon_TCCIP, 2] = counts[2]
is_integer = np.equal(mcf_TCCIP, mcf_TCCIP.astype(int))
mcf_TCCIP[~is_integer] = np.nan
np.save("/work6/L.Teyu/ReductionRatio/MCFfraction_TCCIP.npy", mcf_TCCIP)
