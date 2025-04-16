import os
import netCDF4 as nc
import numpy as np
import tqdm

sigma = 5.67e-8  # W/(m^2·K^4)

# ====== Required input: specify file paths below ======
path_LWDR = r'Y:\MODIS_interpret\reBuild\2022'
path_LWUR = r'Z:\ws\LessRad_LWUR\2022'
OUT_PATH = r'Z:\ws\LessRad_LWNR\2022'

# Get list of LWDR .nc files
os.chdir(path_LWDR)
file_chdir = os.getcwd()
nc_name1 = []  # List of .nc files for LWDR
for root, dirname, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.nc':
            nc_name1.append(path_LWDR + '\\' + file)

# Get list of LWUR .nc files
os.chdir(path_LWUR)
file_chdir = os.getcwd()
nc_name3 = []  # List of .nc files for LWUR
for root, dirname, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.nc':
            nc_name3.append(path_LWUR + '\\' + file)

N = len(nc_name1)

# Loop through each day (24-hour data per file)
for q in tqdm.trange(0, N):
    # Process 24-hour LWDR data
    filename1 = nc_name1[q]
    with nc.Dataset(filename1) as file1:
        file1.set_auto_mask(False)  # Optional
        variables_1 = {x: file1[x][()] for x in file1.variables}
    A = variables_1['LessRad_LWDR']
    # Note: check if longitude ranges need to be adjusted for this year
    # Correction: this is a download format issue — e.g., not needed for 2017, but needed for 2009 and 2015
    LWDR = A
    STORE_LWDR = []
    for k in range(0, 24, 1):
        T = LWDR[k, :, :]
        STORE_LWDR.append(T)
    NSTORE_LWDR = np.array(STORE_LWDR)

    # Process 24-hour LWUR data
    filename3 = nc_name3[q]
    with nc.Dataset(filename3) as file1:
        file1.set_auto_mask(False)  # Optional
        variables_1 = {x: file1[x][()] for x in file1.variables}
    A = variables_1['lwur']
    # Note: check if longitude ranges need to be adjusted for this year
    # Correction: this is a download format issue — e.g., not needed for 2017, but needed for 2009 and 2015
    LWUR = A
    STORE_LWUR = []
    for k in range(0, 24, 1):
        T = LWUR[k][:][:]
        STORE_LWUR.append(T)
    NSTORE_LWUR = np.array(STORE_LWUR)

    # Calculate LWNR = LWDR - LWUR
    predicted = NSTORE_LWDR - NSTORE_LWUR
    predicted_3d = predicted.reshape((24, 3600, 7200))

    # Extract date from filename
    date = filename3[-24:-16]
    output_file = os.path.join(OUT_PATH, f'LessRad_LWNR_{date}_5km_1hour_v1.nc')

    # Write results to NetCDF
    with nc.Dataset(output_file, 'w', format='NETCDF4') as f:
        # Create dimensions and variables
        f.createDimension('time', 24)
        f.createDimension('latitude', 3600)
        f.createDimension('longitude', 7200)
        time_var = f.createVariable('time', 'i4', ('time',))
        lat_var = f.createVariable('latitude', 'f4', ('latitude',))
        lon_var = f.createVariable('longitude', 'f4', ('longitude',))
        lwdr_var = f.createVariable('lwnr', np.int16,
                                    ('time', 'latitude', 'longitude'),
                                    fill_value=-999, zlib=True, complevel=9)

        # Set variable attributes
        time_var.units = 'hours since {}-{}-{} 00:00:00'.format(filename3[-24:-20], filename3[-20:-18], filename3[-18:-16])
        time_var.calendar = 'gregorian'
        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        lwdr_var.units = 'W/m^2'
        lwdr_var.long_name = "longwave net radiation"
        lwdr_var.scale_factor = 0.1

        # Write data
        time = np.arange(24)
        lon = np.linspace(-180, 180, 7200)
        lat = np.linspace(90, -90, 3600)
        time_var[:] = time
        lat_var[:] = lat
        lon_var[:] = lon
        lwdr_var[:] = predicted_3d

    print(f"Data saved to file {output_file}")

