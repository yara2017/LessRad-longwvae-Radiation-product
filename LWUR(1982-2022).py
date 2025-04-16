import os
import netCDF4 as nc
import numpy as np
import tqdm

sigma = 5.67e-8  # W/(m^2·K^4)

# ===== Required input: please specify file paths =====
path_LWDR = r'Y:\MODIS_interpret\reBuild\2022'
path_BBE = r'Y:\BBE\2022'
path_ERA5LST = r'Y:\ws\ERA_5\ERA5LST\2022'
OUT_PATH = r'Z:\ws\LessRad_LWUR\2022'

# Retrieve list of LWDR .nc files
os.chdir(path_LWDR)
file_chdir = os.getcwd()
nc_name1 = []  # List of .nc files for LWDR
for root, dirname, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.nc':
            nc_name1.append(path_LWDR + '\\' + file)

# Retrieve list of BBE .nc files
os.chdir(path_BBE)
file_chdir = os.getcwd()
nc_name2 = []  # List of .nc files for BBE
for root, dirname, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.nc':
            nc_name2.append(path_BBE + '\\' + file)

# Retrieve list of LST .nc files
os.chdir(path_ERA5LST)
file_chdir = os.getcwd()
nc_name3 = []  # List of .nc files for LST
for root, dirname, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.nc':
            nc_name3.append(path_ERA5LST + '\\' + file)

b = len(nc_name2)
N = len(nc_name1)

# First loop: process in 8-day intervals
for p in tqdm.trange(0, b):
    # Process BBE
    filename2 = nc_name2[p]
    with nc.Dataset(filename2) as file1:
        file1.set_auto_mask(False)  # Optional
        variables_1 = {x: file1[x][()] for x in file1.variables}
    A = variables_1['BBE']
    # Note: check if longitude ranges need to be adjusted for the corresponding year
    # Correction: it's a data format issue, e.g., no adjustment needed for 2017, but needed for 2009 and 2022
    ERA5_BBE = A
    STORE_BBE = []
    STORE_BBE.append(A)
    NSTORE_BBE = np.array(STORE_BBE)

    # Second loop: each BBE file is applied to 8 consecutive daily files
    for q in range(p * 8, p * 8 + 8, 1):
        # Process 24-hour LWDR data
        filename1 = nc_name1[q]
        with nc.Dataset(filename1) as file1:
            file1.set_auto_mask(False)  # Optional
            variables_1 = {x: file1[x][()] for x in file1.variables}
        A = variables_1["LessRad_LWDR"]
        # Note: check if longitude ranges need to be adjusted for the corresponding year
        LWDR = A
        STORE_LWDR = []
        for k in range(0, 24, 1):
            T = LWDR[k, :, :]
            STORE_LWDR.append(T)
        NSTORE_LWDR = np.array(STORE_LWDR)
        print(NSTORE_LWDR.shape)

        # Process 24-hour LST data
        filename3 = nc_name3[q]
        with nc.Dataset(filename3) as file1:
            file1.set_auto_mask(False)  # Optional
            variables_1 = {x: file1[x][()] for x in file1.variables}
        A = variables_1['skt']
        # Note: check if longitude ranges need to be adjusted for the corresponding year
        ERA5_LST = A
        STORE_LST = []
        for k in range(0, 24, 1):
            if k == 0:
                T = ERA5_LST[k][:][:]
                STORE_LST.append(T)
            else:
                T = (ERA5_LST[k - 1][:][:] + ERA5_LST[k][:][:]) / 2
                STORE_LST.append(T)
        NSTORE_LST = np.array(STORE_LST)
        print(NSTORE_LST.shape)

        # Calculate LWUR using combined method
        predicted = NSTORE_LWDR * (1 - NSTORE_BBE) + NSTORE_BBE * sigma * NSTORE_LST ** 4

        # Reshape predicted data into 3D
        predicted_3d = predicted.reshape((24, 3600, 7200))

        # Generate date string and output file
        date = filename3[-11:-3]
        output_file = os.path.join(OUT_PATH, f'LessRad_LWUR_{date}.nc')

        # Save results to NetCDF
        with nc.Dataset(output_file, 'w', format='NETCDF4') as f:
            # Create dimensions and variables
            f.createDimension('time', 24)
            f.createDimension('latitude', 3600)
            f.createDimension('longitude', 7200)
            time_var = f.createVariable('time', 'i4', ('time',))
            lat_var = f.createVariable('latitude', 'f4', ('latitude',))
            lon_var = f.createVariable('longitude', 'f4', ('longitude',))
            lwdr_var = f.createVariable('lwur', np.int16,
                                        ('time', 'latitude', 'longitude'),
                                        fill_value=-999, zlib=True, complevel=9)

            # Set variable attributes
            time_var.units = 'hours since {}-{}-{} 00:00:00'.format(filename3[-11:-7], filename3[-7:-5], filename3[-5:-3])
            time_var.calendar = 'gregorian'
            lat_var.units = 'degrees_north'
            lon_var.units = 'degrees_east'
            lwdr_var.units = 'W/m²'
            lwdr_var.standard_name = "longwave upward radiation"
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

