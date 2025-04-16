import netCDF4 as nc
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
import tqdm
import joblib

# Input file paths
path_ERA5LWDN = r'Y:\ws\ERA_5\ERA5LWDR'
path_ERA5TCWV = r'Y:\ws\ERA_5\tcwv'
path_ERA5LST = r'Y:\ws\ERA_5\skt'
path = r'Y:\ws'
OUT_PATH = r'Y:\ws\LessRad_LWDR\1998'

# Retrieve list of .nc files for each variable
os.chdir(path_ERA5LWDN)
file_chdir = os.getcwd()
nc_name1 = []  # List of .nc files for LWDR
for root, dirname, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.nc':
            nc_name1.append(path_ERA5LWDN + '\\' + file)

os.chdir(path_ERA5TCWV)
file_chdir = os.getcwd()
nc_name2 = []  # List of .nc files for TCWV
for root, dirname, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.nc':
            nc_name2.append(path_ERA5TCWV + '\\' + file)

os.chdir(path_ERA5LST)
file_chdir = os.getcwd()
nc_name3 = []  # List of .nc files for LST
for root, dirname, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.nc':
            nc_name3.append(path_ERA5LST + '\\' + file)

N = len(nc_name1)
aaaaa = 102
for p in tqdm.trange(0, N):
    # Process 24 hourly data together for daily average calculations of water vapor and temperature
    filename1 = nc_name1[p]
    with nc.Dataset(filename1) as file1:
        file1.set_auto_mask(False)  # Optional
        variables_1 = {x: file1[x][()] for x in file1.variables}
    A = variables_1['lwdr'] / 3600
    # Note: Check if longitude range needs adjustment for the corresponding year.
    # Correction: It's actually a data download format issue; for example, 2017 doesn't need adjustment, but 2009 and 2015 do.
    ERA5_STRD = A
    STORE_LWDN = []
    for k in range(0, 24, 1):
        T = ERA5_STRD[k, :, :]
        STORE_LWDN.append(T)
    NSTORE_LWDN = np.array(STORE_LWDN)

    # Process water vapor
    filename2 = nc_name2[p]
    with nc.Dataset(filename2) as file1:
        file1.set_auto_mask(False)  # Optional
        variables_1 = {x: file1[x][()] for x in file1.variables}
    A = variables_1['tcwv']
    # Note: Check if longitude range needs adjustment for the corresponding year.
    # Correction: It's actually a data download format issue; for example, 2017 doesn't need adjustment, but 2009 and 2015 do.
    ERA5_TCWV = A
    STORE_TCWV = []
    for k in range(0, 24, 1):
        if k == 0:
            T = ERA5_TCWV[k][:][:]
            STORE_TCWV.append(T)
        else:
            T = (ERA5_TCWV[k - 1][:][:] + ERA5_TCWV[k][:][:]) / 2
            STORE_TCWV.append(T)
    NSTORE_TCWV = np.array(STORE_TCWV)

    # Process temperature
    filename3 = nc_name3[p]
    with nc.Dataset(filename3) as file1:
        file1.set_auto_mask(False)  # Optional
        variables_1 = {x: file1[x][()] for x in file1.variables}
    A = variables_1['skt']
    # Note: Check if longitude range needs adjustment for the corresponding year.
    # Correction: It's actually a data download format issue; for example, 2017 doesn't need adjustment, but 2009 and 2015 do.
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

    # Prepare data for prediction
    X = np.concatenate((NSTORE_LWDN.flatten()[np.newaxis],
                        NSTORE_TCWV.flatten()[np.newaxis],
                        NSTORE_LST.flatten()[np.newaxis]))
    X = np.transpose(X)

    # Load the trained model
    loaded_model = joblib.load(os.path.join(path, '2_1.pkl'))
    # Predict using the model
    predicted = loaded_model.predict(X)
    # Reshape predicted data to 3D
    predicted_3d = predicted.reshape((24, 3600, 7200))

    # Save the predicted data to a NetCDF file
    date = filename1[-11:-3]
    output_file = os.path.join(OUT_PATH, f'LessRad_LWDR_{date}.nc')
    with nc.Dataset(output_file, 'w', format='NETCDF4') as f:
        # Create dimensions and variables
        f.createDimension('time', 24)
        f.createDimension('latitude', 3600)
        f.createDimension('longitude', 7200)
        time_var = f.createVariable('time', 'i4', ('time',))
        lat_var = f.createVariable('latitude', 'f4', ('latitude',))
        lon_var = f.createVariable('longitude', 'f4', ('longitude',))
        lwdr_var = f.createVariable('lwdr', np.int16,
                                    ('time', 'latitude', 'longitude'),
                                    fill_value=-999, zlib=True, complevel=9)
        # Set variable attributes
        time_var.units = 'hours since {}-{}-{} 00:00:00'.format(filename1[-11:-7], filename1[-7:-5], filename1[-5:-3])
        time_var.calendar = 'gregorian'
        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        lwdr_var.units = 'W/m^2'
        # Write data to variables
        time = np.arange(24)
        lon = np.linspace(-180, 180, 7200)
        lat = np.linspace(90, -90, 3600)
        time_var[:] = time
        lat_var[:] = lat
        lon_var[:] = lon
        lwdr_var[:] = predicted_3d
    print(f"save to {output_file} ")
