import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from pyproj import CRS
import multiprocessing
from rasterio import fill
from scipy import ndimage
from collections import defaultdict
from functools import partial
from pyresample import geometry as geom
from pyresample.bucket import BucketResampler
import netCDF4 as nc
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# Return CERES LWDR block data dictionary, where the key is the block's top-left coordinates and the value is the block's xarray data (longitude, latitude, time)
# Read CERES data without parallel processing
def read_era5_nc_file(start_date, num_day_of_read, step):
    # Read ERA5 LWDR data; input parameters: start date and number of days to read
    # start_date: pd.Timestamp;
    era5_lwdr_data_path = r"E:\ERA5LWDR"
    era5_lwdr_datas = []
    for day_i in range(num_day_of_read):
        date = start_date + pd.Timedelta(days=day_i)
        year = date.year
        month = date.month
        day = date.day
        era5_lwdr_file = os.path.join(
            era5_lwdr_data_path + "/" + str(year),
            "era5__%d%02d%02d.nc" % (year, month, day),
        )
        # Only read the required time period and exclude irrelevant variable ttrc
        era5_lwdr_dataset = xr.open_dataset(
            era5_lwdr_file, decode_times=True, decode_cf=True, engine="netcdf4"
        )
        # Read data for the day; unit is J/m2, so divide by 3600 s
        era5_lwdr_data = era5_lwdr_dataset["strd"] / 3600
        # Convert data type to float16
        era5_lwdr_data = era5_lwdr_data.astype(np.float16)
        era5_lwdr_data.attrs["units"] = "W/m^2"
        # If necessary, convert longitude from 0-360 to -180~180
        era5_lwdr_data = (
            era5_lwdr_data.assign_coords(
                longitude=(((era5_lwdr_data.longitude + 180) % 360) - 180)
            )
            .sortby("longitude")
            .astype(np.float16)
        )
        # Append the day’s data to the list (ERA5 data shape: (721, 1440, 288), latitude range: [-90, 90] with 0.25 spacing, longitude range: [-180, 179.75])
        era5_lwdr_datas.append(era5_lwdr_data)
    # Concatenate along the time dimension
    all_era5_lwdr_data = xr.concat(era5_lwdr_datas, dim="time")
    # Pad along the longitude dimension (mirror padding) by extending one element at the edge
    all_era5_lwdr_data = np.pad(all_era5_lwdr_data, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=np.nan)
    all_era5_lwdr_data[:, :, -1] = all_era5_lwdr_data[:, :, 0]
    time = pd.date_range(start_date, start_date + pd.Timedelta(hours=num_day_of_read * 24 - 1), freq='1h')
    new_lats = np.linspace(90, -90, 721)
    new_lons = np.linspace(-180, 180, 1441)
    all_era5_lwdr_da = xr.DataArray(all_era5_lwdr_data, coords=[time, new_lats, new_lons],
                                   dims=['time', 'latitude', 'longitude'])
    # Partition the data by the latitude-longitude grid
    era5_lwdr_block_dict = {}
    for lat_key in np.arange(90, -90, -step):
        for lon_key in np.arange(-180, 180, step):
            era5_lwdr_block_dict[(lat_key, lon_key)] = all_era5_lwdr_da.sel(
                latitude=slice(lat_key + 0.25, lat_key - step - 0.25),
                longitude=slice(lon_key - 0.25, lon_key + step + 0.25)
            )
    print("Reading CERES LWDR data done!")
    return era5_lwdr_block_dict


# Perform resampling and gridding for multiple MODIS LWDR files
def resample_modis_lwdr_files(modis_files, step, resolution):
    modis_lwdr_block_dict = defaultdict(list)
    epsg, proj, pName = "4326", "longlat", "Geographic"
    for modis_file in modis_files:
        ds = nc.Dataset(modis_file)
        LWDR_data = ds.variables['LWDR'][:]
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]
        lons = np.where(lons == -999, np.nan, lons)
        lats = np.where(lats == -999, np.nan, lats)
        year = int(modis_file.split('.')[1][1:5])
        doy = int(modis_file.split('.')[1][5:])
        target_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        hour = int(modis_file.split('.')[2][0:2])
        minute = int(modis_file.split('.')[2][2:])
        date_str = "%d%02d%02d" % (year, target_date.month, target_date.day)
        time_str = "%02d%02d%02d" % (hour, minute, 0)
        at_time = '%s %s' % (date_str, time_str)
        ds.close()
        lons = np.where(lons == -999, np.nan, lons)
        lats = np.where(lats == -999, np.nan, lats)
        min_lon, max_lon = np.nanmin(lons), np.nanmax(lons)
        # If the data crosses the 180° meridian (eastern boundary longitude < western boundary longitude)
        datas = []
        if (max_lon - min_lon) > 180:
            mask1 = lons > 0
            lons1, lats1, LWDR_data1 = lons[mask1], lats[mask1], LWDR_data[mask1]
            datas.append((lons1, lats1, LWDR_data1))
            lons2, lats2, LWDR_data2 = lons[~mask1], lats[~mask1], LWDR_data[~mask1]
            datas.append((lons2, lats2, LWDR_data2))
        else:
            datas.append((lons, lats, LWDR_data))
        # First perform resampling and then grid partitioning
        for data in datas:
            use_lon, use_lat, use_LWDR = data
            min_lon, min_lat, max_lon, max_lat = (
                np.nanmin(use_lon),
                np.nanmin(use_lat),
                np.nanmax(use_lon),
                np.nanmax(use_lat),
            )
            start_lon_key = np.floor((min_lon + 180) / step) * step - 180
            end_lon_key = np.ceil((max_lon + 180) / step) * step - 180
            start_lat_key = 90 - np.floor((90 - max_lat) / step) * step
            end_lat_key = 90 - np.ceil((90 - min_lat) / step) * step
            cols = int((end_lon_key - start_lon_key) / resolution)
            rows = int((start_lat_key - end_lat_key) / resolution)
            areaExtent = (start_lon_key, end_lat_key, end_lon_key, start_lat_key)
            areaDef = geom.AreaDefinition(area_id=epsg, description=pName, proj_id=proj,
                                          projection=CRS('epsg:4326'),
                                          height=rows, width=cols, area_extent=areaExtent)
            resampler = BucketResampler(areaDef, da.from_array(use_lon), da.from_array(use_lat))
            result = resampler.get_average(da.from_array(use_LWDR), fill_value=-999).compute()
            num_small_grid = round(step / resolution)
            row_part_num = round(rows / num_small_grid)
            col_part_num = round(cols / num_small_grid)
            for row_part_i in range(row_part_num):
                lat_key = start_lat_key - row_part_i * step
                for col_part_i in range(col_part_num):
                    lon_key = start_lon_key + col_part_i * step
                    grid_LWDR = result[
                        row_part_i * num_small_grid: (row_part_i + 1) * num_small_grid,
                        col_part_i * num_small_grid: (col_part_i + 1) * num_small_grid
                    ].astype(np.float16)
                    modis_lwdr_block_dict[(lat_key, lon_key)].append((at_time, grid_LWDR))
    return modis_lwdr_block_dict


# Return MODIS LWDR block data dictionary, where the key is the block's top-left coordinates and the value is a list of (time, image block data)
def read_modis_lwdr_file(start_date, num_day_of_read, step, resolution, num_process):
    print("Start reading MODIS LWDR data!")
    all_modis_lwdr_block_dict = defaultdict(list)
    all_modis_files = []
    for day_i in range(num_day_of_read):
        use_day = start_date + pd.Timedelta(days=day_i)
        day_of_year = use_day.dayofyear
        year = use_day.year
        myd_modis_data_path = r"F:\MYD\%d" % year
        mod_modis_data_path = r"E:\MODIS_LWDR_global_noGeo_NC\MOD\%d" % year
        modis_files1 = list(
            glob.glob(
                os.path.join(mod_modis_data_path, "%03d" % day_of_year, "*+LWDR.nc")
            )
        )
        modis_files2 = list(
            glob.glob(
                os.path.join(myd_modis_data_path, "%03d" % day_of_year, "*+LWDR.nc")
            )
        )
        days_modis_files = modis_files1 + modis_files2
        all_modis_files.extend(days_modis_files)
    pool = multiprocessing.Pool(num_process)
    all_modis_files_list = np.array_split(all_modis_files, num_process)
    params = zip(all_modis_files_list, [step] * num_process, [resolution] * num_process)
    modis_lwdr_block_dict_result = pool.starmap(resample_modis_lwdr_files, params)
    pool.close()
    pool.join()
    for modis_lwdr_block_dict_temp in modis_lwdr_block_dict_result:
        for key, value in modis_lwdr_block_dict_temp.items():
            all_modis_lwdr_block_dict[key].extend(value)
    print("Reading MODIS LWDR data done!")
    return all_modis_lwdr_block_dict


def time_space_interpolation(block_key, CERES_LWDR_origin_array, swaths, start_date, num_day_of_read, step, resolution):
    cols = int(step / resolution)
    rows = int(step / resolution)
    start_lat, start_lon = block_key
    print("Space-time scaling at block: lat:%d, lon:%d" % (start_lat, start_lon))
    end_lat, end_lon = start_lat - step, start_lon + step
    end_date = start_date + pd.Timedelta(days=num_day_of_read)
    small_offset = np.round(resolution / 2, 3)
    lat_coords = np.linspace(start_lat - small_offset, end_lat + small_offset, rows)
    lon_coords = np.linspace(start_lon + small_offset, end_lon - small_offset, cols)
    origin_modis_grid_LWDR = xr.DataArray(
        -999 * np.ones((288 * num_day_of_read, rows, cols,), dtype=np.float16),
        [
            (
                "time",
                pd.date_range(start_date, end_date, freq="5min", inclusive="left"),
            ),
            ("latitude", lat_coords),
            ("longitude", lon_coords),
        ],
        name="LessRad LWDR",
        attrs={"long name": "longwave downward radiation", "units": "W m-2"},
    )
    for swath in swaths:
        at_time = swath[0]
        use_LWDR = swath[1]
        origin_modis_grid_LWDR.loc[at_time, start_lat:end_lat, start_lon:end_lon] = use_LWDR
    origin_modis_grid_LWDR = (
        origin_modis_grid_LWDR.where(origin_modis_grid_LWDR != -999, np.nan)
    )
    origin_modis_grid_LWDR = origin_modis_grid_LWDR.astype(np.float16)
    CERES_LWDR_origin_array = CERES_LWDR_origin_array.astype(np.float16)
    CERES_LWDR_temp_array = CERES_LWDR_origin_array.interp(latitude=origin_modis_grid_LWDR.latitude,
                                                           longitude=origin_modis_grid_LWDR.longitude,
                                                           method='linear').astype(np.float16)
    OLR_delta_temp_array = origin_modis_grid_LWDR - CERES_LWDR_temp_array.interp(time=origin_modis_grid_LWDR.time,
                                                                                 method='linear').astype(np.float16)
    del origin_modis_grid_LWDR
    OLR_delta_temp_array = (OLR_delta_temp_array.ffill(dim='time') + OLR_delta_temp_array.bfill(dim='time')) / 2
    OLR_delta_temp_array = OLR_delta_temp_array.resample(time='1h').mean().astype(np.float16)
    modis_LWDR_array = CERES_LWDR_temp_array + OLR_delta_temp_array
    del CERES_LWDR_temp_array, OLR_delta_temp_array
    effect_time = pd.date_range(start_date + pd.Timedelta(days=1), end_date - pd.Timedelta(days=1), freq='1h',
                                inclusive='left')
    modis_LWDR_array = modis_LWDR_array.sel(time=effect_time).astype(np.float16)
    return modis_LWDR_array


def fillled_nan_value_by_rasterio(use_matrix):
    if use_matrix.isnull().any():
        matrix = use_matrix.values
        mask = np.isnan(matrix)
        matrix = fill.fillnodata(
            matrix, mask=~mask, max_search_distance=15, smoothing_iterations=0
        )
        use_matrix.values = matrix
    return use_matrix


def sliding_window_filter(use_matrix):
    if use_matrix.isnull().sum():
        matrix = use_matrix.values
        std_multiplier = 1
        max_window_size = 30
        rows, cols = matrix.shape
        window_size = 3
        for i in range(0, rows, window_size):
            for j in range(0, cols, window_size):
                window_slice = (slice(i, i + window_size), slice(j, j + window_size))
                window = matrix[window_slice]
                valid = np.array([value for value in window.flatten() if value > 0])
                if valid.size > 2:
                    mean = np.mean(valid)
                    std = np.std(valid)
                    lower_bound = mean - std_multiplier * std
                    upper_bound = mean + std_multiplier * std
                    window = np.where(
                        np.logical_or(window < lower_bound, window > upper_bound),
                        np.nan,
                        window,
                    )
                    valid_values = np.array(
                        [value for value in window.flatten() if value > 0]
                    )
                    num_valid_values = len(valid_values)
                else:
                    valid_values = np.empty(0)
                    num_valid_values = 0
                while num_valid_values < 3 and window_size <= max_window_size:
                    window_size = window_size + 2
                    start_row = i
                    end_row = i + window_size
                    start_col = j
                    end_col = j + window_size
                    if i + window_size >= rows:
                        start_row = rows - 1 - window_size
                        end_row = rows - 1
                    if j + window_size >= cols:
                        start_col = cols - 1 - window_size
                        end_col = cols - 1
                    window_slice = (
                        slice(start_row, end_row),
                        slice(start_col, end_col),
                    )
                    window = matrix[window_slice]
                    valid = np.array([value for value in window.flatten() if value > 0])
                    if valid.size > 2:
                        mean = np.mean(valid)
                        std = np.std(valid)
                        lower_bound = mean - std_multiplier * std
                        upper_bound = mean + std_multiplier * std
                        window = np.where(
                            np.logical_or(window < lower_bound, window > upper_bound),
                            np.nan,
                            window,
                        )
                        valid_values = np.array(
                            [value for value in window.flatten() if value > 0]
                        )
                        num_valid_values = len(valid_values)
                    else:
                        valid_values = np.empty(0)
                        num_valid_values = 0
                window_size = 3
                window_slice = (slice(i, i + window_size), slice(j, j + window_size))
                matrix[window_slice] = np.where(
                    np.isnan(matrix[window_slice]),
                    np.mean(valid_values),
                    matrix[window_slice],
                )
        use_matrix.values = matrix
    return use_matrix

# Median filtering function
def func(block):
   return ndimage.median_filter(block, size=(5, 5))

# Filter LWDR data
def filter_out_LWDR(out_part_array):
    at_time = out_part_array.coords["time"]
    image_array = out_part_array.astype(np.float32).copy()
    image_array = image_array.chunk({"latitude": 100, "longitude": 100})
    image_array = image_array.map_blocks(fillled_nan_value_by_rasterio, template=image_array).compute()
    dask_array = da.from_array(image_array, chunks=(180, 360))
    del image_array
    out_array = dask_array.map_overlap(func, depth={0: 2, 1: 2},
                                       boundary={0: 'reflect', 1: 'reflect'}, dtype=np.float32).compute()
    out_part_array.values = out_array.astype(np.float16)
    print("Filtering data at %s done!" % (at_time.dt.strftime("%Y-%m-%d %H:%M:%S").values))
    return out_part_array


def parallel_filter_out_LWDR(out_LWDR_array, num_process_of_filtering):
    low_bound, high_bound = 55, 550
    out_LWDR_array = out_LWDR_array.where(
        (out_LWDR_array > low_bound) & (out_LWDR_array < high_bound)
    )
    filter_pool = multiprocessing.Pool(num_process_of_filtering)
    time_size = out_LWDR_array.time.size
    params = [out_LWDR_array[time_i] for time_i in range(time_size)]
    filter_LWDRs = filter_pool.map(filter_out_LWDR, params)
    filter_pool.close()
    filter_pool.join()
    out_LWDR_array = xr.concat(filter_LWDRs, dim="time")
    return out_LWDR_array


def write_out_LWDR_array(out_LWDR_array, out_date, out_path):
    out_date_str = out_date.strftime("%Y%m%d")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    day_of_year = out_date.timetuple().tm_yday
    year = out_date.year
    out_file = r"D:\duyh\MODIS_interpret\%d\LWDR_%d%03d.nc" % (year, year, day_of_year)
    out_LWDR_array.name = "LessRad LWDR"
    out_LWDR_array.to_netcdf(
        out_file,
        engine="h5netcdf",
        format="NETCDF4",
        encoding={
            "LessRad LWDR": {
                "dtype": "int16",
                "scale_factor": 0.1,
                "add_offset": 0,
                "zlib": True,
                "_FillValue": -999,
            }
        },
    )
    print("Writing file of date:%s done!" % (out_date_str))


if __name__ == "__main__":
    time1 = pd.Timestamp.now()
    origin_date = pd.Timestamp("2022-06-25")
    year = origin_date.year
    day_end = 366
    num_day_of_read = 18
    step = 30
    resolution = 0.05
    era5_lwdr_data_path = r"E:\ERA5LWDR\%d" % year
    out_path = r"D:\duyh\MODIS_interpret\%d" % year
    for day_i in range(0, day_end, num_day_of_read - 2):
        start_date = origin_date + pd.Timedelta(days=day_i)
        end_date = start_date + pd.Timedelta(days=num_day_of_read)
        all_rows = int(180 / resolution)
        all_cols = int(360 / resolution)
        all_lat_coords = np.linspace(
            90 - (resolution / 2), -90 + (resolution / 2), all_rows
        ).round(3)
        all_lon_coords = np.linspace(
            -180 + (resolution / 2), 180 - (resolution) / 2, all_cols
        ).round(3)
        effect_time = pd.date_range(
            start_date + pd.Timedelta(days=1),
            end_date - pd.Timedelta(days=1),
            freq="1h",
            inclusive="left",
        )
        CERES_LWDR_block_dict = read_era5_nc_file(start_date, num_day_of_read, step)
        num_proces_of_reading = 50
        modis_LWDR_block_dict = read_modis_lwdr_file(
            start_date,
            num_day_of_read,
            step,
            resolution,
            num_proces_of_reading,
        )
        num_process_of_space_time_scaling = 16
        pool = multiprocessing.Pool(num_process_of_space_time_scaling)
        block_keys = []
        CERES_LWDR_block_list = []
        modis_swaths_list = []
        for lat_key in np.arange(90, -90, -step):
            for lon_key in np.arange(-180, 180, step):
                block_key = lat_key, lon_key
                block_keys.append(block_key)
                CERES_LWDR_block_list.append(CERES_LWDR_block_dict[block_key])
                swaths = modis_LWDR_block_dict[block_key]
                modis_swaths_list.append(swaths)
        del CERES_LWDR_block_dict, modis_LWDR_block_dict
        params = zip(block_keys, CERES_LWDR_block_list, modis_swaths_list)
        partial_time_space_interpolation = partial(
            time_space_interpolation,
            start_date=start_date,
            num_day_of_read=num_day_of_read,
            step=step,
            resolution=resolution,
        )
        results = pool.starmap(partial_time_space_interpolation, params)
        del CERES_LWDR_block_list, modis_swaths_list
        pool.close()
        pool.join()
        global_modis_LWDR_grid = xr.combine_by_coords(
            results, combine_attrs="drop_conflicts"
        )
        del results
        print("Space-time scaling done!")
        out_dates = [start_date + pd.Timedelta(days=i + 1) for i in range(num_day_of_read - 2)]
        out_LWDR_arrays = [
            global_modis_LWDR_grid.sel(time=slice(out_date, out_date + pd.Timedelta(hours=23))).astype(np.float16) for
            out_date in out_dates]
        del global_modis_LWDR_grid
        num_process_of_filtering = 6
        len_dates = len(out_dates)
        print("start_filtering!")
        for out_i in range(len_dates):
            out_LWDR_array = out_LWDR_arrays[out_i]
            temp_array = parallel_filter_out_LWDR(
                out_LWDR_array, num_process_of_filtering
            )
            out_LWDR_arrays[out_i].values = temp_array.values
        print("Filtering data done!")
        num_process_of_writing = len(out_dates)
        pool = multiprocessing.Pool(num_process_of_writing)
        params = zip(out_LWDR_arrays, out_dates, [out_path] * num_process_of_writing)
        pool.starmap(write_out_LWDR_array, params)
        pool.close()
        pool.join()
        print("Writing all dates' product files done!")
    time2 = pd.Timestamp.now()
    print("Time cost:", time2 - time1)
