#import packages
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cartopy
from cartopy import util
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cft
import matplotlib.gridspec as gridspec
import numpy as np
import h5py
import pandas as pd
# %matplotlib inline

#
def rename_variables(var):
    ''' define your function '''
    if var == SENS:
        return 'long_name'
    else:
        return var

#sst from argo vs satellite plot, LABEL IS NOT CORRECT, SAYS FLOAT 36
def sat_argo_temp_plot(argo_dataset):
    ''' 
    plots argo float sea surface temp (black) and satellite sst (red) over time to compare accuracy
    /PARAMS: argo_dataset: float dataset title (ex: df73)
    /sst_sat selects sst data times based on the times the float collected data
    /must download sst as one file and argo as another file
    '''
    df = argo_dataset
    sst_sat=ds.sst.sel(time= df.JULD,method='nearest').sel(lat=df.LATITUDE,method='nearest').sel(lon=360+df36.LONGITUDE,method='nearest').load()
    fig = plt.figure(figsize=(10,5))
    sst_pts = plt.scatter(sst_sat.time,sst_sat,s=10,color='red',label='NOAA satellite')
    argo_pts = plt.scatter(df.JULD,df.TEMP_ADJUSTED.isel(N_LEVELS=0),s=10,color='black',label='float 36')
    ylabel=plt.ylabel('SST (ºC)')
    title=plt.title('SST Data Comparison')
    legend=plt.legend()
    return fig, sst_pts, argo_pts, ylabel, title, legend

#sst from argo float, with min values in blue (suspected TIW) by using a mask
def min_sst_plot(argo_dataset, limit):
    ''' 
    plots all float sst vs time in grey, then masks all but the min temps and plots those in blue
    /PARAMS: argo_dataset: float dataset title (ex: df73); limit: limiting temp value to isolate TIW (ex:24.5)
    /mask: masks all values above the limit from the dataset's surface temperature values
    /must download float data in one file
    '''
    df = argo_dataset
    sst = df.TEMP_ADJUSTED.isel(N_LEVELS=0)
    mask = sst <= limit
    fig = plt.figure(figsize=(10,5))
    all_temps = plt.scatter(df['JULD'],df['TEMP_ADJUSTED'].isel(N_LEVELS=0),s=10,color='grey')
    min_temps = plt.scatter(df['JULD'].where(mask),df['TEMP_ADJUSTED'].isel(N_LEVELS=0).where(mask),s=20,color='blue')
    ylabel = plt.ylabel('SST (ºC)')
    title = plt.title('TIW profile')
    return fig, all_temps, min_temps, ylabel, title

#plot vertical profile (ex: pressure vs temp/nitrate/etc), DOES NOT HAVE AXIS LABELS
def vert_prof_plot(argo_dataset, limit, sensorx, sensory):
    ''' 
    plots one sensor vs another, must have same profile numbers
    /works well with pressure as sensory and a bgc sensor as sensorx
    /PARAMS: argo_dataset: float dataset title (ex: df73), limit: limiting temp value to isolate TIW (ex:24.5), sensorx: sensor that changes with depth (ex: 'NITRATE_ADJUSTED'), sensor y: es: 'PRES_ADJSUTED'
    /mask: masks all values above the limit from the dataset's surface temperature values
    /must download float data in one file 
    '''
    df = argo_dataset
    sst = df.TEMP_ADJUSTED.isel(N_LEVELS=0)
    mask = sst <= limit
    fig = plt.figure(figsize=(10,5))
    all_profs = plt.scatter(df[sensorx], df36[sensory], c='grey',s=1)
    masked_profs = plt.scatter(df[sensorx].where(mask), df[sensory].where(mask), c='red',s=3)
    ylim = plt.ylim(2000,0)
    return fig, all_profs, masked_profs, ylim

#apply a mask function to get minimum temperature values
def get_masked_values(argo_dataset, limit):
    '''
    takes sst values and masks all that are above a certain limit (gives you the minimum temperatures)
    /PARAMS: argo_dataset: float dataset title (ex: df73), limit: limiting temp value to isolate TIW (ex:24.5)
    /mask: masks all values above the limit from the dataset's surface temperature values
    /masked_var: turns float values into a dataset
    /must download float data in one file    
    '''
    df = argo_dataset
    variable = df.TEMP_ADJUSTED.isel(N_LEVELS=0).load()
    mask = variable <= limit
    masked_var = variable.where(mask, drop=True)
    return masked_var

#change dimensions of the dataset
def change_dims(dataset_path, out_dim, in_dim):
    '''
    change dimensions of the dataset, used to make time a dimension to be used for taking means
    /PARAMS: dataset_path: acutal path to dataset from computer (ex: 'data/Equatorial_Pacific/ep-5906473_Sprof.nc'); out_dim: dimension to be replaced (ex:'N_PROF'); in_dim: variable being made the new dimension (ex:'JULD')
    /in and out dimensions must have the same number of profiles (be the same size/shape..... idk what else would work in place of JULD or N_PROF if I wanted to change it to a different dimension)
    /must load the dataset new as it reconfigures the entire dataset; would suggest giving it a different name than the dataset with the old dimensions
    '''
    dfa = xr.open_mfdataset(dataset_path)
    dfb = dfa.swap_dims({out_dim:in_dim})
    return dfb

##############
#PLOTS IN PROGRESS#
    
#example plot of gridspec sst satellite data used from Top_Pacific_Ref Notebook, WOULD NEED TO BE EDITED
def sst_grid_old(sst_dataset, argo_dataset):
    
    fig=plt.figure(figsize=(20,12),dpi=100,facecolor='white')
    gs=fig.add_gridspec(nrows=4,ncols=2,wspace=0.29,hspace=0.0,)
    I=0
    i=0

    ds = sst_dataset
    df = argo_dataset

    #sets list for lon and lat to be used in ax2
    LAT=[0,0,0];LON=[250.0,235.0,220.0]
    
    #creates a loop for all the functions that will apply to indices in the list (np.arrange creates a list) where (lat,long,increments)
    for I in np.arange(0,30,10):
      #makes ax1 the first sublots, in the gridspec in the ith row, 0th column
      ax1=fig.add_subplot(gs[i,0])
      #makes the plots, contourf plot with x,y parameters, levels (number/positions of lines and regions), ocean color with color extending in both directions
      pc=ax1.contourf(ds.sst.lon,ds.sst.lat,ds.sst.isel(time=I).squeeze(),levels=np.arange(10,30,0.01),cmap='ocean',extend='both')
      #ax1.text places text on top of the plot, str sets string of the time values, .values only uses the values, [:10] only uses the first 10 characters in the string of values
      # then uses other arguments for the alignment and text characteristics
      #   could not use ax1.set_title() becuase the title could only be aligned on top of the plot
      ax1.text(0.02,0.98,str(ds.time[I].values)[:10],horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=12,fontweight='bold',color='black')
      #sets the range of x values on the x axis
      ax1.set_xlim(150,290)
      ax1.set_xlabel('Longitude (ºE)',fontsize=10,color='black',fontstyle='oblique')
      ax1.set_ylabel('Lattitude (ºN)',fontsize=10,color='black',fontstyle='oblique', rotation=90)
      ax1.tick_params(axis='both', which='major', labelsize=8, labelcolor='black', width=0.5, length=6)
      #sets the increasing increments of the indices
      i+=1
        
    #cax designates axis into which colorbar will be drawn; coordinates are given relative to the figure[horizontal position,vertical position,width,height(from bottom)]
    cax=fig.add_axes([0.475, 0.3, 0.015, 0.575])
    #cax=cax : specifies that the colorbar should be drawn on the same axes instance (cax) that was created earlier (kwargs? looked this up)
    plt.colorbar(pc, cax=cax, orientation='vertical', extend='both',)
    #designates colorbar label (which is on the y axis) and ticks
    cax.set_ylabel('ºC',rotation=0,labelpad=10,fontsize=13, color='black', fontstyle='oblique')
    cax.tick_params(axis='y',which='major',labelsize=13,labelcolor='black',width=0.5,length=6)

    #creates a loop using indices from multiple lists simultaneously; in python (not matplotlib or pyplot)
    #the rest is structured the same as ax1
    for i, (lat, lon) in enumerate(zip(LAT, LON)):
      ax2=fig.add_subplot(gs[i,1],)
      pc=ax2.plot(ds.time,ds.sst.sel(lon=LON[i],lat=LAT[i],method='nearest'))
      ax2.text(0.02,0.98,f'Longitude(ºE):{LON[i]}, Latitude(ºN):{LAT[i]}',horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=12,fontweight='bold',color='black')
      ax2.set_xlabel('Time',fontsize=10, color='black', fontstyle='oblique')
      ax2.tick_params(axis='both', which='major',labelsize=7.25, labelcolor='black',width=0.5, length=6)
      ax2.set_ylabel('SST (ºC)',fontsize=10, color='black', fontstyle='oblique')

#plots the contour plots of sst w a zoom on the argo float occurance, NEEDS BETTER LABELS - str NOT CALLABLE, yassir did this one
#this done not work as an actual function, this is just the code used for df73 that I added a def() to, timestamps are insignificant
#could take out the first def() line and use it to make a plot
#had to take out the plt.title lines bc the +str doesn't work when I'm by myself but it works for everybody else
#is seperated into individual plots so what is exported is a screenshot
def ssta_grid_plot_ocean(sat_dataset,argo_dataset):
    from tqdm import tqdm
    for TIME in tqdm(['2022-06-18','2022-06-28','2022-07-06','2022-07-16','2022-07-26','2022-08-4',]):
    
        east_lon = np.where(df73['LONGITUDE'] < 0, df73['LONGITUDE'] + 360, df73['LONGITUDE'])
        east_lon0 = np.where(df73.sel(time=TIME,method='nearest')['LONGITUDE'] < 0, df73.sel(time=TIME,method='nearest')['LONGITUDE'] + 360, df73.sel(time=TIME,method='nearest')['LONGITUDE'])
            
        fig=plt.figure(figsize=(10,8))
        gs=fig.add_gridspec(nrows=3,ncols=3,wspace=0.29,hspace=0.25,)
        
        ax1=fig.add_subplot(gs[1,0:2])
        pc=ax1.contourf(ds.sst.lon,ds.sst.lat,ds.sst.sel(time=TIME).squeeze(),levels=np.arange(18,30,0.01),cmap='ocean',extend='both')
        plt.scatter(east_lon0,df73.sel(time=TIME,method='nearest')['LATITUDE'],zorder=2,s=20,color='cyan')
        
        plt.xlim(200,280)
        plt.ylim(-10,10)
        plt.scatter(east_lon,df73['LATITUDE'],zorder=2,s=2,color='grey')
        #plt.scatter(east_lon,df73['LATITUDE'],df73.JULD.isel(N_PROF=4),zorder=3,color='blue')
        #plt.title('Float 73 and SST on'+str(TIME))
        
        ax1=fig.add_subplot(gs[1,2])
        pc=ax1.contourf(ds.sst.lon,ds.sst.lat,ds.sst.sel(time=TIME).squeeze(),levels=np.arange(18,30,0.01),cmap='ocean',extend='both')
        # ax1.set()
        pc=ax1.contour(ds.sst.lon,ds.sst.lat,ds.sst.sel(time=TIME).squeeze(),levels=np.arange(18,30,0.1),colors='gray',linewidths=0.5)
        pc=ax1.contour(ds.sst.lon,ds.sst.lat,ds.sst.sel(time=TIME).squeeze(),levels=np.arange(23,23.1,0.1),colors='black',linewidths=2)
        plt.xlim(210,220)
        plt.ylim(-2,6)
        plt.scatter(east_lon,df73['LATITUDE'],zorder=2,s=2,color='grey')
        plt.scatter(east_lon0,df73.sel(time=TIME,method='nearest')['LATITUDE'],zorder=2,s=20,color='cyan')
        #plt.scatter(east_lon,df73['LATITUDE'],df73.JULD.isel(N_PROF=4),zorder=3,color='blue')
        #plt.title('Float 73 and SST on'+str(TIME)) 
        plt.vlines(25,0,5,ls='--')

#plots the contour plots of sst w a zoom on the argo float occurance, NEEDS BETTER LABELS - str NOT CALLABLE, I changed this one
#does work as a funciton
def ssta_grid_plot_bwr(sat_dataset,argo_dataset):
    from tqdm import tqdm
    for TIME in tqdm(['2022-06-18','2022-06-28','2022-07-06','2022-07-16','2022-07-26','2022-08-4',]):

        df = argo_dataset
        ds = sat_dataset
        
        east_lon = np.where(df['LONGITUDE'] < 0, df['LONGITUDE'] + 360, df['LONGITUDE'])
        east_lon0 = np.where(df.sel(time=TIME,method='nearest')['LONGITUDE'] < 0, df.sel(time=TIME,method='nearest')['LONGITUDE'] + 360, df.sel(time=TIME,method='nearest')['LONGITUDE'])
        
        fig=plt.figure(figsize=(10,8))
        gs=fig.add_gridspec(nrows=3,ncols=3,wspace=0.29,hspace=0.25,)
        
        ax1=fig.add_subplot(gs[1,0:2])
        p1=ax1.contourf(ds.sst.lon,ds.sst.lat,ds.sst.sel(time=TIME).squeeze(),levels=np.arange(18,30,0.01),cmap='bwr',extend='both')
        plt.scatter(east_lon0,df.sel(time=TIME,method='nearest')['LATITUDE'],zorder=2,s=20,color='cyan')
        plt.xlim(200,280)
        plt.ylim(-10,10)
        plt.scatter(east_lon,df['LATITUDE'],zorder=2,s=2,color='yellow')
        #plt.scatter(east_lon,df['LATITUDE'],df.JULD.isel(N_PROF=4),zorder=3,color='blue')
        #plt.title('Float 73 and SST on'+str(TIME))
        
        ax2=fig.add_subplot(gs[1,2])
        #color contour of ssta
        p2=ax2.contourf(ds.sst.lon,ds.sst.lat,ds.sst.sel(time=TIME).squeeze(),levels=np.arange(22,30,0.01),cmap='bwr',extend='both')
        #line contour of ssta
        p2=ax2.contour(ds.sst.lon,ds.sst.lat,ds.sst.sel(time=TIME).squeeze(),levels=np.arange(18,30,0.1),colors='gray',linewidths=0.5)
        #contour of temps between 24-24.5
        p2=ax2.contour(ds.sst.lon,ds.sst.lat,ds.sst.sel(time=TIME).squeeze(),levels=np.arange(24,24.5,0.1),colors='black',linewidths=2)
        plt.xlim(210,220)
        plt.ylim(-2,6)
        plt.scatter(east_lon,df['LATITUDE'],zorder=2,s=2,color='yellow')
        plt.scatter(east_lon0,df.sel(time=TIME,method='nearest')['LATITUDE'],zorder=2,s=20,color='cyan')
        #plt.scatter(east_lon,df['LATITUDE'],df.JULD.isel(N_PROF=4),zorder=3,color='blue')
        #plt.title('Float 73 and SST on'+str(TIME)) 
        plt.vlines(25,0,5,ls='--') 
    return fig, gs, ax1, p1, ax2, p2

##############

#function to get summer-winter times
def get_times(dataset):
    ''' returns dataset with times from summer-winter (takes out months 3-4)
    /PARAMS: dataset: float dataset title (ex: df73)
    '''
    months = pd.to_datetime(dataset.time).month
    is_spring = (months >= 3) & (months <= 4)
    sum_wint = dataset.time[~is_spring]#sets summer-winter time values in the dataset
    return sum_wint

# function to get sla-dome times
def get_sla_dome(argo_dataset, sla_dataset, times):
    ''' returns dataset with times where there was an SLA positive anomaly (sla dome)
    /PARAMS: dataset: float dataset title (ex: df73); sla dataset title (ex: dh); times: timerange in data-array (ex:tiw_times)
    /zoom_sla compiles a datarray with the sla taken for a set range around a float at a given timestamp; gives unique lat/lon data for each timestep 
    /sla_mask_array compiles a new dataset that only includes sla values that are greater than mean+2sigma (taken as sla dome) for each given range at a timestamp
    /sla_dome_times gives the times for where there is an sla_dome; take away ".time" for the sla values
    '''
    sla_zooms = [] #starts list from each box
    for time in times:
        # sets lat and lon for each box, and sets a range for the lon
        lat = argo_dataset.LATITUDE.sel(time=time, method='nearest').compute()
        lon = (argo_dataset.LONGITUDE % 360).sel(time=time, method='nearest').compute()
        lon_range_min, lon_range_max = lon-10, lon+10

        #take box for each time
        sla_zoom = sla_dataset.sla.sel(longitude=slice(lon_range_min, lon_range_max), latitude=slice(3,8)).sel(time=time, method='nearest') #xarray object
        sla_zooms.append(sla_zoom) #make a list

    zoom_sla = xr.concat(sla_zooms, dim='time') #xarray dataset (3D)

    sla_means = zoom_sla.mean(dim=['longitude', 'latitude']) #takes means along lat and lon, leaves time active

    sla_stds = zoom_sla.std(dim=['longitude', 'latitude']) #takes stdevs along lat and lon, leaves time active

    for sla_mean, sla_std in zip(sla_means, sla_stds):
        sla_mask = np.greater_equal(zoom_sla,(sla_mean + 3*np.abs(sla_std)))
    sla_masks = xr.where(sla_mask, zoom_sla, np.nan) #replaces nan values with nan
    sla_masks = sla_masks.dropna(dim='time', thresh=0.95*len(sla_masks))#removes data points where all values are nan... still leaves data points where there is probably only one integer value...
    tiw_times_valid = sla_masks.time #makes new timeframe that now only includes values where there was an sla dome
    sla_mask_array = xr.DataArray(sla_masks, dims=('time', 'latitude', 'longitude'), coords={'time': tiw_times_valid, 'latitude': zoom_sla.latitude, 'longitude': zoom_sla.longitude}) #makes this into an array
    
    return sla_mask_array #returns time array from sla array from summer-winter where there was an sla dome (32 days)

#function to get cold cores
def gets_cold_core(argo_dataset, sst_dataset, times):
    ''' returns dataset with sst minimums taken around a float (cold core)
    /PARAMS: dataset: float dataset title (ex: df73); times: timerange in data-array (ex: sla_dome_times)
    /zoom_sst compiles a datarray with the sst taken for a set range around a float at a given timestamp; gives unique lat/lon data for each timestep 
    /sst_mask_array compiles a new dataset that only includes ssst values that are less than mean-2sigma (taken as cold core) for each given range at a timestamp
    /cold_core gives a dataarray with the cold core values, can take the times of this with ".times"
    '''
    sst_zooms = [] #starts list from each box
    for time in times:
        # sets lat and lon for each box, and sets a range for the lon
        lat = argo_dataset.LATITUDE.sel(time=time, method='nearest').compute()
        lon = (argo_dataset.LONGITUDE % 360).sel(time=time, method='nearest').compute()
        lon_range_min, lon_range_max = lon-10, lon+10
        lat_range_min, lat_range_max = -2, 5
    
        #take box for each time
        sst_zoom = sst_dataset.sst.sel(lon=slice(lon_range_min, lon_range_max), lat=slice(lat_range_min, lat_range_max)).sel(time=time, method='nearest') #xarray object
        sst_zooms.append(sst_zoom) #make a list
        
    zoom_sst = xr.concat(sst_zooms, dim='time') #xarray dataset (3D)
    sst_means = zoom_sst.mean(dim=['lon', 'lat']) #takes means along lat and lon, leaves time active
    sst_stds = zoom_sst.std(dim=['lon', 'lat']) #takes stdevs along lat and lon, leaves time active

    for sst_mean, sst_std in zip(sst_means, sst_stds):
        sst_mask = np.less_equal(zoom_sst,(sst_mean - 2*np.abs(sst_std)))
    sst_masks = xr.where(sst_mask, zoom_sst, np.nan) #replaces nan values with nan
    sst_masks = sst_masks.dropna(dim='time', how='all')#removes data points where all values are nan... still leaves data points where there is probably only one integer value...
    tiw_times_valid = sst_masks.time #makes new timeframe that now only includes values where there was an sla dome
    sst_mask_array = xr.DataArray(sst_masks, dims=('time', 'lat', 'lon'), coords={'time': tiw_times_valid, 'lat': zoom_sst.lat, 'lon': zoom_sst.lon}) #makes this into an array
        
    return sst_mask_array

#compares temp of argo with cold core
def argo_in_core(argo_dataset, sst_dataset, times):
    ''' returns dataset times where the argo float was the less than or equal to that in the cold core
    /PARAMS: argo_dataset: float dataset title (ex: df73); times: timerange in data-array (ex: sla_dome_times, or cold core times)
    / sst_zooms compiles a datarray with the sst taken for a set range around a float at a given timestamp; gives unique lat/lon data for each timestep 
    / argo_locs creates a new dataset with only surface temperatures for argos in selected time frame
    / sst_masked compiles a new dataset that only includes argo surface temperatures profiles that are less than or equal to the cold core; ".time" gives a list of those timestamps, 
        can be used as an identifier for argo floats in cold core and profiles that can then be analysed
    '''  
    #taking box for mean/stdev data
    sst_zooms = [] #starts list for each box
    for time in times:
        # sets lat and lon for each box, and sets a range for the lon
        lat = argo_dataset.LATITUDE.sel(time=time, method='nearest').compute()
        lon = (argo_dataset.LONGITUDE % 360).sel(time=time, method='nearest').compute()
        lon_range_min, lon_range_max = lon-10, lon+10
        lat_range_min, lat_range_max = -2, 5
    
        #take box for each time
        sst_zoom = sst_dataset.sst.sel(lon=slice(lon_range_min, lon_range_max), lat=slice(lat_range_min, lat_range_max)).sel(time=time, method='nearest') #xarray object
        sst_zooms.append(sst_zoom) #make a list
        
    zoom_sst = xr.concat(sst_zooms, dim='time') #xarray dataset (3D)
    sst_means = zoom_sst.mean(dim=['lon', 'lat']) #takes means along lat and lon, leaves time active
    sst_stds = zoom_sst.std(dim=['lon', 'lat']) #takes stdevs along lat and lon, leaves time active

    #setting float in box
    argo_locs = []
    for time in times:
        # make float list
        argo_loc = argo_dataset.TEMP_ADJUSTED.sel(time=time,method='nearest').isel(N_LEVELS=0)
        argo_locs.append(argo_loc)
    
    argo_loc_pts = xr.concat(argo_locs, dim='time')
    argo_loc_pts

    #compare float temp to cold_core temp
    for sst_mean, sst_std in zip(sst_means, sst_stds):
        sst_mask = np.less_equal(argo_loc_pts,(sst_mean - 2*np.abs(sst_std)))
    sst_masked = xr.where(sst_mask, argo_loc_pts, np.nan) #replaces nan values with nan
    sst_masked = sst_masked.dropna(dim='time', how='all')#removes data points where all values are nan... still leaves data points where there is probably only one integer value...
        
    return sst_masked


#gets LEFs
def get_LEF_mask(sst_dataset, times):
    ''' returns dataset times where the LEF was at it's maximum (LEF)
    /PARAMS: dataset: sst dataset (ex: ds); times: timerange in data-array (ex: sla_dome_times, or cold core times)
    / grads compiles a datarray with the sst gradient from entire dataset
    / min_bounds creates a new dataset with only minimum gradient (LEF)
    / gradient_mask_arrays compiles a new dataset that only contains the min gradient values for each timestep
    '''
    grad_list = []
    for time in times:
        dsst_dx = (sst_dataset.sst - sst_dataset.sst.roll(lon=1, roll_coords=False)) / (sst_dataset.lon - sst_dataset.lon.roll(lon=1, roll_coords=False))
        dsst_dy = (sst_dataset.sst - sst_dataset.sst.roll(lat=1, roll_coords=False)) / (sst_dataset.lat - sst_dataset.lat.roll(lat=1, roll_coords=False))
        dsst_h = dsst_dx**2 + dsst_dy**2
    grad_list.append(dsst_h)
    grads = xr.concat(grad_list, dim='time')
    
    min_bound_list = []
    for time in times:
        valid_dsst_h = grads.sel(time=time, method='nearest').values[~np.isnan(grads.sel(time=time, method='nearest').values)]
        upper_quantile = np.quantile(valid_dsst_h, 0.975)
        far_reaching_outliers = valid_dsst_h[valid_dsst_h > upper_quantile]
        min_bound = np.min(far_reaching_outliers)
        min_bound_list.append(min_bound)
    min_bounds = xr.DataArray(min_bound_list, coords=[times], dims=['time'])
    
    grad_mask_list = []
    for grad, min_bound in zip(grads, min_bounds.values):
        sst_grad_mask = np.greater_equal(grad, min_bound)
        grad_mask = xr.where(sst_grad_mask, grad, np.nan)
        grad_mask_list.append(grad_mask)
    
    gradient_mask_array = xr.concat(grad_mask_list, dim='time').transpose('time', 'lat', 'lon')
    gradient_mask_arrays = gradient_mask_array.dropna(dim='time', thresh=0.95*len(gradient_mask_array))
    return gradient_mask_arrays

#climatology mean
def clim_anomaly(VARIABLE):
    ''' returns dataset times where the LEF was at it's maximum (LEF)
    /PARAMS: VARIABLE: float dataset + specified variable (ex: df73.NITRATE_ADJUSTED)
    '''
    VARIABLEp=VARIABLE.groupby('time.month')-VARIABLE.groupby('time.month').mean('time')
    return VARIABLEp



















