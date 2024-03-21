import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pystac_client
import planetary_computer
import geopandas as gpd
from streamlit_folium import st_folium


# Function to generate synthetic data
def generate_synthetic_data():
    # Seed for reproducibility
    np.random.seed(0)

    # Grid and timeframe setup
    n_rows, n_cols = 25, 25
    start_date, end_date = "1990-01-01", "2022-12-31"
    forecast_start, forecast_end = "2023-01-01", "2023-12-31"

    # Generate synthetic historical and forecast precipitation data
    historical_dates = pd.date_range(start_date, end_date)
    forecast_dates = pd.date_range(forecast_start, forecast_end)

    historical_precip = np.random.uniform(0, 100, (len(historical_dates), n_rows, n_cols))
    n_ensembles = 50  # Number of ensemble members
    forecast_precip = np.random.uniform(0, 98, (n_ensembles, len(forecast_dates), n_rows, n_cols))

    # Convert to xarray DataArrays for easier handling
    historical_precip_da = xr.DataArray(historical_precip, coords=[historical_dates, np.linspace(-25, 25, n_rows), np.linspace(-25, 25, n_cols)], dims=["time", "lat", "lon"], name='precipitation')
    forecast_precip_da = xr.DataArray(forecast_precip, coords=[np.arange(n_ensembles), forecast_dates, np.linspace(-25, 25, n_rows), np.linspace(-25, 25, n_cols)], dims=["ensemble", "time", "lat", "lon"], name='precipitation')
    
    return historical_precip_da, forecast_precip_da



def calculate_djf_sum(data_array):
    # Create a new 'year' coordinate for grouping, shifting December to the next year
    year_shifted = data_array.time.dt.year + (data_array.time.dt.month == 12)

    # Assign this 'year' coordinate to the DataArray
    data_array = data_array.assign_coords(djf_year=year_shifted)

    # Select DJF months (Dec of the previous year from the original, Jan and Feb of the current 'djf_year')
    djf_data = data_array.where(data_array.time.dt.month.isin([1, 2, 12]), drop=True)

    # Now, group by this 'djf_year' and sum to get DJF precipitation sum
    djf_sum = djf_data.groupby('djf_year').sum(dim="time")

    return djf_sum

def filter_djf_months(data_array):
    """
    Filters the input data_array for DJF months and adjusts the year for December entries.

    Parameters:
    - data_array: xarray.DataArray with a 'time' dimension in datetime64 format.

    Returns:
    - xarray.DataArray filtered for DJF months with adjusted years for December.
    """
    # Ensure time is a pandas DatetimeIndex for vectorized operations
    times = pd.to_datetime(data_array['time'].values)

    # Create adjusted times for year shift in December
    adjusted_times = times.where(times.month != 12, times + pd.offsets.DateOffset(years=1))

    # Extract the year, now directly using pandas DatetimeIndex.year for vectorized operation
    adjusted_years = adjusted_times.year

    # Assign adjusted year back to data_array as a new coordinate
    data_array = data_array.assign_coords(djf_year=('time', adjusted_years))

    # Filter for DJF months, using original times for condition to maintain original dataset's size
    djf_data = data_array.where(data_array['time.month'].isin([12, 1, 2]), drop=True)

    return djf_data


# Step 2: Calculate the DJF median sum for each pixel across ensembles
def calculate_djf_median_sums(data_array):
    # Group by DJF year after adjusting December's year
    djf_grouped = data_array.groupby('time.year').sum(dim=['time'])
    return djf_grouped.sum(dim='year')
    
# Modify the load_netcdf function to specify the engine explicitly
def load_netcdf(file, engine='netcdf4'):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        # Specify the engine explicitly when opening the dataset
        dataset = xr.open_dataset(tmp_path, engine=engine)
        os.remove(tmp_path)
        return dataset
    return None

def calculate_below_normal_probability_ensemble(forecast_djf_data, lower_tercile, n_ensembles):
    """
    Calculate the probability of below-normal precipitation for each pixel across the forecast dataset,
    considering all ensemble members.

    Parameters:
    - forecast_djf_data: xarray.DataArray with dimensions ('ensemble', 'djf_year', 'lat', 'lon')
    - lower_tercile: xarray.DataArray with dimensions ('lat', 'lon') representing the lower tercile threshold

    Returns:
    - xarray.DataArray representing the probability of below-normal precipitation for each pixel
    """
    # Ensure lower_tercile is correctly calculated and has the right dimensions
    if not isinstance(lower_tercile, xr.DataArray) or 'lat' not in lower_tercile.dims or 'lon' not in lower_tercile.dims:
        raise ValueError("lower_tercile must be an xarray.DataArray with 'lat' and 'lon' dimensions.")

    # Assuming forecast_djf_sums is already summed over 'time' and needs to be compared across 'ensemble'

    # Create an empty array with the target shape
    expanded_shape = (n_ensembles,) + lower_tercile.shape
    # Use np.tile to replicate lower_tercile values across the new 'ensemble' dimension
    replicated_values = np.tile(lower_tercile.values[None, :, :], (n_ensembles, 1, 1))

    # Construct the new DataArray with the replicated values
    lower_tercile_expanded = xr.DataArray(
        replicated_values,
        dims=['ensemble', 'lat', 'lon'],
        coords={
            'ensemble': np.arange(n_ensembles),
            'lat': lower_tercile.lat,
            'lon': lower_tercile.lon
        }
    )

    # Align 'lat' and 'lon' coordinates with forecast_djf_data to ensure exact match
    # This step is crucial for operations involving both DataArrays
    lower_tercile_expanded = lower_tercile_expanded.assign_coords(
        lat=forecast_djf_data.lat,
        lon=forecast_djf_data.lon
    )
    #print (lower_tercile_expanded)
    #print (forecast_djf_data)

    # Verify dimensions
    #print("Expanded Lower Tercile Dimensions:", lower_tercile_expanded.dims)
    #print("Forecast Data Dimensions:", forecast_djf_data.dims)

    # Compare each ensemble's DJF sum with the lower_tercile threshold
    below_normal = forecast_djf_data < lower_tercile_expanded
    #print (below_normal)

    # Count instances where the DJF sum is below the lower_tercile threshold for each pixel, across all ensembles
    below_normal_count = below_normal.sum(dim='ensemble')

    # Calculate the probability by dividing by the number of ensemble members
    probability = below_normal_count / n_ensembles

    return probability


def round_coordinates(coord, interval=0.25):
    """Rounds the coordinates to the nearest grid point."""
    return [round(c / interval) * interval for c in coord]

def fetch_rain_bbox(varname, factor, location, start_date, end_date):
    """
    Fetches ERA5 precipitation data for a specified bounding box, date range, and variable name,
    accumulating the data monthly.
    """
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1/")
    monthly_dataarrays = []

    current_month_start = pd.to_datetime(start_date)
    while current_month_start <= pd.to_datetime(end_date):
        next_month_start = (current_month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        current_month_end = next_month_start - timedelta(days=1)

        search_results = catalog.search(
            collections=["era5-pds"], datetime=[current_month_start.isoformat(), current_month_end.isoformat()], query={"era5:kind": {"eq": "fc"}}
        )

        items = list(search_results.items())
        for item in items:
            signed_item = planetary_computer.sign(item)
            asset = signed_item.assets.get(varname)
            if asset:
                dataset = xr.open_dataset(asset.href, **asset.extra_fields["xarray:open_kwargs"])
                wind_ds = dataset[varname]
                interval = 0.25
                rounded_coord = round_coordinates(location, interval)
                wind_ds_sliced = wind_ds.sel(lat=slice(rounded_coord[1], rounded_coord[0]), lon=slice(rounded_coord[2], rounded_coord[3])) * factor
                monthly_dataarrays.append(wind_ds_sliced)
        
        current_month_start = next_month_start

    # Concatenate all monthly DataArrays along the time dimension if any data was fetched
    if monthly_dataarrays:
        combined = xr.concat(monthly_dataarrays, dim="time")
        return combined
    else:
        return None



def main():
    # Streamlit interface for inputs
    st.title("ERA5 Precipitation Data Download")
    start_date = st.date_input("Start date", value=datetime(1995, 1, 1))
    end_date = st.date_input("End date", value=datetime(1995, 12, 31))
    
    lat_min = st.number_input("Latitude Min:", value=-31.0)
    lat_max = st.number_input("Latitude Max:", value=-29.0)
    lon_min = st.number_input("Longitude Min:", value=26.5)
    lon_max = st.number_input("Longitude Max:", value=29.5)
    location = [lat_min, lat_max, lon_min, lon_max]
    
    varname_Rain = "precipitation_amount_1hour_Accumulation"
    factor = 1  # Adjust the factor as necessary
    
    # Button to fetch and process the data
    if st.button("Fetch ERA5 Precipitation Data"):
        precipitation_data = fetch_rain_bbox(varname_Rain, factor, location, start_date, end_date)
        precipitation_data = precipitation_data.rename('precipitation')
        daily_precipitation = precipitation_data.resample(time='D').sum()
        historical_djf_sum = calculate_djf_sum(daily_precipitation)
        historical_djf_sum = historical_djf_sum.chunk({'djf_year': -1})
        lower_tercile = historical_djf_sum.quantile(0.33, dim="djf_year")
        upper_tercile = historical_djf_sum.quantile(0.67, dim="djf_year")
         # Plotting lower tercilet with custom color scale
        fig = px.imshow(lower_tercile, 
                        labels=dict(x="Longitude", y="Latitude", color="lower_tercile"),
                        x=lower_tercile.lon,
                        y=lower_tercile.lat)
        
        fig.update_traces(hoverinfo='x+y+z', showscale=True)
        st.plotly_chart(fig, use_container_width=True)  

        # Convert to GeoDataFrame (if necessary)
        # Plot with Mapbox overlay
        # Mapbox access token
        
        
        '''# Add raster layer using xarray and Folium integration (might need additional libraries)
        # Create a basic Folium map
        map_center = [lower_tercile.lat.mean(), lower_tercile.lon.mean()]
        my_map = folium.Map(location=map_center, zoom_start=5, tiles="OpenStreetMap")  
 
        folium.raster_layers.ImageOverlay(
            image=lower_tercile.values,
            bounds=[[lower_tercile.lat.min(), lower_tercile.lat.min()],
                    [lower_tercile.lon.max(), lower_tercile.lon.max()]]
        ).add_to(my_map)
        
        # Display in Streamlit
        st_folium(my_map, width=700)'''
        
       
        
        



    
    st.title("I-CISK LL Lesotho Drought Climate Data Analysis")
    n_ensembles = 50


    # Step 0: Upload and process historical data -Optional
#    historical_file = st.file_uploader("Upload historical netCDF", type=["nc"], key="historical")
#    if historical_file is not None:
#         st.success("Historical data uploaded successfully.")
#         historical_data = load_netcdf(historical_file)
#         # Perform operations with historical_data...
#         st.write("Historical data processed.")

        # After historical data is uploaded and processed, allow forecast data upload
        # Step 2: Upload and process forecast data
 #        forecast_file = st.file_uploader("Upload forecast netCDF for 2023", type=["nc"], key="forecast")
 #        if forecast_file is not None:
 #            st.success("Forecast data uploaded successfully.")
 #           forecast_data = load_netcdf(forecast_file)
 #            # Perform operations with forecast_data...
 #            st.write("Forecast data processed.")
 #            # Here you can add additional steps, such as visualizations or calculations,
 #            # that depend on both historical and forecast data.

            # Assuming 'precipitation' variable exists; adjust as necessary
 #            historical_precip_da = historical_data['precipitation']
 #            forecast_precip_da = forecast_data['precipitation']
# Step 1: enerate Synthetic Data
    if st.button('Generate Synthetic Data'):
        historical_data, forecast_data = generate_synthetic_data()
        st.session_state['historical_data'] = historical_data
        st.session_state['forecast_data'] = forecast_data
        st.success("Synthetic data generated successfully.")

    # Check if data is already generated and stored in session state
    if 'historical_data' in st.session_state and 'forecast_data' in st.session_state:

        historical_precip_da, forecast_precip_da = generate_synthetic_data()


        historical_djf_sum = calculate_djf_sum(historical_precip_da)


        event_threshold = historical_djf_sum.quantile(0.2)
        # Step 2: Determine Terciles for Historical Data        
        lower_tercile = historical_djf_sum.quantile(0.33, dim="djf_year")
        upper_tercile = historical_djf_sum.quantile(0.67, dim="djf_year")

        forecast_djf_filtered = filter_djf_months(forecast_precip_da)

        forecast_djf_median_sums = calculate_djf_median_sums(forecast_djf_filtered)

        # Step 4: Plot probability below normal for forecast
        # Calculate the below-normal probability using the lower_tercile from historical data
        below_normal_probability_forecast = calculate_below_normal_probability_ensemble(forecast_djf_median_sums, lower_tercile,n_ensembles)

    
        # Custom colormap for probability plot
        colors = ['red', 'green']  # Red for below 0.33, green for above
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([0, 0.33, 1], cmap.N)

        fig, ax = plt.subplots()
        below_normal_probability_forecast.plot(ax=ax, cmap=cmap, norm=norm)
        st.pyplot(fig)
       
        
        # Assuming below_normal_probability_forecast is your dataset
        # Create a custom color scale
        custom_color_scale = [
            [0, "red"],  # Values normalized to 0 will be red
            [0.33, "red"],  # Up to 0.33, also red
            [0.34, "green"],  # Values normalized exactly at 0.33 will be an abrupt change to green
            [1, "green"]  # Up to the maximum normalized value (1), will be green
        ]
        
        # Plotting below_normal_probability_forecast with custom color scale
        below_normal_probability_forecast = below_normal_probability_forecast.sortby('lat', ascending=True)

        fig = px.imshow(below_normal_probability_forecast, 
                        labels=dict(x="Longitude", y="Latitude", color="Probability"),
                        x=below_normal_probability_forecast.lon,
                        y=below_normal_probability_forecast.lat,
                        color_continuous_scale=custom_color_scale)
        # Reverse the y-axis (latitude) to display negative values on the bottom
        #fig.update_yaxes(autorange="reversed")
 
        fig.update_traces(hoverinfo='x+y+z', showscale=True)
        st.plotly_chart(fig, use_container_width=True)            
        
        # Plotting lower tercilet with custom color scale
        fig = px.imshow(lower_tercile, 
                        labels=dict(x="Longitude", y="Latitude", color="lower_tercile"),
                        x=lower_tercile.lon,
                        y=lower_tercile.lat)
        
        fig.update_traces(hoverinfo='x+y+z', showscale=True)
        st.plotly_chart(fig, use_container_width=True)            

    
                    

        # Step 5: Interactive year selection and plotting
        year = st.slider("Select a Year", int(historical_djf_sum.djf_year.min()), int(historical_djf_sum.djf_year.max()))
        selected_year_data = historical_djf_sum.sel(djf_year=year)
        fig, ax = plt.subplots()
        selected_year_data.plot(ax=ax)
        st.pyplot(fig)


        # Step 6: Interactive year selection and plotting
        ens = st.slider("Select Ensemble", int(0), int(49))
        selected_ensemble = forecast_djf_median_sums.sel(ensemble=ens)
        fig, ax = plt.subplots()
        selected_ensemble.plot(ax=ax)
        st.pyplot(fig)
        # Add more functionality as needed
        # Interactive Ensemble Selection
        #ens1 = st.slider("Select Ensemble", 0, 49)
        #selected_ensemble = forecast_djf_median_sums.sel(ensemble=ens)
    
        # Plotting selected ensemble
        fig = px.imshow(selected_ensemble, 
                        labels=dict(x="Longitude", y="Latitude", color="Value"),
                        x=selected_ensemble.lon,
                        y=selected_ensemble.lat)
        fig.update_traces(hoverinfo='x+y+z', showscale=True)
        st.plotly_chart(fig, use_container_width=True)
    
        # Interactive Click to display value and plot boxplot
        # Note: Due to limitations in streamlit's direct integration with interactive clicks on plotly maps,
        # the actual interaction to display values and plot a boxplot on click would need a different approach.
        # Consider providing instructions to the user to select a specific longitude and latitude from dropdowns or sliders
        # and then use those to plot the boxplot.
    
        lon_1 = st.select_slider('Select Longitude', options=selected_ensemble.lon.values)
        lat_1 = st.select_slider('Select Latitude', options=selected_ensemble.lat.values)



        
        # Extracting values for the selected pixel across all ensembles
        pixel_values = forecast_djf_median_sums.sel(lon=lon_1, lat=lat_1, method="nearest")
      # Plotting a boxplot of the selected pixel across all ensembles
        fig = px.box(pixel_values.to_dataframe().reset_index(), y="precipitation")


        
        lower_tercile_value =lower_tercile.sel(lon=lon_1, lat=lat_1, method="nearest").item()
        
        st.write(int(lower_tercile_value))

        # Add a horizontal line for the lower_tercile_value
        # Add a horizontal line for the lower_tercile_value with adjusted properties
        fig.add_trace(go.Scatter(x=[0, 1], y=[lower_tercile_value, lower_tercile_value], mode="lines",
                                 name="Lower Tercile", line=dict(color="FireBrick", width=4, dash='dash')))
             
        # Annotate the lower_tercile_value on the plot
        fig.add_annotation(x=1.0, xref="paper", y=lower_tercile_value, text=f"Lower Tercile: {lower_tercile_value}",
                              showarrow=True, arrowhead=1, ax=0, ay=-40)




        # Plotting a boxplot of the selected pixel across all ensembles
        #fig = px.box(pixel_values.to_dataframe().reset_index(), y="precipitation")



    

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Please generate synthetic data.")

if __name__ == "__main__":
    main()
