import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go

def calculate_djf_sum(data_array):
    """
    Calculate the DJF (December, January, February) sum for a given DataArray.
    Adjusts years for December to align with meteorological DJF definition.
    """
    data_array['time'] = pd.to_datetime(data_array['time'].values)
    data_array = data_array.assign_coords(djf_year=(data_array['time'].dt.month==12) + data_array['time'].dt.year)
    djf_data = data_array.where(data_array['time'].dt.month.isin([12, 1, 2]), drop=True)
    return djf_data.groupby('djf_year').sum(dim='time')

def load_netcdf(file, engine='netcdf4'):
    """
    Load a NetCDF file into an xarray DataSet with a specified engine.
    """
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
            tmp.write(file.getvalue())
            tmp.flush()
            dataset = xr.open_dataset(tmp.name, engine=engine)
        os.remove(tmp.name)
        return dataset
    return None

def calculate_probabilities(forecast_djf_data, lower_tercile):
    """
    Calculate probabilities of below-normal precipitation.
    """
    expanded_lt = lower_tercile.broadcast_like(forecast_djf_data.isel(ensemble=0))
    below_normal = forecast_djf_data < expanded_lt
    return below_normal.mean(dim='ensemble')

def prepare_custom_colormap():
    """
    Prepare a custom colormap for below-normal probability visualization.
    """
    colors = ["red", "green"]
    return px.colors.make_colorscale(colors, positions=[0, 0.33, 1])

def main():
    st.title("Climate Data Analysis")

    historical_file = st.file_uploader("Upload historical netCDF", type=["nc"], key="historical")
    if historical_file:
        historical_data = load_netcdf(historical_file)
        st.success("Historical data uploaded successfully.")

    forecast_file = st.file_uploader("Upload forecast netCDF for 2023", type=["nc"], key="forecast")
    if forecast_file:
        forecast_data = load_netcdf(forecast_file)
        st.success("Forecast data uploaded successfully.")

        # Example process on the uploaded data
        if historical_data and forecast_data:
            # Process your data here
            st.write("Data processed.")
            
            # Visualization example
            color_scale = prepare_custom_colormap()
            fig = px.imshow(forecast_data, color_continuous_scale=color_scale, labels={"color": "Probability"})
            st.plotly_chart(fig, use_container_width=True)

            # Additional visualization and analysis as required

if __name__ == "__main__":
    main()
