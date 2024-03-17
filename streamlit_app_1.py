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
    # Define a custom color scale with specific transition points
    custom_color_scale = [
        (0, "red"),    # start with red at 0
        (0.33, "red"), # transition to red by 0.33
        (0.33, "green"),  # immediately switch to green at 0.33
        (1, "green")   # end with green at 1
    ]
    return custom_color_scale

def main():
    st.title("Climate Data Analysis")

    # Step 1: Data Upload
    historical_file = st.file_uploader("Upload historical netCDF", type=["nc"], key="historical")
    forecast_file = st.file_uploader("Upload forecast netCDF", type=["nc"], key="forecast")

    if historical_file and forecast_file:
        st.success("Data uploaded successfully.")
        historical_data = load_netcdf(historical_file)
        forecast_data = load_netcdf(forecast_file)

        # Assuming 'precipitation' is a variable in both datasets
        historical_precip = historical_data['precipitation']
        forecast_precip = forecast_data['precipitation']

        # Step 2: Process Historical Data
        djf_historical_sum = calculate_djf_sum(historical_precip)
        lower_tercile, upper_tercile = djf_historical_sum.quantile([0.33, 0.67], dim="djf_year")

        # Visualize Historical DJF Sum
        st.header("Historical DJF Sum")
        fig = px.imshow(djf_historical_sum.mean(dim='djf_year'), labels={'color': 'DJF Sum'}, aspect='auto')
        st.plotly_chart(fig)

        # Step 3: Process Forecast Data
        forecast_djf_sum = calculate_djf_sum(forecast_precip)

        # Step 4: Calculate Below-Normal Probability
        below_normal_probability = calculate_probabilities(forecast_djf_sum, lower_tercile)

        # Visualize Below-Normal Probability
        st.header("Below-Normal Probability")
        color_scale = prepare_custom_colormap()
        fig = px.imshow(below_normal_probability, color_continuous_scale=color_scale, labels={'color': 'Probability'}, aspect='auto')
        st.plotly_chart(fig)

        # Step 5: Interactive Visualization of Forecast Data
        ensemble_selection = st.selectbox("Select Ensemble", forecast_djf_sum.ensemble.values)
        selected_ensemble_data = forecast_djf_sum.sel(ensemble=ensemble_selection)

        st.header(f"Forecast Data for Ensemble {ensemble_selection}")
        fig = px.imshow(selected_ensemble_data, labels={'color': 'Precipitation'}, aspect='auto')
        st.plotly_chart(fig)

        # Add more plots as needed...
if __name__ == "__main__":
    main()
