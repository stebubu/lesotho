import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import tempfile

def calculate_djf_sum(data_array):
    # Insert the DJF calculation function here
    pass
    
def load_netcdf(file):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
            # Write the uploaded file's contents to the temporary file
            tmp.write(file.read())
            tmp.seek(0)  # Move to the beginning of the file for reading
            # Load the dataset from the temporary file
            dataset = xr.open_dataset(tmp.name)
        os.unlink(tmp.name)  # Delete the temporary file
        return dataset
    return None

def main():
    st.title("Climate Data Analysis")

    # File uploaders
    historical_file = st.file_uploader("Upload historical netCDF", type=["nc"])
    forecast_file = st.file_uploader("Upload forecast netCDF for 2023", type=["nc"])


    if historical_file and forecast_file:
        historical_data = load_netcdf(historical_file)
        forecast_data = load_netcdf(forecast_file)
        #historical_data = xr.open_dataset(historical_file, engine='netcdf4')

        # Assuming 'precipitation' variable exists; adjust as necessary
        historical_precip_da = historical_data['precipitation']
        forecast_precip_da = forecast_data['precipitation']

        # Calculate DJF sums for historical data
        historical_djf_sum = calculate_djf_sum(historical_precip_da)

        # Calculate event threshold
        event_threshold = historical_djf_sum.quantile(0.2, dim="djf_year")

        # Step 4: Plot probability below normal for forecast
        forecast_djf_sum = calculate_djf_sum(forecast_precip_da)
        below_normal_forecast = forecast_djf_sum < event_threshold
        prob_below_normal = below_normal_forecast.mean(dim="djf_year")
        
        # Custom colormap for probability plot
        colors = ['red', 'green']  # Red for below 0.33, green for above
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([0, 0.33, 1], cmap.N)

        fig, ax = plt.subplots()
        prob_below_normal.plot(ax=ax, cmap=cmap, norm=norm)
        st.pyplot(fig)

        # Step 5: Interactive year selection and plotting
        year = st.slider("Select a Year", int(historical_djf_sum.djf_year.min()), int(historical_djf_sum.djf_year.max()))
        selected_year_data = historical_djf_sum.sel(djf_year=year)
        fig, ax = plt.subplots()
        selected_year_data.plot(ax=ax)
        st.pyplot(fig)

        # Add more functionality as needed

if __name__ == "__main__":
    main()
