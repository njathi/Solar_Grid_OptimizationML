import streamlit as st
import pandas as pd
import numpy as np

# Try to import matplotlib, with fallback
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    st.warning("Matplotlib not available. Plots will use Streamlit's built-in charting.")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import io
import json

st.set_page_config(page_title="Solar Grid Optimization Dashboard", layout="wide")
st.title("Solar Grid Optimization Dashboard")
st.markdown("""
This dashboard allows you to upload TMY weather data and real load profiles, forecast solar generation, and optimize load distribution with battery storage.\
You can adjust system parameters and visualize the results interactively.
""")

# --- Sidebar for user inputs ---
st.sidebar.header("System Parameters")
battery_capacity = st.sidebar.slider("Battery Capacity (kWh)", 100, 1000, 400, 50)
battery_charge_limit = st.sidebar.slider("Battery Charge Limit (kWh/h)", 10, 100, 30, 5)
battery_discharge_limit = st.sidebar.slider("Battery Discharge Limit (kWh/h)", 10, 100, 30, 5)
battery_efficiency = st.sidebar.slider("Battery Efficiency", 0.7, 1.0, 0.9, 0.01)
grid_price = st.sidebar.number_input("Grid Price (USD/kWh)", value=0.10, step=0.01)
price_peak = st.sidebar.number_input("Peak Price (USD/kWh)", value=0.30, step=0.01)
peak_start = st.sidebar.number_input("Peak Start Hour", value=18, min_value=0, max_value=23)
peak_end = st.sidebar.number_input("Peak End Hour", value=22, min_value=0, max_value=23)

# --- File uploaders ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Weather Data")
    tmy_file = st.file_uploader("Upload TMY.csv (skiprows=17)", type=["csv"])
with col2:
    st.subheader("Load Profile")
    load_input_method = st.selectbox(
        "Choose load data input method:",
        ["File Upload", "Manual Text Input", "Use Synthetic Load"]
    )

# Load profile processing function
def process_load_data(load_input_method, load_file=None, manual_text=None):
    """Process load data from various input methods"""
    load_profile = None
    
    if load_input_method == "File Upload" and load_file:
        try:
            # Determine file type and read accordingly
            file_extension = load_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                load_df = pd.read_csv(load_file)
            elif file_extension == 'txt':
                # Enhanced text file processing with multiple format support
                try:
                    # First try: standard CSV format
                    load_df = pd.read_csv(load_file, delimiter=',')
                except:
                    try:
                        # Second try: tab-separated
                        load_df = pd.read_csv(load_file, delimiter='\t')
                    except:
                        try:
                            # Third try: semicolon-separated
                            load_df = pd.read_csv(load_file, delimiter=';')
                        except:
                            # Fourth try: space-separated with custom parsing
                            content = load_file.read().decode('utf-8')
                            lines = content.strip().split('\n')
                            data = []
                            
                            for line in lines:
                                line = line.strip()
                                # Skip empty lines and comments
                                if not line or line.startswith('#'):
                                    continue
                                
                                # Try to parse space-separated values
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        # Try to parse datetime and load value
                                        # Assume last part is load value, rest is datetime
                                        datetime_str = ' '.join(parts[:-1])
                                        load_value = float(parts[-1])
                                        data.append([datetime_str, load_value])
                                    except ValueError:
                                        continue
                            
                            if data:
                                load_df = pd.DataFrame(data, columns=['datetime', 'load'])
                            else:
                                st.error("Could not parse text file. Please check the format.")
                                return None
                            
                            # Reset file pointer for potential future reads
                            load_file.seek(0)
            elif file_extension == 'xlsx':
                load_df = pd.read_excel(load_file)
            elif file_extension == 'json':
                load_df = pd.read_json(load_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
            
            st.write("Load Profile Data Sample:")
            st.dataframe(load_df.head())
            
            # Try to identify datetime and load columns
            datetime_col = None
            load_col = None
            
            for col in load_df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['time', 'date', 'datetime', 'timestamp']):
                    datetime_col = col
                if any(keyword in col_lower for keyword in ['load', 'power', 'demand', 'consumption', 'energy']):
                    load_col = col
            
            if datetime_col and load_col:
                load_df[datetime_col] = pd.to_datetime(load_df[datetime_col], errors='coerce')
                load_df = load_df.dropna(subset=[datetime_col, load_col])
                load_df.set_index(datetime_col, inplace=True)
                load_profile = load_df[load_col].values
                st.success(f"Successfully loaded load profile with {len(load_profile)} data points from {file_extension.upper()} file")
            else:
                st.error("Could not identify datetime and load columns. Please ensure your file has columns with 'time'/'date' and 'load'/'power'/'demand' in their names.")
                st.write("Available columns:", list(load_df.columns))
                
        except Exception as e:
            st.error(f"Error loading load profile: {e}")
            return None
    
    elif load_input_method == "Manual Text Input" and manual_text:
        try:
            # Parse manual text input
            lines = manual_text.strip().split('\n')
            data = []
            
            for line in lines:
                if line.strip():
                    # Try different formats
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            # Try to parse datetime and load value
                            datetime_str = ' '.join(parts[:-1])  # All but last part as datetime
                            load_value = float(parts[-1])  # Last part as load value
                            data.append([datetime_str, load_value])
                        except ValueError:
                            continue
            
            if data:
                load_df = pd.DataFrame(data, columns=['datetime', 'load'])
                load_df['datetime'] = pd.to_datetime(load_df['datetime'], errors='coerce')
                load_df = load_df.dropna()
                load_df.set_index('datetime', inplace=True)
                load_profile = load_df['load'].values
                st.success(f"Successfully parsed {len(load_profile)} data points from manual input")
            else:
                st.error("Could not parse any valid data from manual input")
                return None
                
        except Exception as e:
            st.error(f"Error parsing manual input: {e}")
            return None
    
    elif load_input_method == "Use Synthetic Load":
        st.info("Using synthetic load profile")
        return None
    
    return load_profile

# Load file uploader (only show if file upload is selected)
if load_input_method == "File Upload":
    load_file = st.file_uploader(
        "Upload Load Profile", 
        type=["csv", "txt", "xlsx", "json"],
        help="Supported formats: CSV, TXT, XLSX, JSON. File should have datetime and load columns."
    )
else:
    load_file = None

# Manual text input (only show if manual input is selected)
if load_input_method == "Manual Text Input":
    st.subheader("Manual Load Data Input")
    st.markdown("""
    **Format Instructions:**
    - One data point per line
    - Format: `YYYY-MM-DD HH:MM:SS load_value`
    - Example:
    ```
    2025-01-01 00:00:00 45.2
    2025-01-01 01:00:00 42.1
    2025-01-01 02:00:00 38.5
    ```
    """)
    manual_text = st.text_area(
        "Enter load data:",
        height=200,
        placeholder="Enter your load data here...\nFormat: YYYY-MM-DD HH:MM:SS load_value"
    )
else:
    manual_text = None

if tmy_file:
    df = pd.read_csv(tmy_file, skiprows=17)
    st.subheader("Raw Weather Data Sample")
    st.dataframe(df.head())
    # Data cleaning
    if 'time(UTC)' in df.columns:
        df['time(UTC)'] = pd.to_datetime(df['time(UTC)'], format='%Y%m%d:%H%M', errors='coerce')
    df['T2m'] = pd.to_numeric(df['T2m'], errors='coerce')
    clean_df = df.dropna()
    clean_df.set_index('time(UTC)', inplace=True)
    st.subheader("Cleaned Weather Data Sample")
    st.dataframe(clean_df.head())

    # --- Load Profile Processing ---
    st.subheader("Load Profile")
    load_profile = process_load_data(load_input_method, load_file, manual_text)

    # --- Forecasting ---
    st.subheader("Solar Generation Forecasting (Random Forest)")
    feature_cols = ['T2m', 'RH', 'Gb(n)', 'Gd(h)', 'IR(h)', 'WS10m', 'WD10m', 'SP']
    if all(col in clean_df.columns for col in feature_cols):
        X = clean_df[feature_cols]
        y = clean_df['G(h)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"Random Forest RMSE on test set: {rmse:.2f}")
        
        if matplotlib_available:
            fig1, ax1 = plt.subplots(figsize=(10,3))
            ax1.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
            ax1.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
            ax1.set_title('Actual vs Predicted G(h) (Test Set)')
            ax1.set_xlabel('Time (UTC)')
            ax1.set_ylabel('G(h)')
            ax1.legend()
            st.pyplot(fig1)
        else:
            # Use Streamlit's built-in charting
            forecast_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred
            }, index=y_test.index)
            st.line_chart(forecast_df)
        
        # Feature importances
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        st.write("Feature Importances:")
        st.dataframe(importance_df)
        
        if matplotlib_available:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.barh(importance_df['Feature'], importance_df['Importance'])
            ax2.set_xlabel('Importance')
            ax2.set_title('Random Forest Feature Importances')
            ax2.invert_yaxis()
            st.pyplot(fig2)
        else:
            st.bar_chart(importance_df.set_index('Feature'))

        # --- Optimization ---
        st.subheader("Load Distribution Optimization (with Battery)")
        hours = len(y_test)
        solar_available = y_pred
        
        # Load profile handling
        if load_profile is not None:
            if len(load_profile) >= hours:
                load_profile = load_profile[:hours]
                st.success(f"Using real load profile with {len(load_profile)} data points")
            else:
                st.warning(f"Load profile has only {len(load_profile)} points, but need {hours}. Using synthetic load.")
                load_profile = None
        
        if load_profile is None:
            # Synthetic load profile
            t = np.arange(hours)
            load_profile = 50 + 30 * np.sin(2 * np.pi * (t % 24) / 24) + np.random.normal(0, 5, size=hours)
            load_profile = np.clip(load_profile, 0, None)
            st.info("Using synthetic load profile")
        
        # Display load and solar
        load_solar_df = pd.DataFrame({
            'Load': load_profile, 
            'Solar Available': solar_available
        }, index=y_test.index)
        st.line_chart(load_solar_df)
        
        # Battery simulation
        battery_soc = 0.0
        solar_used = np.zeros(hours)
        battery_used = np.zeros(hours)
        grid_import = np.zeros(hours)
        excess_solar = np.zeros(hours)
        soc_history = []
        
        for i in range(hours):
            load = load_profile[i]
            solar = solar_available[i]
            used_from_solar = min(solar, load)
            remaining_load = load - used_from_solar
            excess = max(solar - used_from_solar, 0)
            charge_possible = min(battery_charge_limit, battery_capacity - battery_soc)
            charge = min(excess, charge_possible)
            battery_soc += charge * battery_efficiency
            discharge_possible = min(battery_discharge_limit, battery_soc)
            discharge = min(remaining_load, discharge_possible)
            battery_soc -= discharge / battery_efficiency
            unmet = remaining_load - discharge
            grid = max(unmet, 0)
            solar_used[i] = used_from_solar
            battery_used[i] = discharge
            grid_import[i] = grid
            excess_solar[i] = excess - charge
            soc_history.append(battery_soc)
        
        # Plot results
        if matplotlib_available:
            fig3, ax3 = plt.subplots(figsize=(12,5))
            ax3.plot(y_test.index, load_profile, label='Load', color='black', alpha=0.7)
            ax3.plot(y_test.index, solar_available, label='Solar Available', color='gold', alpha=0.7)
            ax3.fill_between(y_test.index, 0, solar_used, label='Solar Used', color='green', alpha=0.3)
            ax3.fill_between(y_test.index, solar_used, solar_used + battery_used, label='Battery Used', color='orange', alpha=0.3)
            ax3.fill_between(y_test.index, solar_used + battery_used, load_profile, label='Grid Import', color='red', alpha=0.2)
            ax3.fill_between(y_test.index, load_profile, solar_available, where=solar_available>load_profile, label='Excess Solar', color='blue', alpha=0.2)
            ax3.set_title('Load Distribution Optimization (Battery)')
            ax3.set_xlabel('Time (UTC)')
            ax3.set_ylabel('Energy (arbitrary units)')
            ax3.legend(loc='upper right')
            st.pyplot(fig3)
            
            # Battery SOC plot
            fig4, ax4 = plt.subplots(figsize=(10,2))
            ax4.plot(y_test.index, soc_history, label='Battery SOC')
            ax4.set_title('Battery State of Charge Over Time')
            ax4.set_xlabel('Time (UTC)')
            ax4.set_ylabel('SOC')
            st.pyplot(fig4)
        else:
            # Use Streamlit charts for optimization results
            optimization_df = pd.DataFrame({
                'Load': load_profile,
                'Solar Available': solar_available,
                'Solar Used': solar_used,
                'Battery Used': battery_used,
                'Grid Import': grid_import
            }, index=y_test.index)
            st.line_chart(optimization_df)
            
            # Battery SOC
            soc_df = pd.DataFrame({'Battery SOC': soc_history}, index=y_test.index)
            st.line_chart(soc_df)
        
        # --- Results Summary and Download ---
        st.subheader("Results Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Load", f"{np.sum(load_profile):.2f}")
            st.metric("Total Solar Used", f"{np.sum(solar_used):.2f}")
        with col2:
            st.metric("Total Battery Used", f"{np.sum(battery_used):.2f}")
            st.metric("Total Grid Import", f"{np.sum(grid_import):.2f}")
        with col3:
            st.metric("Total Excess Solar", f"{np.sum(excess_solar):.2f}")
            st.metric("Final Battery SOC", f"{soc_history[-1]:.2f}")
        
        # Create results DataFrame for download
        results_df = pd.DataFrame({
            'datetime': y_test.index,
            'load': load_profile,
            'solar_available': solar_available,
            'solar_used': solar_used,
            'battery_used': battery_used,
            'grid_import': grid_import,
            'excess_solar': excess_solar,
            'battery_soc': soc_history
        })
        
        # Download buttons
        st.subheader("Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # Download CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="solar_grid_optimization_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download summary report
            summary_text = f"""
Solar Grid Optimization Results
==============================

System Parameters:
- Battery Capacity: {battery_capacity} kWh
- Battery Charge Limit: {battery_charge_limit} kWh/h
- Battery Discharge Limit: {battery_discharge_limit} kWh/h
- Battery Efficiency: {battery_efficiency}
- Grid Price: ${grid_price}/kWh
- Peak Price: ${price_peak}/kWh

Results:
- Total Load: {np.sum(load_profile):.2f}
- Total Solar Used: {np.sum(solar_used):.2f}
- Total Battery Used: {np.sum(battery_used):.2f}
- Total Grid Import: {np.sum(grid_import):.2f}
- Total Excess Solar: {np.sum(excess_solar):.2f}
- Final Battery SOC: {soc_history[-1]:.2f}
- Forecast RMSE: {rmse:.2f}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            st.download_button(
                label="Download Summary Report",
                data=summary_text,
                file_name="solar_grid_optimization_summary.txt",
                mime="text/plain"
            )
        
        # Display detailed results table
        st.subheader("Detailed Results")
        st.dataframe(results_df)
        
    else:
        st.warning("Not all required feature columns are present in the uploaded data.")
else:
    st.info("Please upload a TMY.csv file to begin.") 