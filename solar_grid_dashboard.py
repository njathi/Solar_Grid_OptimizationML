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
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Solar Grid Optimization Dashboard", layout="wide")
st.title("Solar Grid Optimization Dashboard")
st.markdown("""
This dashboard allows you to upload TMY weather data and real load profiles, forecast solar generation, and optimize load distribution with battery storage.\
You can adjust system parameters and visualize the results interactively.
""")

# --- Sidebar for user inputs ---
st.sidebar.header("System Parameters")

# --- Custom Scenario Management ---
if 'scenarios' not in st.session_state:
    st.session_state['scenarios'] = {}
if 'selected_scenario' not in st.session_state:
    st.session_state['selected_scenario'] = None

st.sidebar.header("Custom Scenarios")
scenario_names = list(st.session_state['scenarios'].keys())
selected_scenario = st.sidebar.selectbox(
    "Load Saved Scenario",
    ["<None>"] + scenario_names,
    index=0 if st.session_state['selected_scenario'] is None else scenario_names.index(st.session_state['selected_scenario']) + 1
)
if selected_scenario != "<None>":
    if st.sidebar.button("Load Scenario"):
        params = st.session_state['scenarios'][selected_scenario]
        st.session_state['selected_scenario'] = selected_scenario
        # Update parameter widgets
        st.session_state['battery_capacity'] = params['battery_capacity']
        st.session_state['battery_charge_limit'] = params['battery_charge_limit']
        st.session_state['battery_discharge_limit'] = params['battery_discharge_limit']
        st.session_state['battery_efficiency'] = params['battery_efficiency']
        st.session_state['grid_price'] = params['grid_price']
        st.session_state['price_peak'] = params['price_peak']
        st.session_state['peak_start'] = params['peak_start']
        st.session_state['peak_end'] = params['peak_end']
    if st.sidebar.button("Delete Scenario"):
        del st.session_state['scenarios'][selected_scenario]
        st.session_state['selected_scenario'] = None
        st.experimental_rerun()

with st.sidebar.expander("Save Current Scenario"):
    scenario_name = st.text_input("Scenario Name", key="scenario_name")
    if st.button("Save Scenario"):
        if scenario_name:
            st.session_state['scenarios'][scenario_name] = {
                'battery_capacity': st.session_state.get('battery_capacity', 400),
                'battery_charge_limit': st.session_state.get('battery_charge_limit', 30),
                'battery_discharge_limit': st.session_state.get('battery_discharge_limit', 30),
                'battery_efficiency': st.session_state.get('battery_efficiency', 0.9),
                'grid_price': st.session_state.get('grid_price', 0.10),
                'price_peak': st.session_state.get('price_peak', 0.30),
                'peak_start': st.session_state.get('peak_start', 18),
                'peak_end': st.session_state.get('peak_end', 22)
            }
            st.session_state['selected_scenario'] = scenario_name
            st.success(f"Scenario '{scenario_name}' saved!")
        else:
            st.warning("Please enter a scenario name.")

# --- Reset to Defaults Button ---
defaults = {
    'battery_capacity': 400,
    'battery_charge_limit': 30,
    'battery_discharge_limit': 30,
    'battery_efficiency': 0.9,
    'grid_price': 0.10,
    'price_peak': 0.30,
    'price_offpeak': 0.10,
    'peak_start': 18,
    'peak_end': 22
}
if st.sidebar.button('Reset to defaults'):
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state['selected_scenario'] = None
    st.experimental_rerun()

# --- Parameter widgets (use session_state for dynamic updates) ---
battery_capacity = st.sidebar.slider(
    "Battery Capacity (kWh)", 100, 1000, st.session_state.get('battery_capacity', 400), 50, key='battery_capacity',
    help="Total energy storage capacity of the battery in kilowatt-hours (kWh). Higher values mean more stored energy."
)
battery_charge_limit = st.sidebar.slider(
    "Battery Charge Limit (kWh/h)", 10, 100, st.session_state.get('battery_charge_limit', 30), 5, key='battery_charge_limit',
    help="Maximum rate at which the battery can be charged per hour (kWh/h)."
)
battery_discharge_limit = st.sidebar.slider(
    "Battery Discharge Limit (kWh/h)", 10, 100, st.session_state.get('battery_discharge_limit', 30), 5, key='battery_discharge_limit',
    help="Maximum rate at which the battery can be discharged per hour (kWh/h)."
)
battery_efficiency = st.sidebar.slider(
    "Battery Efficiency", 0.7, 1.0, st.session_state.get('battery_efficiency', 0.9), 0.01, key='battery_efficiency',
    help="Fraction of energy retained during charge/discharge cycles. 1.0 means no losses."
)
grid_price = st.sidebar.number_input(
    "Grid Price (USD/kWh)", value=st.session_state.get('grid_price', 0.10), step=0.01, key='grid_price',
    help="Standard price per kWh for grid electricity (used if TOU pricing is not enabled)."
)
# --- Time-of-Use (TOU) Pricing ---
st.sidebar.header("Time-of-Use (TOU) Pricing")
peak_price = st.sidebar.number_input(
    "Peak Price (USD/kWh)", value=st.session_state.get('price_peak', 0.30), step=0.01, key='price_peak',
    help="Grid electricity price per kWh during peak hours."
)
offpeak_price = st.sidebar.number_input(
    "Off-Peak Price (USD/kWh)", value=st.session_state.get('price_offpeak', 0.10), step=0.01, key='price_offpeak',
    help="Grid electricity price per kWh during off-peak hours."
)
peak_start = st.sidebar.number_input(
    "Peak Start Hour", value=st.session_state.get('peak_start', 18), min_value=0, max_value=23, key='peak_start',
    help="Hour of day (0-23) when peak pricing starts."
)
peak_end = st.sidebar.number_input(
    "Peak End Hour", value=st.session_state.get('peak_end', 22), min_value=0, max_value=23, key='peak_end',
    help="Hour of day (0-23) when peak pricing ends."
)

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

# --- Solar Plant Data uploader ---
st.subheader("Solar Plant Data (Optional)")
solarplant_file = st.file_uploader("Upload solarplant_data.csv", type=["csv"])
solarplant_df = None
hourly_plant = None
if solarplant_file is not None:
    solarplant_df = pd.read_csv(solarplant_file)
    solarplant_df['datetime'] = pd.to_datetime(solarplant_df['Updated Time'], dayfirst=True)
    solarplant_df.set_index('datetime', inplace=True)
    # Convert W to kW
    solarplant_df['solar_generation_kw'] = solarplant_df['Production Power(W)'] / 1000.0
    solarplant_df['load_kw'] = solarplant_df['Consumption Power(W)'] / 1000.0
    # Resample only numeric columns
    numeric_cols = ['solar_generation_kw', 'load_kw']
    hourly_plant = solarplant_df[numeric_cols].resample('H').mean()
    st.write("Solar Plant Data Sample (Hourly):")
    st.dataframe(hourly_plant[['solar_generation_kw', 'load_kw']].head())

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

# --- Main logic: Use solar plant data if provided, else use TMY/forecast ---
if solarplant_df is not None and hourly_plant is not None:
    st.success("Using real solar plant data for both solar generation and load profile.")
    solar_available = hourly_plant['solar_generation_kw'].values
    load_profile = hourly_plant['load_kw'].values
    hours = len(solar_available)
    time_index = hourly_plant.index
    rmse = np.nan  # Not applicable
else:
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
                forecast_df = pd.DataFrame({
                    'Actual': y_test.values,
                    'Predicted': y_pred
                }, index=y_test.index)
                st.line_chart(forecast_df)
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
            if load_profile is not None:
                if len(load_profile) >= hours:
                    load_profile = load_profile[:hours]
                    st.success(f"Using real load profile with {len(load_profile)} data points")
                else:
                    st.warning(f"Load profile has only {len(load_profile)} points, but need {hours}. Using synthetic load.")
                    load_profile = None
            if load_profile is None:
                t = np.arange(hours)
                load_profile = 50 + 30 * np.sin(2 * np.pi * (t % 24) / 24) + np.random.normal(0, 5, size=hours)
                load_profile = np.clip(load_profile, 0, None)
                st.info("Using synthetic load profile")
            time_index = y_test.index
        else:
            st.warning("Not all required feature columns are present in the uploaded data.")
            solar_available = None
            time_index = None
            rmse = np.nan
    else:
        st.info("Please upload a TMY.csv file to begin.")
        solar_available = None
        load_profile = None
        time_index = None
        rmse = np.nan

# --- Scenario Analysis Selection ---
st.sidebar.header("Scenario Analysis")
scenario = st.sidebar.selectbox(
    "Select Scenario to View in Detail",
    ["Hybrid (Solar + Battery + Grid)", "Grid-only", "Solar-only"]
)

# --- Scenario Analysis Logic ---
# Constants for carbon emissions and tree equivalence
CO2_PER_KWH_GRID = 0.7  # kg CO2 per kWh (adjust as needed for your grid)
CO2_ABSORBED_PER_TREE_PER_YEAR = 21  # kg CO2 per tree per year (conservative estimate)

scenario_results = {}
scenarios_to_run = ["Hybrid (Solar + Battery + Grid)", "Grid-only", "Solar-only"]

def get_tou_price(hour):
    if peak_start <= peak_end:
        return peak_price if peak_start <= hour < peak_end else offpeak_price
    else:
        # Peak period wraps around midnight
        return peak_price if (hour >= peak_start or hour < peak_end) else offpeak_price

for sc in scenarios_to_run:
    if solar_available is not None and load_profile is not None and time_index is not None:
        if sc == "Grid-only":
            solar_used = np.zeros_like(load_profile)
            battery_used = np.zeros_like(load_profile)
            grid_import = load_profile.copy()
            excess_solar = np.zeros_like(load_profile)
            soc_history = [0] * len(load_profile)
            unmet_load = np.zeros_like(load_profile)
        elif sc == "Solar-only":
            solar_used = np.minimum(solar_available, load_profile)
            battery_used = np.zeros_like(load_profile)
            grid_import = np.zeros_like(load_profile)
            excess_solar = np.maximum(solar_available - load_profile, 0)
            soc_history = [0] * len(load_profile)
            unmet_load = np.maximum(load_profile - solar_available, 0)
        elif sc == "Hybrid (Solar + Battery + Grid)":
            battery_soc = 0.0
            solar_used = np.zeros(len(load_profile))
            battery_used = np.zeros(len(load_profile))
            grid_import = np.zeros(len(load_profile))
            excess_solar = np.zeros(len(load_profile))
            soc_history = []
            unmet_load = np.zeros(len(load_profile))
            for i in range(len(load_profile)):
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
                unmet_load[i] = unmet if unmet > 0 else 0
        # TOU cost calculation
        if time_index is not None:
            tou_prices = np.array([get_tou_price(ts.hour) for ts in time_index])
            tou_cost = np.sum(grid_import * tou_prices)
        else:
            tou_cost = np.sum(grid_import) * offpeak_price
        # Carbon emissions and tree equivalence
        total_grid_import = np.sum(grid_import)
        total_co2 = total_grid_import * CO2_PER_KWH_GRID
        trees_needed = total_co2 / CO2_ABSORBED_PER_TREE_PER_YEAR
        scenario_results[sc] = {
            'Load': np.sum(load_profile),
            'Solar Used': np.sum(solar_used),
            'Battery Used': np.sum(battery_used),
            'Grid Import': total_grid_import,
            'Excess Solar': np.sum(excess_solar),
            'Final Battery SOC': soc_history[-1] if soc_history else 0,
            'Unmet Load': np.sum(unmet_load),
            'CO2 Emissions (kg)': total_co2,
            'Trees Needed': trees_needed,
            'Cost ($)': tou_cost,
            'Time Index': time_index,
            'Load Profile': load_profile,
            'Solar Available': solar_available,
            'Solar Used Array': solar_used,
            'Battery Used Array': battery_used,
            'Grid Import Array': grid_import,
            'Excess Solar Array': excess_solar,
            'SOC History': soc_history,
            'Unmet Load Array': unmet_load
        }

# --- Visualization and Comparison Table ---
if scenario_results:
    st.subheader("Scenario Comparison Table")
    comp_df = pd.DataFrame([
        {
            'Scenario': sc,
            'Grid Import (kWh)': scenario_results[sc]['Grid Import'],
            'Solar Used (kWh)': scenario_results[sc]['Solar Used'],
            'Battery Used (kWh)': scenario_results[sc]['Battery Used'],
            'Unmet Load (kWh)': scenario_results[sc]['Unmet Load'],
            'CO2 Emissions (kg)': scenario_results[sc]['CO2 Emissions (kg)'],
            'Trees Needed': scenario_results[sc]['Trees Needed'],
            'Cost ($)': scenario_results[sc]['Cost ($)']
        }
        for sc in scenarios_to_run
    ])
    # Plotly bar chart for scenario comparison
    fig_comp = px.bar(
        comp_df.melt(id_vars='Scenario', var_name='Metric', value_name='Value'),
        x='Scenario', y='Value', color='Metric', barmode='group',
        title='Scenario Comparison',
        hover_data=['Metric', 'Value']
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    st.dataframe(comp_df.set_index('Scenario'))
    # Visualize selected scenario
    st.subheader(f"Detailed Results: {scenario}")
    res = scenario_results[scenario]
    results_df = pd.DataFrame({
        'datetime': res['Time Index'],
        'Load': res['Load Profile'],
        'Solar Available': res['Solar Available'],
        'Solar Used': res['Solar Used Array'],
        'Battery Used': res['Battery Used Array'],
        'Grid Import': res['Grid Import Array'],
        'Excess Solar': res['Excess Solar Array'],
        'Battery SOC': res['SOC History'],
        'Unmet Load': res['Unmet Load Array']
    })
    # Plotly time-series chart
    fig_ts = go.Figure()
    for col in ['Load', 'Solar Available', 'Solar Used', 'Battery Used', 'Grid Import']:
        fig_ts.add_trace(go.Scatter(x=results_df['datetime'], y=results_df[col], mode='lines', name=col))
    fig_ts.update_layout(title='Energy Flows Over Time', xaxis_title='Time', yaxis_title='kW', hovermode='x unified')
    st.plotly_chart(fig_ts, use_container_width=True)
    # Battery SOC time-series
    fig_soc = px.line(results_df, x='datetime', y='Battery SOC', title='Battery State of Charge Over Time')
    st.plotly_chart(fig_soc, use_container_width=True)
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Load", f"{res['Load']:.2f}")
        st.metric("Total Solar Used", f"{res['Solar Used']:.2f}")
    with col2:
        st.metric("Total Battery Used", f"{res['Battery Used']:.2f}")
        st.metric("Total Grid Import", f"{res['Grid Import']:.2f}")
    with col3:
        st.metric("CO2 Emissions (kg)", f"{res['CO2 Emissions (kg)']:.2f}")
        st.metric("Trees Needed", f"{res['Trees Needed']:.2f}")
    st.subheader("Download Results")
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"solar_grid_optimization_results_{scenario.replace(' ', '_').lower()}.csv",
        mime="text/csv"
    )
    summary_text = f"""
Solar Grid Optimization Results ({scenario})
===========================================

System Parameters:
- Battery Capacity: {battery_capacity} kWh
- Battery Charge Limit: {battery_charge_limit} kWh/h
- Battery Discharge Limit: {battery_discharge_limit} kWh/h
- Battery Efficiency: {battery_efficiency}
- Grid Price: ${grid_price}/kWh
- Peak Price: ${price_peak}/kWh

Results:
- Total Load: {res['Load']:.2f}
- Total Solar Used: {res['Solar Used']:.2f}
- Total Battery Used: {res['Battery Used']:.2f}
- Total Grid Import: {res['Grid Import']:.2f}
- Total Excess Solar: {np.sum(res['Excess Solar Array']):.2f}
- Final Battery SOC: {res['Final Battery SOC']:.2f}
- Unmet Load: {res['Unmet Load']:.2f}
- CO2 Emissions: {res['CO2 Emissions (kg)']:.2f} kg
- Trees Needed: {res['Trees Needed']:.2f}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    st.download_button(
        label="Download Summary Report",
        data=summary_text,
        file_name=f"solar_grid_optimization_summary_{scenario.replace(' ', '_').lower()}.txt",
        mime="text/plain"
    )
    st.dataframe(results_df) 