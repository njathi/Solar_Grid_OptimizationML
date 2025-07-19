import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# --- Global Configuration for Economic, Social, and Environmental Metrics ---
grid_price = 0.10  # USD per kWh (average grid price)
grid_emission_factor = 0.7  # kg CO2 per kWh (typical for Kenya)
battery_cost_per_kwh = 400  # USD per kWh (example value)
battery_capacity = 400.0  # max energy the battery can store (arbitrary units, doubled)
battery_capacity_kwh = battery_capacity  # assuming 1 unit = 1 kWh
battery_charge_limit = 30.0  # max charge per hour
battery_discharge_limit = 30.0  # max discharge per hour
battery_efficiency = 0.9   # round-trip efficiency
avg_household_size = 4  # people per household
N_households = 5  # keep in sync with simulation

# Load the data, skipping the first 17 metadata lines
df = pd.read_csv('TMY.csv', skiprows=17)

# Diagnostics: Print first 5 rows after loading
print('--- Raw Data Sample (first 5 rows) ---')
print(df.head())

# Print unique values in 'time(UTC)' and 'T2m' before conversion
if 'time(UTC)' in df.columns:
    print('\nUnique values in time(UTC) (first 10):')
    print(df['time(UTC)'].unique()[:10])
if 'T2m' in df.columns:
    print('\nUnique values in T2m (first 10):')
    print(df['T2m'].unique()[:10])

# Data Cleaning (updated datetime parsing)
# Convert time(UTC) to datetime with specified format
if 'time(UTC)' in df.columns:
    df['time(UTC)'] = pd.to_datetime(df['time(UTC)'], format='%Y%m%d:%H%M', errors='coerce')

# Convert T2m to float (if not already)
df['T2m'] = pd.to_numeric(df['T2m'], errors='coerce')

# Handle missing values: drop rows with any missing values
clean_df = df.dropna()

# Set time(UTC) as index
clean_df.set_index('time(UTC)', inplace=True)

# Display cleaned data info
print('\n--- Cleaned Data Sample ---')
print(clean_df.head())
print('\n--- Cleaned Data Info ---')
print(clean_df.info())

# Visualize the target variable G(h)
if not clean_df.empty:
    plt.figure(figsize=(12, 4))
    plt.plot(clean_df.index, clean_df['G(h)'])
    plt.title('Global Horizontal Irradiance (G(h)) Over Time')
    plt.xlabel('Time (UTC)')
    plt.ylabel('G(h)')
    plt.tight_layout()
    plt.show()
else:
    print('\nNo data to plot after cleaning.')

# --- Forecasting Model ---
if not clean_df.empty:
    # Features and target
    feature_cols = ['T2m', 'RH', 'Gb(n)', 'Gd(h)', 'IR(h)', 'WS10m', 'WD10m', 'SP']
    X = clean_df[feature_cols]
    y = clean_df['G(h)']

    # Train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'\nRandom Forest RMSE on test set: {rmse:.2f}')

    # Plot actual vs predicted
    plt.figure(figsize=(12, 4))
    plt.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
    plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted G(h) (Test Set)')
    plt.xlabel('Time (UTC)')
    plt.ylabel('G(h)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Feature Importances ---
    importances = model.feature_importances_
    feature_names = feature_cols
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print('\n--- Feature Importances ---')
    print(importance_df)

    # Plot feature importances
    plt.figure(figsize=(8, 5))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
else:
    print('No data available for modeling.')

# --- Load Distribution Optimization ---
if not clean_df.empty:
    # --- Load Real Household Data for Load Profile ---
    household_df = pd.read_csv(
        'household_power_consumption.txt',
        sep=';',
        na_values=['?'],
        low_memory=False
    )
    # Combine Date and Time columns into a single datetime, with dayfirst=True
    household_df['datetime'] = pd.to_datetime(
        household_df['Date'] + ' ' + household_df['Time'],
        dayfirst=True, errors='coerce'
    )
    household_df.set_index('datetime', inplace=True)
    household_df['Global_active_power'] = pd.to_numeric(household_df['Global_active_power'], errors='coerce')
    household_df = household_df.dropna(subset=['Global_active_power'])
    # Use lowercase 'h' for resampling
    household_hourly = household_df['Global_active_power'].resample('h').mean()
    num_hours = len(y_test)
    if len(household_hourly) < num_hours:
        raise ValueError("Not enough data in the household dataset for the test set period.")
    load_profile = household_hourly.iloc[:num_hours].values  # shape: (num_hours,)
    # Optionally, scale the load if needed (e.g., to match your solar units)
    # load_profile = load_profile * scaling_factor

    # 2. Optimization: allocate as much solar as possible to load, rest from grid
    solar_available = y_pred  # predicted G(h)
    solar_used = np.minimum(solar_available, load_profile)
    grid_import = np.maximum(load_profile - solar_available, 0)
    excess_solar = np.maximum(solar_available - load_profile, 0)

    # 3. Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.index, load_profile, label='Load Demand (Real)', color='black', alpha=0.7)
    plt.plot(y_test.index, solar_available, label='Solar Available (Predicted)', color='gold', alpha=0.7)
    plt.fill_between(y_test.index, 0, solar_used, label='Solar Used', color='green', alpha=0.3)
    plt.fill_between(y_test.index, solar_used, load_profile, label='Grid Import', color='red', alpha=0.2)
    plt.fill_between(y_test.index, load_profile, solar_available, where=solar_available>load_profile, label='Excess Solar', color='blue', alpha=0.2)
    plt.title('Load Distribution Optimization (Real Load Data)')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Energy (arbitrary units)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # 4. Print summary statistics
    total_load = np.sum(load_profile)
    total_solar_used = np.sum(solar_used)
    total_grid_import = np.sum(grid_import)
    total_excess_solar = np.sum(excess_solar)
    print(f'\nTotal Load: {total_load:.2f}')
    print(f'Total Solar Used: {total_solar_used:.2f}')
    print(f'Total Grid Import: {total_grid_import:.2f}')
    print(f'Total Excess Solar: {total_excess_solar:.2f}')

    # --- CO2 Emissions Calculation ---
    co2_emissions = grid_import * grid_emission_factor
    total_co2_emissions = np.sum(co2_emissions)
    print(f'Total Grid CO2 Emissions: {total_co2_emissions:.2f} kg')
    # Cumulative CO2 emissions plot
    plt.figure(figsize=(10, 3))
    plt.plot(y_test.index, np.cumsum(co2_emissions), label='Cumulative Grid CO2 Emissions (kg)', color='brown')
    plt.title('Cumulative Grid CO2 Emissions Over Time')
    plt.xlabel('Time (UTC)')
    plt.ylabel('CO2 Emissions (kg)')
    plt.tight_layout()
    plt.legend()
    plt.show()

# --- Advanced Load Distribution Optimization: Two Loads + Battery ---
if not clean_df.empty:
    # Use predicted G(h) for the test set as available solar energy
    hours = len(y_test)
    # Use real load profile for total load
    total_load = load_profile

    # Battery parameters
    battery_capacity = 400.0  # max energy the battery can store (arbitrary units, doubled)
    battery_soc = 0.0         # initial state of charge
    battery_soc_history = []
    battery_charge_limit = 30.0  # max charge per hour
    battery_discharge_limit = 30.0  # max discharge per hour
    battery_efficiency = 0.9   # round-trip efficiency

    solar_available = y_pred
    solar_used = np.zeros(hours)
    battery_used = np.zeros(hours)
    grid_import = np.zeros(hours)
    excess_solar = np.zeros(hours)
    battery_charge = np.zeros(hours)
    battery_discharge = np.zeros(hours)
    soc = battery_soc

    for i in range(hours):
        load = total_load[i]
        solar = solar_available[i]
        # 1. Use solar to meet load
        used_from_solar = min(solar, load)
        remaining_load = load - used_from_solar
        # 2. Store excess solar in battery
        excess = max(solar - used_from_solar, 0)
        charge_possible = min(battery_charge_limit, battery_capacity - soc)
        charge = min(excess, charge_possible)
        soc += charge * battery_efficiency
        # 3. Discharge battery to meet remaining load
        discharge_possible = min(battery_discharge_limit, soc)
        discharge = min(remaining_load, discharge_possible)
        soc -= discharge / battery_efficiency
        # 4. Any remaining unmet load is grid import
        unmet = remaining_load - discharge
        grid = max(unmet, 0)
        # 5. Track everything
        solar_used[i] = used_from_solar
        battery_used[i] = discharge
        grid_import[i] = grid
        excess_solar[i] = excess - charge
        battery_charge[i] = charge
        battery_discharge[i] = discharge
        battery_soc_history.append(soc)

    # Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, total_load, label='Total Load (Real)', color='black', alpha=0.7)
    plt.plot(y_test.index, solar_available, label='Solar Available (Predicted)', color='gold', alpha=0.7)
    plt.fill_between(y_test.index, 0, solar_used, label='Solar Used', color='green', alpha=0.3)
    plt.fill_between(y_test.index, solar_used, solar_used + battery_used, label='Battery Used', color='orange', alpha=0.3)
    plt.fill_between(y_test.index, solar_used + battery_used, total_load, label='Grid Import', color='red', alpha=0.2)
    plt.fill_between(y_test.index, total_load, solar_available, where=solar_available>total_load, label='Excess Solar', color='blue', alpha=0.2)
    plt.title('Advanced Load Distribution Optimization (Real Load + Battery)')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Energy (arbitrary units)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Battery state of charge
    plt.figure(figsize=(12, 3))
    plt.plot(y_test.index, battery_soc_history, label='Battery State of Charge')
    plt.title('Battery State of Charge Over Time')
    plt.xlabel('Time (UTC)')
    plt.ylabel('State of Charge')
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f'\nTotal Load: {np.sum(total_load):.2f}')
    print(f'Total Solar Used: {np.sum(solar_used):.2f}')
    print(f'Total Battery Used: {np.sum(battery_used):.2f}')
    print(f'Total Grid Import: {np.sum(grid_import):.2f}')
    print(f'Total Excess Solar: {np.sum(excess_solar):.2f}')
    print(f'Final Battery State of Charge: {battery_soc_history[-1]:.2f}')

    # --- CO2 Emissions Calculation ---
    co2_emissions = grid_import * grid_emission_factor
    total_co2_emissions = np.sum(co2_emissions)
    print(f'Total Grid CO2 Emissions: {total_co2_emissions:.2f} kg')
    plt.figure(figsize=(10, 3))
    plt.plot(y_test.index, np.cumsum(co2_emissions), label='Cumulative Grid CO2 Emissions (kg)', color='brown')
    plt.title('Cumulative Grid CO2 Emissions Over Time (Battery)')
    plt.xlabel('Time (UTC)')
    plt.ylabel('CO2 Emissions (kg)')
    plt.tight_layout()
    plt.legend()
    plt.show()

# --- Cost Optimization with TOU Pricing ---
if not clean_df.empty:
    # Simple TOU price profile: high price (0.30) from 18:00-22:00, low (0.10) otherwise
    price_profile = np.full(hours, 0.10)
    for i, idx in enumerate(y_test.index):
        if 18 <= idx.hour < 22:
            price_profile[i] = 0.30

    # Re-run battery dispatch to minimize cost: prioritize battery discharge during high price hours
    soc = battery_soc
    battery_soc_history_cost = []
    solar_used_cost = np.zeros(hours)
    battery_used_cost = np.zeros(hours)
    grid_import_cost = np.zeros(hours)
    excess_solar_cost = np.zeros(hours)
    battery_charge_cost = np.zeros(hours)
    battery_discharge_cost = np.zeros(hours)
    for i in range(hours):
        load = total_load[i]  # Use real load profile
        solar = solar_available[i]
        # 1. Use solar to meet load
        used_from_solar = min(solar, load)
        remaining_load = load - used_from_solar
        # 2. Store excess solar in battery
        excess = max(solar - used_from_solar, 0)
        charge_possible = min(battery_charge_limit, battery_capacity - soc)
        charge = min(excess, charge_possible)
        soc += charge * battery_efficiency
        # 3. Discharge battery to meet remaining load, but only if price is high or battery is full
        discharge = 0
        if price_profile[i] >= 0.30 or soc > battery_capacity * 0.95:
            discharge_possible = min(battery_discharge_limit, soc)
            discharge = min(remaining_load, discharge_possible)
            soc -= discharge / battery_efficiency
        # 4. Any remaining unmet load is grid import
        unmet = remaining_load - discharge
        grid = max(unmet, 0)
        # 5. Track everything
        solar_used_cost[i] = used_from_solar
        battery_used_cost[i] = discharge
        grid_import_cost[i] = grid
        excess_solar_cost[i] = excess - charge
        battery_charge_cost[i] = charge
        battery_discharge_cost[i] = discharge
        battery_soc_history_cost.append(soc)

    # Calculate cost
    total_cost = np.sum(grid_import_cost * price_profile)
    print(f'\n--- Cost Optimization Results ---')
    print(f'Total Cost: ${total_cost:.2f}')
    print(f'Total Grid Import: {np.sum(grid_import_cost):.2f}')
    print(f'Total Battery Used: {np.sum(battery_used_cost):.2f}')
    print(f'Final Battery State of Charge: {battery_soc_history_cost[-1]:.2f}')

    # --- CO2 Emissions Calculation ---
    co2_emissions = grid_import_cost * grid_emission_factor
    total_co2_emissions = np.sum(co2_emissions)
    print(f'Total Grid CO2 Emissions: {total_co2_emissions:.2f} kg')
    plt.figure(figsize=(10, 3))
    plt.plot(y_test.index, np.cumsum(co2_emissions), label='Cumulative Grid CO2 Emissions (kg)', color='brown')
    plt.title('Cumulative Grid CO2 Emissions Over Time (Cost Opt)')
    plt.xlabel('Time (UTC)')
    plt.ylabel('CO2 Emissions (kg)')
    plt.tight_layout()
    plt.legend()
    plt.show()

# --- Lookahead Battery Optimization with cvxpy ---
try:
    import cvxpy as cp
    if not clean_df.empty:
        # Use the same test set period
        n = hours
        # Parameters
        soc_init = 0.0
        soc_max = battery_capacity
        charge_max = battery_charge_limit
        discharge_max = battery_discharge_limit
        eff = battery_efficiency
        load = load_profile  # Use real load profile
        solar = solar_available
        price = price_profile

        # Variables
        grid = cp.Variable(n)
        charge = cp.Variable(n)
        discharge = cp.Variable(n)
        soc = cp.Variable(n+1)
        solar_used = cp.Variable(n)
        excess_solar = cp.Variable(n)

        constraints = []
        # Initial SOC
        constraints += [soc[0] == soc_init]
        for t in range(n):
            # Battery SOC update
            constraints += [soc[t+1] == soc[t] + eff*charge[t] - discharge[t]/eff]
            # SOC limits
            constraints += [soc[t+1] >= 0, soc[t+1] <= soc_max]
            # Charge/discharge limits
            constraints += [charge[t] >= 0, charge[t] <= charge_max]
            constraints += [discharge[t] >= 0, discharge[t] <= discharge_max]
            # Solar allocation
            constraints += [solar_used[t] + charge[t] + excess_solar[t] == solar[t]]
            constraints += [solar_used[t] >= 0, charge[t] >= 0, excess_solar[t] >= 0]
            # Load balance
            constraints += [solar_used[t] + discharge[t] + grid[t] == load[t]]
            constraints += [grid[t] >= 0]
        # Objective: minimize total grid cost
        total_cost = cp.sum(cp.multiply(grid, price))
        prob = cp.Problem(cp.Minimize(total_cost), constraints)
        prob.solve(solver=cp.ECOS, verbose=True)

        print(f'\n--- Lookahead Optimization Results ---')
        print(f'Total Cost: ${total_cost.value:.2f}')
        print(f'Total Grid Import: {cp.sum(grid).value:.2f}')
        print(f'Total Battery Used: {cp.sum(discharge).value:.2f}')
        print(f'Final Battery State of Charge: {soc[n].value:.2f}')

        # --- CO2 Emissions Calculation ---
        co2_emissions = grid.value * grid_emission_factor
        total_co2_emissions = np.sum(co2_emissions)
        print(f'Total Grid CO2 Emissions: {total_co2_emissions:.2f} kg')
        plt.figure(figsize=(10, 3))
        plt.plot(y_test.index, np.cumsum(co2_emissions), label='Cumulative Grid CO2 Emissions (kg)', color='brown')
        plt.title('Cumulative Grid CO2 Emissions Over Time (Lookahead)')
        plt.xlabel('Time (UTC)')
        plt.ylabel('CO2 Emissions (kg)')
        plt.tight_layout()
        plt.legend()
        plt.show()

        # Plot results
        plt.figure(figsize=(14, 7))
        plt.plot(y_test.index, load, label='Total Load (Real)', color='black', alpha=0.7)
        plt.plot(y_test.index, solar, label='Solar Available (Predicted)', color='gold', alpha=0.7)
        plt.fill_between(y_test.index, 0, solar_used.value, label='Solar Used', color='green', alpha=0.3)
        plt.fill_between(y_test.index, solar_used.value, solar_used.value + discharge.value, label='Battery Used', color='orange', alpha=0.3)
        plt.fill_between(y_test.index, solar_used.value + discharge.value, load, label='Grid Import', color='red', alpha=0.2)
        plt.fill_between(y_test.index, load, solar, where=solar>load, label='Excess Solar', color='blue', alpha=0.2)
        plt.title('Lookahead Battery Optimization (cvxpy, Real Load)')
        plt.xlabel('Time (UTC)')
        plt.ylabel('Energy (arbitrary units)')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        # Battery state of charge
        plt.figure(figsize=(12, 3))
        plt.plot(y_test.index, soc.value[1:], label='Battery State of Charge')
        plt.title('Battery State of Charge Over Time (Lookahead)')
        plt.xlabel('Time (UTC)')
        plt.ylabel('State of Charge')
        plt.tight_layout()
        plt.show()

        # --- Economic and Social Metrics ---
        baseline_grid_cost = np.sum(load_profile) * grid_price
        actual_grid_cost = np.sum(grid.value) * grid_price
        cost_savings = baseline_grid_cost - actual_grid_cost
        battery_investment = battery_capacity_kwh * battery_cost_per_kwh
        payback_period = battery_investment / (cost_savings + 1e-6) if cost_savings > 0 else float('inf')
        cost_per_household = actual_grid_cost / N_households
        people_benefiting = N_households * avg_household_size
        print(f'Economic & Social Metrics:')
        print(f'  Baseline Grid-Only Cost: ${baseline_grid_cost:.2f}')
        print(f'  Actual Grid Cost: ${actual_grid_cost:.2f}')
        print(f'  Cost Savings: ${cost_savings:.2f}')
        print(f'  Battery Investment: ${battery_investment:.2f}')
        print(f'  Payback Period: {payback_period:.2f} years')
        print(f'  Cost per Household: ${cost_per_household:.2f}')
        print(f'  People Benefiting: {people_benefiting}')
except ImportError:
    print('cvxpy is not installed. Please install it with "pip install cvxpy" to run lookahead optimization.')

# --- Simulate Multiple Household Load Profiles and Aggregate ---
if not clean_df.empty:
    np.random.seed(42)
    # Create N-1 synthetic variations of the real load profile
    base_profile = load_profile
    household_profiles = [base_profile]
    for i in range(N_households - 1):
        # Add random daily and hourly variation
        daily_variation = np.random.normal(1.0, 0.05, size=len(base_profile)//24 + 1)
        daily_variation = np.repeat(daily_variation, 24)[:len(base_profile)]
        hourly_variation = np.random.normal(1.0, 0.10, size=len(base_profile))
        synthetic_profile = base_profile * daily_variation * hourly_variation
        synthetic_profile = np.clip(synthetic_profile, 0, None)
        household_profiles.append(synthetic_profile)
    household_profiles = np.array(household_profiles)  # shape: (N_households, num_hours)
    # Aggregate to form community load
    community_load = np.sum(household_profiles, axis=0)
    # Visualize
    plt.figure(figsize=(14, 5))
    for i in range(N_households):
        plt.plot(y_test.index, household_profiles[i], alpha=0.5, label=f'Household {i+1}' if i < 5 else None)
    plt.plot(y_test.index, community_load, color='black', label='Community Total', linewidth=2)
    plt.title('Simulated Household and Community Load Profiles')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Load (kW)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Use community_load as the new load_profile for all subsequent optimization
    load_profile = community_load

# --- Automated Scenario Analysis ---
if not clean_df.empty:
    scenarios = {}
    # 1. Grid-only (all load from grid)
    grid_only_cost = np.sum(load_profile) * grid_price
    grid_only_co2 = np.sum(load_profile) * grid_emission_factor
    scenarios['Grid Only'] = {'cost': grid_only_cost, 'co2': grid_only_co2, 'savings': 0.0}

    # 2. Solar-only (use all available solar, no battery, no grid backup)
    solar_only_used = np.minimum(solar_available, load_profile)
    solar_only_grid = np.maximum(load_profile - solar_available, 0)
    solar_only_cost = np.sum(solar_only_grid) * grid_price
    solar_only_co2 = np.sum(solar_only_grid) * grid_emission_factor
    scenarios['Solar Only'] = {'cost': solar_only_cost, 'co2': solar_only_co2, 'savings': grid_only_cost - solar_only_cost}

    # 3. Solar + Battery (no grid backup: unmet load is lost)
    soc = 0.0
    solar_used_sb = np.zeros_like(load_profile)
    battery_used_sb = np.zeros_like(load_profile)
    grid_import_sb = np.zeros_like(load_profile)
    for i in range(len(load_profile)):
        load = load_profile[i]
        solar = solar_available[i]
        # Use solar to meet load
        used_from_solar = min(solar, load)
        remaining_load = load - used_from_solar
        # Store excess solar in battery
        excess = max(solar - used_from_solar, 0)
        charge_possible = min(battery_charge_limit, battery_capacity - soc)
        charge = min(excess, charge_possible)
        soc += charge * battery_efficiency
        # Discharge battery to meet remaining load
        discharge_possible = min(battery_discharge_limit, soc)
        discharge = min(remaining_load, discharge_possible)
        soc -= discharge / battery_efficiency
        # Any remaining unmet load is lost (no grid)
        unmet = remaining_load - discharge
        # Track
        solar_used_sb[i] = used_from_solar
        battery_used_sb[i] = discharge
        grid_import_sb[i] = 0  # no grid
    sb_cost = 0.0
    sb_co2 = 0.0
    scenarios['Solar+Battery'] = {'cost': sb_cost, 'co2': sb_co2, 'savings': grid_only_cost - sb_cost}

    # 4. Solar+Battery+Grid (current system)
    # Use results from previous optimization (assume grid_import, cost, co2 already calculated)
    sbg_cost = np.sum(grid_import) * grid_price
    sbg_co2 = np.sum(grid_import) * grid_emission_factor
    scenarios['Solar+Battery+Grid'] = {'cost': sbg_cost, 'co2': sbg_co2, 'savings': grid_only_cost - sbg_cost}

    # Print scenario comparison table
    print('\n--- Scenario Comparison Table ---')
    print(f'| Scenario              | Total Cost (USD) | Total CO2 (kg) | Cost Savings (USD) |')
    print(f'|-----------------------|------------------|----------------|--------------------|')
    for name, vals in scenarios.items():
        print(f'| {name:<21} | {vals["cost"]:>16.2f} | {vals["co2"]:>14.2f} | {vals["savings"]:>18.2f} |')
