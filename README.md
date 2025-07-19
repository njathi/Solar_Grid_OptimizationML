# Solar-Grid OptimizationML

## Project Overview
This project models, forecasts, and optimizes the integration of solar, grid, and battery systems for a community or microgrid. It uses real and simulated household load data, machine learning for solar forecasting, and advanced optimization to minimize costs and emissions. The project is designed to support the United Nations Sustainable Development Goals (SDGs), especially SDG 7 (Affordable and Clean Energy), SDG 13 (Climate Action), SDG 11 (Sustainable Cities and Communities), SDG 9 (Industry, Innovation, and Infrastructure), and SDG 12 (Responsible Consumption and Production).

## Features
- **Solar Forecasting:** Uses Random Forest regression to predict solar irradiance.
- **Load Modeling:** Integrates real and simulated household/community load profiles.
- **Optimization:** Supports grid-only, solar-only, solar+battery, and solar+battery+grid scenarios, including lookahead battery dispatch.
- **Scenario Analysis:** Automatically compares cost, CO₂ emissions, and savings across all scenarios.
- **CO₂, Cost, and Social Metrics:** Calculates emissions, cost savings, payback period, affordability, and number of people benefiting.
- **Visualization:** Plots for load, solar, battery state, grid import, and cumulative CO₂ emissions.

## How to Run
1. **Install dependencies:**
   ```bash
   pip install pandas matplotlib scikit-learn numpy cvxpy
   ```
2. **Place data files:**
   - `TMY.csv` (solar/meteorological data)
   - `household_power_consumption.txt` (UCI household load data)
3. **Run the script:**
   ```bash
   python TMY.py
   ```

## Required Dependencies
- pandas
- matplotlib
- scikit-learn
- numpy
- cvxpy (for lookahead optimization)

## Interpreting Outputs
- **Plots:** Visualize actual vs. predicted solar, load profiles, battery state, grid import, and cumulative CO₂ emissions.
- **Scenario Table:** Compares total cost, CO₂ emissions, and cost savings for each scenario (grid-only, solar-only, solar+battery, solar+battery+grid).
- **Metrics:** Printed after each optimization: cost savings, payback period, affordability, and people benefiting.

## SDG Alignment
| SDG   | Project Contribution |
|-------|---------------------|
| 7     | Expands access to affordable, clean energy via solar and storage optimization |
| 13    | Reduces CO₂ emissions by maximizing renewable use and minimizing grid reliance |
| 11    | Supports resilient, sustainable community energy systems |
| 9     | Demonstrates innovative microgrid and demand-side management techniques |
| 12    | Promotes efficient, responsible energy consumption |

## Example Impact Summary (for Stakeholders)
- **Community Size:** 5 households (can be scaled)
- **People Benefiting:** 20 (with avg. household size = 4)
- **CO₂ Emissions Avoided:** See scenario table output (compare grid-only vs. optimized)
- **Cost Savings:** See scenario table output
- **Payback Period:** Calculated for battery investment
- **Resilience:** System can operate with solar, battery, and grid backup

## Customization
- Adjust the number of households, battery size, grid price, and emission factors at the top of `TMY.py`.
- Replace or expand load data for larger communities or different regions.

## Contact
For questions or collaboration, please contact the project maintainer. "# Solar_Grid_OptimazationML" 
