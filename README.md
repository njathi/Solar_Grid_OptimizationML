# Solar Grid Optimization ML Dashboard

A comprehensive web application for solar energy forecasting, load distribution optimization, and battery storage management. This dashboard enables users to upload weather data and load profiles, run machine learning forecasts, and optimize energy distribution to minimize grid dependency and costs.

## 🌟 Features

### Core Functionality
- **Solar Energy Forecasting**: Machine learning-based prediction of solar generation using weather data
- **Load Distribution Optimization**: Intelligent allocation of solar, battery, and grid energy
- **Battery Storage Management**: Real-time battery state-of-charge tracking and optimization
- **Cost Analysis**: Time-of-use pricing optimization and cost savings calculation

### Data Management
- **Weather Data Upload**: Support for TMY (Typical Meteorological Year) CSV files
- **Real Load Profile Upload**: Upload actual load consumption data in CSV format
- **Synthetic Load Generation**: Built-in load profile generation for testing and demonstration
- **Smart Column Detection**: Automatic identification of datetime and load columns in uploaded files

### Visualization & Analysis
- **Interactive Plots**: Real-time visualization of solar forecasts, load profiles, and optimization results
- **Feature Importance Analysis**: ML model interpretability with feature importance rankings
- **Battery State Tracking**: Real-time monitoring of battery charge/discharge cycles
- **Energy Flow Visualization**: Clear breakdown of solar used, battery used, and grid import

### Export & Reporting
- **CSV Export**: Download complete optimization results as CSV files
- **Summary Reports**: Generate detailed text reports with key metrics and parameters
- **Parameter Tracking**: All system parameters and results are documented in exports

### User Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Parameter Adjustment**: Interactive sliders and inputs for system configuration
- **Multi-user Support**: Each user gets their own session and data isolation
- **Error Handling**: Graceful handling of file uploads and processing errors

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd solar-grid-optimizationml
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run solar_grid_dashboard.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Data Format Requirements

### Weather Data (TMY.csv)
- **Format**: CSV file with TMY weather data
- **Required Columns**: `time(UTC)`, `T2m`, `RH`, `G(h)`, `Gb(n)`, `Gd(h)`, `IR(h)`, `WS10m`, `WD10m`, `SP`
- **Time Format**: `YYYYMMDD:HHMM` (e.g., `20150101:0000`)
- **Skip Rows**: First 17 rows should be metadata

### Load Profile Data
- **Format**: CSV file with load consumption data
- **Required Columns**: 
  - Datetime column (containing "time", "date", or "datetime" in name)
  - Load column (containing "load", "power", or "demand" in name)
- **Example**:
  ```csv
  datetime,load
  2025-01-01 00:00:00,45.2
  2025-01-01 01:00:00,42.1
  ```

## 🎛️ System Parameters

### Battery Configuration
- **Battery Capacity**: 100-1000 kWh (default: 400 kWh)
- **Charge Limit**: 10-100 kWh/h (default: 30 kWh/h)
- **Discharge Limit**: 10-100 kWh/h (default: 30 kWh/h)
- **Efficiency**: 0.7-1.0 (default: 0.9)

### Grid Pricing
- **Base Grid Price**: USD per kWh (default: $0.10)
- **Peak Price**: USD per kWh (default: $0.30)
- **Peak Hours**: Configurable start and end hours (default: 18:00-22:00)

## 🔧 Usage Guide

### Step 1: Upload Data
1. **Upload Weather Data**: Use the "Weather Data" uploader to upload your TMY.csv file
2. **Upload Load Profile** (Optional): Use the "Load Profile" uploader for real load data
3. **Use Synthetic Load**: Check the box to use a generated load profile if no real data is available

### Step 2: Configure Parameters
1. **Adjust Battery Settings**: Use the sidebar sliders to configure battery capacity and limits
2. **Set Grid Prices**: Configure base and peak electricity prices
3. **Define Peak Hours**: Set the time window for peak pricing

### Step 3: Run Analysis
1. **Review Data**: Check the uploaded data samples and cleaning results
2. **Monitor Forecasting**: View the ML model performance and feature importances
3. **Analyze Optimization**: Examine the load distribution optimization results
4. **Download Results**: Export CSV files and summary reports

## 📈 Understanding Results

### Forecasting Metrics
- **RMSE**: Root Mean Square Error of solar generation predictions
- **Feature Importance**: Ranking of weather variables' impact on solar generation

### Optimization Results
- **Total Load**: Sum of all energy demand
- **Solar Used**: Energy directly consumed from solar generation
- **Battery Used**: Energy supplied by battery storage
- **Grid Import**: Energy purchased from the grid
- **Excess Solar**: Solar energy that couldn't be used or stored
- **Battery SOC**: Final state of charge of the battery

### Cost Analysis
- **Grid Cost**: Total cost of grid electricity imports
- **Cost Savings**: Reduction in grid costs compared to grid-only scenario
- **Payback Period**: Time to recover battery investment through savings

## 🌐 Deployment

### Local Deployment
```bash
streamlit run solar_grid_dashboard.py
```

### Streamlit Cloud Deployment
1. **Push to GitHub**: Upload your code to a public GitHub repository
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
3. **Deploy**: Click "Deploy" and wait for the build to complete

### Requirements for Cloud Deployment
- **Public Repository**: Code must be in a public GitHub repository
- **requirements.txt**: Must be present in the root directory
- **Main File**: Must be named `solar_grid_dashboard.py` or specified in deployment settings

## 🔍 Technical Details

### Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Features**: Temperature, humidity, irradiance, wind speed, pressure
- **Target**: Global horizontal irradiance (G(h))
- **Validation**: 80/20 train-test split with time series preservation

### Optimization Algorithm
- **Method**: Greedy optimization with battery constraints
- **Objective**: Minimize grid import while meeting load demand
- **Constraints**: Battery capacity, charge/discharge limits, efficiency losses

### Visualization Backend
- **Primary**: Matplotlib for detailed, customizable plots
- **Fallback**: Streamlit native charts for compatibility
- **Interactive**: Real-time updates based on parameter changes

## 🛠️ Troubleshooting

### Common Issues

**Matplotlib Import Error**
- **Solution**: The app automatically falls back to Streamlit charts
- **Prevention**: Ensure matplotlib is installed: `pip install matplotlib`

**File Upload Errors**
- **Check Format**: Ensure CSV files have the correct column names
- **Check Encoding**: Use UTF-8 encoding for CSV files
- **Check Size**: Large files may take time to process

**Memory Issues**
- **Reduce Data**: Use smaller time periods for analysis
- **Close Other Apps**: Free up system memory
- **Restart**: Restart the Streamlit app if needed

### Performance Tips
- **Data Size**: Optimal performance with datasets under 10,000 rows
- **Browser**: Use modern browsers (Chrome, Firefox, Safari, Edge)
- **Network**: Stable internet connection for cloud deployment

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the development team

## 🔄 Version History

### v1.0.0 (Current)
- Initial release with core functionality
- Solar forecasting with Random Forest
- Battery optimization
- Real load profile support
- Export capabilities
- Streamlit Cloud deployment support

---

**Built with ❤️ for sustainable energy optimization** 
