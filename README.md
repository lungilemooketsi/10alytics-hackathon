# African Sovereign Debt Crisis - Early Warning System

## 📊 Project Overview
Machine Learning Early Warning System for predicting fiscal crises and simulating economic contagion across African nations.

Built for the 10alytics Data Science Hackathon.

## 🏗️ Project Structure
```
10alytics-hackathon/
│
├── data/                          # Raw and processed data
│   ├── raw_fiscal_data.xlsx      # Input: Raw fiscal data (you provide)
│   └── cleaned_fiscal_data.csv   # Output: Cleaned data (auto-generated)
│
├── src/                           # Python scripts
│   ├── data_cleaner.py           # Data cleaning & feature engineering
│   ├── model_engine.py           # ML model training & prediction
│   └── pioneer_simulation.py     # Stress testing & contagion analysis
│
├── outputs/                       # Generated results
│   ├── feature_importance.png    # Feature importance plot
│   ├── at_risk_countries_2025.csv # Top 5 at-risk countries
│   ├── stress_test_results.csv   # Stress test outcomes
│   ├── contagion_correlation_heatmap.png  # Contagion visualization
│   └── high_contagion_pairs.csv  # Highly correlated country pairs
│
├── notebooks/                     # Jupyter notebooks for exploration
│
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your raw fiscal data in `data/raw_fiscal_data.xlsx`

**Expected format:**
- **Columns:** `Country`, `Year`, `Indicator`, `Value`, `Unit`
- **Indicators:** Total_Debt, GDP, Budget_Deficit, Tax_Revenue, etc.
- **Units:** Million, Billion, or Trillion

**Example:**
| Country | Year | Indicator | Value | Unit |
|---------|------|-----------|-------|------|
| Nigeria | 2023 | Total_Debt | 87.3 | Billion |
| Nigeria | 2023 | GDP | 477.4 | Billion |
| Kenya | 2023 | Total_Debt | 68.5 | Billion |

### 3. Run the Pipeline

#### Step 1: Clean Data
```powershell
python src/data_cleaner.py
```
**What it does:**
- ✅ Normalizes all values to Billions
- ✅ Pivots from long to wide format
- ✅ Engineers features: `Debt_to_GDP`, `Deficit_to_GDP`, `Tax_to_GDP`
- ✅ Saves to `data/cleaned_fiscal_data.csv`

#### Step 2: Train Model
```powershell
python src/model_engine.py
```
**What it does:**
- ✅ Trains Random Forest Classifier
- ✅ Predicts if Debt_to_GDP will exceed 70% next year
- ✅ Saves feature importance plot to `outputs/`
- ✅ Identifies Top 5 At-Risk Countries for 2025

#### Step 3: Run Simulations
```powershell
python src/pioneer_simulation.py
```
**What it does:**
- ✅ Simulates 20% revenue shock for Nigeria (oil crash scenario)
- ✅ Recalculates risk probability post-shock
- ✅ Analyzes debt correlation across countries
- ✅ Identifies contagion pairs (correlation > 0.8)
- ✅ Generates correlation heatmap

## 🧠 Key Features

### 1. Unit Normalization
Handles mixed units (Million/Billion/Trillion) and normalizes to Billions:
```python
def normalize_unit(value, unit):
    if 'million' in unit.lower():
        return value / 1000
    elif 'trillion' in unit.lower():
        return value * 1000
    return value
```

### 2. Feature Engineering
Calculates critical fiscal ratios:
- **Debt_to_GDP:** `(Total_Debt / GDP) × 100`
- **Deficit_to_GDP:** `(Budget_Deficit / GDP) × 100`
- **Tax_to_GDP:** `(Tax_Revenue / GDP) × 100`

### 3. Crisis Prediction
Binary classification target:
```python
Crisis = 1 if Next_Year_Debt_to_GDP > 70% else 0
```

### 4. Stress Testing
Simulates economic shocks:
- Revenue drop by 20% (oil crash scenario)
- Recalculates deficit and risk scores
- Compares pre/post-shock probabilities

### 5. Contagion Analysis
Identifies systemic risk:
- Calculates debt correlation matrix
- Flags country pairs with >0.8 correlation
- Visualizes contagion networks

## 📈 Expected Outputs

### 1. Feature Importance Plot
`outputs/feature_importance.png`
- Shows top 15 features driving crisis predictions

### 2. At-Risk Countries Report
`outputs/at_risk_countries_2025.csv`
```
Rank  Country      Risk Score  Current Debt/GDP
1     Ghana        87.3%       85.4%
2     Egypt        81.2%       92.1%
3     Kenya        76.5%       68.3%
...
```

### 3. Stress Test Results
`outputs/stress_test_results.csv`
```
Country   Shock_Type     Shock_Magnitude  Risk_Increase
Nigeria   Revenue Shock  -20%             +23.4%
```

### 4. Contagion Pairs
`outputs/high_contagion_pairs.csv`
```
Country_1  Country_2  Correlation
Egypt      Ghana      0.87
Kenya      Tanzania   0.82
```

## 🔧 Customization

### Change Target Country for Stress Test
Edit `pioneer_simulation.py`:
```python
target_country = "Kenya"  # Change from Nigeria
shock_percentage = -30    # Change shock magnitude
```

### Adjust Crisis Threshold
Edit `model_engine.py`:
```python
df['Crisis_Next_Year'] = (df['Next_Year_Debt_to_GDP'] > 80).astype(int)  # Change from 70
```

### Modify Correlation Threshold
Edit `pioneer_simulation.py`:
```python
if corr_value > 0.7:  # Change from 0.8
```

## 📊 Sample Data Format

If you don't have data yet, create `data/raw_fiscal_data.xlsx` with this structure:

| Country | Year | Indicator | Value | Unit |
|---------|------|-----------|-------|------|
| Nigeria | 2020 | Total_Debt | 85.0 | Billion |
| Nigeria | 2020 | GDP | 432.3 | Billion |
| Nigeria | 2020 | Budget_Deficit | 15.2 | Billion |
| Nigeria | 2020 | Tax_Revenue | 35.4 | Billion |
| Nigeria | 2021 | Total_Debt | 92.6 | Billion |
| Nigeria | 2021 | GDP | 440.8 | Billion |
| Kenya | 2020 | Total_Debt | 65.0 | Billion |
| Kenya | 2020 | GDP | 101.0 | Billion |

## 🛠️ Troubleshooting

### Issue: "Input file not found"
**Solution:** Ensure `data/raw_fiscal_data.xlsx` exists and has correct format.

### Issue: "No revenue columns found"
**Solution:** Rename columns in your Excel file to include "Revenue" or "Tax" in the name.

### Issue: Feature columns mismatch
**Solution:** Run `data_cleaner.py` first before running other scripts.

### Issue: Not enough data for predictions
**Solution:** Ensure you have at least 2+ years of data per country.

## 📦 Dependencies
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning
- **seaborn** - Statistical visualization
- **matplotlib** - Plotting
- **openpyxl** - Excel file handling

## 🎯 Hackathon Deliverables

✅ **Data Pipeline:** Robust cleaning with unit normalization  
✅ **ML Model:** Random Forest with 80/20 train-test split  
✅ **Risk Reports:** Top 5 at-risk countries for 2025  
✅ **Stress Testing:** Revenue shock simulation  
✅ **Contagion Analysis:** Correlation-based systemic risk  
✅ **Visualizations:** Feature importance & correlation heatmap  

## 🤝 Contributing
This is a hackathon project. Feel free to extend:
- Add more indicators (inflation, interest rates)
- Implement LSTM for time-series forecasting
- Add Streamlit dashboard for interactive exploration

## 📄 License
MIT License - Open for educational and research purposes

## 👨‍💻 Author
Built for 10alytics Data Science Hackathon

---

**Ready to predict the next fiscal crisis? Run the pipeline and good luck! 🚀**
