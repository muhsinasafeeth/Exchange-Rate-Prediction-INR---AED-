# 💱 INR to AED Exchange Rate Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> A comprehensive machine learning project that predicts **Indian Rupee (INR) to UAE Dirham (AED)** exchange rates using four powerful algorithms — from classical statistics to deep learning.

---

## 📌 Overview

Exchange rate forecasting is one of the most challenging problems in financial machine learning. This project fetches **real historical INR/AED forex data** using `yfinance` and applies four different prediction approaches to compare their accuracy and behavior.

---

## 🤖 Algorithms Implemented

| # | Algorithm | Type | Key Strength |
|---|-----------|------|-------------|
| 1 | **Linear Regression** | Classical ML | Fast, interpretable baseline |
| 2 | **Random Forest** | Ensemble ML | Captures non-linear patterns |
| 3 | **ARIMA** | Statistical Time Series | Statistically rigorous, no feature engineering |
| 4 | **LSTM** | Deep Learning | Learns long-range temporal dependencies |

---

## 📂 Project Structure

```
├── INR_AED_Exchange_Rate_Prediction.ipynb   # Main Jupyter Notebook
├── README.md                                # Project documentation
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | Yahoo Finance via `yfinance` (live fetch) |
| **Ticker** | `INRAED=X` |
| **Period** | 5 Years of daily data |
| **Frequency** | Daily (Business days) |
| **Target** | Closing exchange rate (1 INR = ? AED) |
| **No. of Features** | 18+ engineered features (lags, rolling stats, calendar) |

> ✅ **No manual CSV download needed** — data is fetched live every time the notebook runs!

---

## ⚙️ Feature Engineering

Since ML models don't inherently understand time, the following features were engineered to give temporal context:

| Feature Group | Features |
|--------------|---------|
| **Lag Features** | Rate from 1, 2, 3, 5, 7, 14, 21, 30 days ago |
| **Moving Averages** | 7-day, 14-day, 30-day rolling means |
| **Rolling Std Dev** | 7-day and 14-day rolling standard deviations |
| **Calendar Features** | Day of month, Month, Day of week |
| **LSTM Sequences** | Sliding window of 60 timesteps |

---

## 🧠 Model Details

### 📈 Linear Regression
- Uses lag features to learn: *"given rates over the past N days, predict tomorrow"*
- Coefficients reveal which lag periods influence the rate most

### 🌲 Random Forest
- 200 decision trees trained on bootstrap samples
- Captures non-linear relationships between lag features and future rates
- Built-in feature importance ranking

### 📉 ARIMA (5, 1, 0)
- **p=5** → uses 5 past values (AutoRegressive)
- **d=1** → first-order differencing for stationarity
- **q=0** → no moving average term
- Walk-forward validation for realistic performance measurement

### 🧠 LSTM Architecture
```
Input (60 timesteps, 1 feature)
    ↓
LSTM (128 units) → Dropout (0.2)
    ↓
LSTM (64 units)  → Dropout (0.2)
    ↓
Dense (32, ReLU)
    ↓
Dense (1) → Predicted Rate
```
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error
- Early Stopping: patience=10 on validation loss

---

## 📈 Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ | MAPE ↓ |
|-------|--------|--------|------|--------|
| Linear Regression | ~0.00012 | ~0.00016 | ~0.97 | ~0.28% |
| Random Forest | ~0.00010 | ~0.00014 | ~0.98 | ~0.23% |
| ARIMA | ~0.00009 | ~0.00013 | ~0.98 | ~0.21% |
| **LSTM** | **~0.00007** | **~0.00010** | **~0.99** | **~0.16%** |

> 🏆 **Best Model: LSTM** — lowest error and highest R²
> ⚠️ **Baseline Model: Linear Regression** — fastest but least accurate

*Exact values depend on the live data fetched at runtime.*

---

## 📉 Visualizations Included

- 📊 5-year historical INR/AED rate chart
- 📈 Moving averages (30 / 90 / 200 day)
- 📉 Daily returns with positive/negative fill
- 🔁 Rolling volatility chart
- 📦 Yearly boxplot distribution
- 🔬 ACF & PACF plots (for ARIMA tuning)
- 🆚 Actual vs Predicted plots for all 4 models
- 📊 Grouped bar chart — MAE / RMSE / MAPE comparison
- ⭐ R² score comparison chart
- 🔮 30-day future forecast with confidence band

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install yfinance statsmodels tensorflow scikit-learn matplotlib seaborn pandas numpy jupyter
```

### Run the Notebook
```bash
# Clone the repository
git clone https://github.com/your-username/INR-AED-Exchange-Rate-Prediction.git
cd INR-AED-Exchange-Rate-Prediction

# Launch Jupyter
jupyter notebook INR_AED_Exchange_Rate_Prediction.ipynb
```

### Or Open in Google Colab (Recommended)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/INR-AED-Exchange-Rate-Prediction/blob/main/INR_AED_Exchange_Rate_Prediction.ipynb)

> All libraries install automatically in the first cell — just click **Run All**!

---

## 🛠️ Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `yfinance` | Latest | Fetch live forex data |
| `pandas` | 1.5+ | Data manipulation |
| `numpy` | 1.23+ | Numerical operations |
| `scikit-learn` | 1.0+ | Linear Regression, Random Forest, metrics |
| `statsmodels` | 0.13+ | ARIMA, ADF test, ACF/PACF plots |
| `tensorflow` | 2.x | LSTM deep learning model |
| `matplotlib` | 3.5+ | Plotting and visualizations |
| `seaborn` | 0.12+ | Statistical plots and styling |

---

## 📝 Key Takeaways

1. **Never shuffle** time series data — always split chronologically (past → train, future → test)
2. **Feature scaling** (MinMaxScaler to [0,1]) is critical for LSTM convergence
3. **ARIMA** requires stationarity — ADF test + differencing (d=1) ensures this
4. **LSTM** outperforms classical models by learning complex long-range patterns
5. **Walk-forward validation** gives the most realistic ARIMA performance estimate
6. Exchange rates are influenced by geopolitical events that no model can predict from history alone

---

## ⚠️ Disclaimer

> This project is built **purely for educational and academic purposes**.
> The predictions made by these models should **NOT** be used as financial or investment advice.
> Forex markets are highly volatile and influenced by many unpredictable factors.

---

## 👤 Author

**Muhsina Safeeth**
- GitHub: [@muhsinasafeeth](https://github.com/muhsinasafeeth)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
