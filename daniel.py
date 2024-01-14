import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import itertools
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io
import contextlib
import matplotlib.dates as mdates
import time

# Function to fetch data (simulated UN data)
@st.cache
def load_dataset():
    data = pd.read_csv('minyak2.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_dataset()

# Exploratory data analysis
st.title("Exploratory Data Analysis")
st.subheader("Overview of the Dataset")
st.write(data.head())

st.subheader("General Information about the Dataset:")
st.code(data.info())

st.subheader("Descriptive Statistics for Numeric Columns:")
st.write(data.describe())

# Visualizing historical closing prices over time
st.subheader("Historical Closing Prices")
st.line_chart(data.set_index('Date')['Close'])

# Exploratory Data Analysis
window_size = 10
rolling_mean = data['Close'].rolling(window=window_size).mean()
data['Rolling Mean'] = rolling_mean

st.subheader("Closing Prices with Rolling Mean")
st.line_chart(data.set_index('Date')[['Close', 'Rolling Mean']])

# Correlation between columns
correlation_matrix = data.corr()

# Visualizing correlation heatmap
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(correlation_matrix, cmap='coolwarm')
fig.colorbar(cax)

plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

st.pyplot(fig)

# Outlier replacement and trend observation
def replace_outliers_with_mean(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    data_without_outliers = np.clip(data, lower_bound, upper_bound)
    return data_without_outliers

data['Close_wo'] = replace_outliers_with_mean(data['Close'])

st.subheader("Closing Price Trends")
st.line_chart(data.set_index('Date')['Close_wo'])

# Log transformation
data['ln_close'] = np.log(data['Close_wo'])

st.subheader("Trends in Log-Transformed Closing Prices")
st.line_chart(data.set_index('Date')['ln_close'])

# Augmented Dickey-Fuller Test
result_adf = adfuller(data['ln_close'])

st.subheader("Augmented Dickey-Fuller Test Results:")
st.write('ADF Statistic:', result_adf[0])
st.write('p-value:', result_adf[1])
st.write('Critical Values:', result_adf[4])

# First differencing
data['diff_first'] = data['ln_close'].diff()
data = data.dropna()

# ADF test for first differencing
result_diff_first = adfuller(data['diff_first'])

st.subheader("ADF Test Results (After First Differencing):")
st.write('ADF Statistic:', result_diff_first[0])
st.write('p-value:', result_diff_first[1])
st.write('Critical Values:', result_diff_first[4])

# Plot data, differencing, mean, and standard deviation
window_size = 10
mean_diff_first = data['diff_first'].rolling(window=window_size).mean()
std_diff_first = data['diff_first'].rolling(window=window_size).std()

data['std_diff_first'] = std_diff_first

st.line_chart(data.set_index('Date')[['Close', 'diff_first', 'Rolling Mean', 'std_diff_first']])

# ACF and PACF plots
st.subheader("Autocorrelation Function (ACF)")
fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
plot_acf(data['diff_first'], lags=30, zero=False, ax=ax_acf)
plt.xlabel('Lag')
plt.ylabel('ACF')
st.pyplot(fig_acf)

st.subheader("Partial Autocorrelation Function (PACF)")
fig_pacf, ax_pacf = plt.subplots(figsize=(12, 6))
plot_pacf(data['diff_first'], lags=21, zero=False, ax=ax_pacf)
plt.xlabel('Lag')
plt.ylabel('PACF')
st.pyplot(fig_pacf)

# Second differencing
data['diff_second'] = data['diff_first'].diff()

# ADF test for second differencing
result_diff_second = adfuller(data['diff_second'].dropna())

st.subheader("ADF Test Results (After Second Differencing):")
st.write('ADF Statistic:', result_diff_second[0])
st.write('p-value:', result_diff_second[1])
st.write('Critical Values:', result_diff_second[4])

# Train-test split
train_data, test_data = train_test_split(data['diff_second'], train_size=0.8)

# Grid Search for ARIMA parameters
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))

best_mae = float('inf')
best_params = None

for param in pdq:
    try:
        model = pm.ARIMA(order=param)
        result = model.fit(train_data)
        forecast = result.predict(n_periods=len(test_data))
        mae = mean_absolute_error(test_data, forecast)

        if mae < best_mae:
            best_mae = mae
            best_params = param

    except:
        continue

# Display best ARIMA parameters
st.subheader("Best ARIMA Model Parameters (Grid Search):")
st.write(best_params)

# Create ARIMA model with selected parameters
model_ARIMA = pm.ARIMA(order=best_params)

# Train the model with training data
result = model_ARIMA.fit(train_data)

# ARIMA model projection
n_periods = len(test_data)
forecast, conf_int = result.predict(n_periods=n_periods, return_conf_int=True)

# Display projection results
st.subheader("Projection:")
st.write(forecast)

# Visualize data with training, testing, and projected data
st.subheader("Model Evaluation")

fig, ax = plt.subplots()

ax.plot(data['Date'][:train_data.shape[0]], train_data, label='Training Data')
ax.plot(data['Date'][train_data.shape[0]:], test_data, label='Testing Data')
ax.plot(data['Date'][train_data.shape[0]:], forecast, label='Projection')

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
ax.xaxis.get_major_formatter()._use_tzinfo = False

ax.set_xlabel('Date')
ax.legend()

st.pyplot(fig)

# Model evaluation with MAE on testing data
st.subheader("Model Evaluation (Mean Absolute Error):")
st.write('Mean Absolute Error:', best_mae)
