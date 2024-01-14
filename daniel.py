# Install paket yang diperlukan
# !pip install pandas pmdarima streamlit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import itertools
import matplotlib.dates as mdates
import io
import contextlib

# Muat dataset CSV
# Contoh: data = pd.read_csv('nama_file.csv')
data = pd.read_csv('minyak2.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Tampilkan beberapa baris pertama dari dataset
st.title("Daniel Irwansyah")
st.title("Eksplorasi Data")
st.subheader("Beberapa baris pertama dari dataset:")
st.write(data.head())

# Informasi umum tentang dataset
st.subheader("Informasi umum tentang dataset:")

# Alirkan pernyataan print ke dalam stream
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    data.info()

# Dapatkan informasi yang dicetak sebagai string
info_str = stream.getvalue()

# Tampilkan informasi menggunakan st.code()
st.code(info_str)

# Statistik deskriptif untuk kolom numerik
st.subheader("Statistik deskriptif untuk kolom numerik:")
st.write(data.describe())

# Visualisasi data: Harga penutupan dari waktu ke waktu
st.subheader("Harga Penutupan Historis")
st.line_chart(data.set_index('Date')['Close'])

# Analisis data eksploratif
# Analisis tren menggunakan moving average
window_size = 10
rolling_mean = data['Close'].rolling(window=window_size).mean()
data['Rolling Mean'] = rolling_mean

st.subheader("Harga Penutupan dengan Rolling Mean")
st.line_chart(data.set_index('Date')[['Close', 'Rolling Mean']])

# Korelasi antar kolom
correlation_matrix = data.corr()

# Visualisasi korelasi heatmap
st.subheader("Heatmap Korelasi")

# Buat heatmap menggunakan Matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(correlation_matrix, cmap='coolwarm')
fig.colorbar(cax)

# Atur label
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

# Tampilkan plot
st.pyplot(fig)

# Fungsi untuk menggantikan outlier dengan mean
def replace_outliers_with_mean(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    data_without_outliers = np.clip(data, lower_bound, upper_bound)
    return data_without_outliers

# Gunakan fungsi replace_outliers_with_mean
data['Close_wo'] = replace_outliers_with_mean(data['Close'])

# Plot untuk mengamati tren data
st.subheader("Tren Harga Penutupan")
st.line_chart(data.set_index('Date')['Close_wo'])

# Hitung logaritma natural
data['ln_close'] = np.log(data['Close_wo'])

# Plot untuk mengamati tren pada data yang di-log transform
st.subheader("Tren Harga Penutupan yang Di-log Transform")
st.line_chart(data.set_index('Date')['ln_close'])

# Uji ADF
result_adf = adfuller(data['ln_close'])

# Tampilkan hasil uji ADF
st.subheader("Hasil Uji ADF:")
st.write('Statistik ADF:', result_adf[0])
st.write('Nilai p:', result_adf[1])
st.write('Nilai Kritis:', result_adf[4])

# Differencing tahap pertama
data['diff_first'] = data['ln_close'].diff()

# Buang nilai NaN
data = data.dropna()

# Uji ADF untuk data setelah differencing tahap pertama
result_diff_first = adfuller(data['diff_first'])

# Tampilkan hasil uji ADF
st.subheader("Hasil Uji ADF (Setelah Differencing Tahap Pertama):")
st.write('Statistik ADF:', result_diff_first[0])
st.write('Nilai p:', result_diff_first[1])
st.write('Nilai Kritis:', result_diff_first[4])

# Plot data, data differencing, mean, dan standard deviation
window_size = 10
mean_diff_first = data['diff_first'].rolling(window=window_size).mean()
std_diff_first = data['diff_first'].rolling(window=window_size).std()

# Tambahkan kolom 'std_diff_first' ke DataFrame
data['std_diff_first'] = std_diff_first

# Perbarui nama kolom menjadi 'std_diff_first'
st.line_chart(data.set_index('Date')[['Close', 'diff_first', 'Rolling Mean', 'std_diff_first']])

# Uji ACF dan PACF
st.subheader("Fungsi Autokorelasi (ACF)")
fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
plot_acf(data['diff_first'], lags=30, zero=False, ax=ax_acf)
plt.xlabel('Lag')
plt.ylabel('ACF')
st.pyplot(fig_acf)

st.subheader("Fungsi Autokorelasi Parsial (PACF)")
fig_pacf, ax_pacf = plt.subplots(figsize=(12, 6))
plot_pacf(data['diff_first'], lags=21, zero=False, ax=ax_pacf)
plt.xlabel('Lag')
plt.ylabel('PACF')
st.pyplot(fig_pacf)

# Differencing kedua
data['diff_second'] = data['diff_first'].diff()

# Uji ADF untuk differencing kedua
result_diff_second = adfuller(data['diff_second'].dropna())
st.subheader("Hasil Uji ADF (Setelah Differencing Kedua):")
st.write('Statistik ADF:', result_diff_second[0])
st.write('Nilai p:', result_diff_second[1])
st.write('Nilai Kritis:', result_diff_second[4])

# Bagi data menjadi set pelatihan dan pengujian
train_data, test_data = train_test_split(data['diff_second'], train_size=0.8)

# Grid Search untuk parameter ARIMA
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

# Tampilkan parameter ARIMA terbaik
st.subheader("Parameter Model ARIMA (Grid Search):")
st.write(best_params)

# Buat model ARIMA dengan parameter yang dipilih
model_ARIMA = pm.ARIMA(order=best_params)

# Latih model dengan data pelatihan
result = model_ARIMA.fit(train_data)

# Proyeksi dengan model ARIMA
n_periods = len(test_data)
forecast, conf_int = result.predict(n_periods=n_periods, return_conf_int=True)

# Tampilkan hasil proyeksi
st.subheader("Proyeksi:")
st.write(forecast)

# Visualisasi data dengan data pelatihan, data pengujian, dan data yang diproyeksikan
st.subheader("Evaluasi Model")

fig, ax = plt.subplots()

# Plot data pelatihan
ax.plot(data['Date'][:train_data.shape[0]], train_data, label='Data Pelatihan')

# Plot data pengujian dan proyeksi bersama-sama
ax.plot(data['Date'][train_data.shape[0]:], test_data, label='Data Pengujian')
ax.plot(data['Date'][train_data.shape[0]:], forecast, label='Proyeksi')

# Atur lokator mayor sumbu x
ax.xaxis.set_major_locator(mdates.AutoDateLocator())

# Format label sumbu x sebagai tanggal
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Putar dan ratakan label sumbu x
plt.setp(ax.get_xticklabels(), rotation=90, ha="right")

# Hapus informasi zona waktu
ax.xaxis.get_major_formatter()._use_tzinfo = False

ax.set_xlabel('Tanggal')
ax.legend()

st.pyplot(fig)

# Evaluasi model dengan MAE pada data pengujian
st.subheader("Evaluasi Model (Mean Absolute Error):")
st.write('Mean Absolute Error:', best_mae)
