import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Set Halaman
# ----------------------------
st.set_page_config(page_title="Prediksi", layout="wide")

# ----------------------------
# CONFIG
# ----------------------------
config = {
    "Temperature (°C)": {"look_back": 14},
    "Water Table (meter)": {"look_back": 14},
    "Soil Moisture (%)": {"look_back": 10},
    "Rainfall (milimeter)": {"look_back": 5},
}

# ----------------------------
# FUNGSI PREDIKSI MULTI-STEP LSTM
# ----------------------------
def predict_next_steps_lstm(series, jenis, n_steps_out):
    model = load_model(f"model_{jenis}.h5", custom_objects={'mse': MeanSquaredError()})

    data = series.values.reshape(-1, 1)

    # Buat scaler baru untuk deret waktu yang ingin di prediksi
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    look_back = config[jenis]['look_back']
    if len(scaled_data) < look_back:
        print(f"Minimum {look_back} data points required for {jenis}.")
        return None

    last_window = scaled_data[-look_back:].reshape(1, look_back, 1)
    predictions = []

    for _ in range(n_steps_out):
        pred_scaled = model.predict(last_window, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred)

        # Update window
        new_input = pred_scaled.reshape(1, 1, 1)
        last_window = np.concatenate((last_window[:, 1:, :], new_input), axis=1)

    return predictions

# ----------------------------
# DASHBOARD
# ----------------------------
st.title("Prediction Page")

# ----------------------------
# Predict Time Series (LSTM)
# ----------------------------
st.header("Predict Time Series using LSTM")

uploaded_file = st.file_uploader("Upload CSV file with 4 columns: Temperature(°C ), Water Table(m), Soil Moisture(%), and Rainfall(mm)", type=["csv"])

n_steps_out = st.number_input(
    "Enter the number of steps ahead to predict (e.g., 3 = 3 days ahead)", 
    min_value=1, max_value=30, value=3, step=1
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if df.shape[1] != 4:
        st.error("CSV must contain exactly 4 columns.")
    else:
        st.write("Data Snapshot:")
        st.dataframe(df.head())


        predictions = {}

        for jenis in config.keys():
            preds = predict_next_steps_lstm(df.iloc[:, list(config.keys()).index(jenis)], jenis, n_steps_out)
            if preds is not None:
                predictions[jenis] = preds

        if predictions:
            st.success("Multi-Step Prediction Results:")

            # Gabungkan jadi DataFrame
            combined_df = pd.DataFrame(predictions)
            combined_df.index = [f"Step {i+1}" for i in range(len(combined_df))]

            # Ganti nama kolom agar sesuai
            combined_df.columns = [
                "Temperature(°C )", 
                "Water Table(m)", 
                "Soil Moisture(%)", 
                "Rainfall(mm)"
            ]

            st.dataframe(combined_df.style.format(precision=3))

            import matplotlib.pyplot as plt

            st.subheader("Time Series Plot: Input Data + Prediction")

            # Buat layout 2 baris x 2 kolom
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            axes = axes.flatten()  # Ubah ke 1D array biar lebih mudah di-loop

            name = ["Temperature(°C )", "Water Table(m)", "Soil Moisture(%)", "Rainfall(mm)"]
            i=0
            for idx, jenis in enumerate(config.keys()):
                ax = axes[idx]

                # Ambil data input asli
                data_asli = df.iloc[:, idx].values
                data_prediksi = predictions[jenis]

                input_len = len(data_asli)
                pred_len = len(data_prediksi)
                x_input = list(range(input_len))
                x_pred = list(range(input_len, input_len + pred_len))

                # Plot input (biru) dan prediksi (merah)
                ax.plot(x_input, data_asli, label='Input Data', color='blue')
                ax.plot(x_pred, data_prediksi, label='Predicted Data', color='red', marker='o', linestyle='None')

                ax.set_title(name[i])
                ax.set_xlabel("Time Step (day)")
                ax.set_ylabel(name[i])
                ax.legend()
                i+=1

            plt.tight_layout()
            st.pyplot(fig)

# ----------------------------
# Predict Fire Risk
# ----------------------------
# ----------------------------
# Predict Fire Risk
# ----------------------------
st.header("Predict Fire Risk")

if 'predictions' not in locals() or not predictions:
    st.warning("Please make your LSTM prediction first in section 1.")
else:
    st.subheader("Fire Risk Prediction based on LSTM prediction steps")

    latitude = st.number_input("Input Latitude", format="%.6f")
    longitude = st.number_input("Input Longitude", format="%.6f")

    model_choice = st.selectbox("Select classification model", ["XGBoost", "SVM", "Random Forest"])

    if st.button("🔍 Fire Risk Prediction for All Steps"):
        if model_choice == "XGBoost":
            model = joblib.load('xgboost_model.pkl')
        elif model_choice == "SVM":
            model = joblib.load('svm_model.pkl')
        elif model_choice == "Random Forest":
            model = joblib.load('randomforest_model.pkl')

        # Siapkan data input semua langkah dalam format (n_steps, 4 fitur)
        n_steps = len(next(iter(predictions.values())))
        lstm_inputs = np.array([
            [predictions["Temperature (°C)"][i],
            predictions["Water Table (meter)"][i],
            predictions["Soil Moisture (%)"][i],
            predictions["Rainfall (milimeter)"][i],
            latitude,
            longitude]
            for i in range(n_steps)
        ])

        # 💡 Perbaikan: konversi ke DataFrame dan pastikan urutan fiturnya benar
        feature_order = ['Temperature (°C)', 'Water Table (meter)', 'Soil Moisture (%)', 'Rainfall (milimeter)', 'Latitude', 'Longitude']
        rename_mapping = {
            "Temperature(°C )": "Temperature (°C)",
            "Water Table(m)": "Water Table (meter)",
            "Soil Moisture(%)": "Soil Moisture (%)",
            "Rainfall(mm)": "Rainfall (milimeter)"
        }

        # Rename kolom prediksi agar cocok dengan model training
        for old, new in rename_mapping.items():
            value = predictions.get(old)
            if value is not None:
                predictions[new] = value
                del predictions[old]


        lstm_inputs = pd.DataFrame(lstm_inputs, columns=feature_order)

        # 🔍 Prediksi
        preds = model.predict(lstm_inputs)
        try:
            probas = model.predict_proba(lstm_inputs)[:, 1]
        except:
            probas = ["N/A"] * n_steps

        # Ubah ke label deskriptif
        risk_labels = ["Not Fire-Prone" if pred == 0 else "Fire-Prone" for pred in preds]

        # Buat hasil prediksi ke dalam tabel
        result_df = pd.DataFrame({
            "Step": [f"Step {i+1}" for i in range(n_steps)],
            "Fire Risk Prediction": risk_labels,
            "Probability": [f"{p:.4f}" if p != "N/A" else p for p in probas]
        })

        st.dataframe(result_df)

