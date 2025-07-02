import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Dropout
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG PER JENIS DATA
# ----------------------------
config = {
    "Temperature (°C)": {
        "look_back": 14,
        "lstm_units": 256,
        "epochs": 50,
        "layers": 3,
        "dropout": 0,
    },
    "Water Table (meter)": {
        "look_back": 14,
        "lstm_units": 32,
        "epochs": 50,
        "layers": 1,
        "dropout": 0.0,
    },
    "Soil Moisture (%)": {
        "look_back": 10,
        "lstm_units": 64,
        "epochs": 30,
        "layers": 1,
        "dropout": 0.0,
    },
    "Rainfall (milimeter)": {
        "look_back": 5,
        "lstm_units": 16,
        "epochs": 20,
        "layers": 1,
        "dropout": 0.0,
    },
}


# ----------------------------
# FUNGSI: Sliding Window
# ----------------------------
def create_dataset(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# ----------------------------
# FUNGSI: Plot Prediksi
# ----------------------------
def plot_predictions(jenis, original_data, y_train_pred, y_val_pred, y_test_pred, look_back, len_train, len_val):
    y_total = np.array(original_data[:]).ravel()

    # Buat array kosong untuk setiap bagian prediksi
    plot_train = np.empty_like(y_total)
    plot_val = np.empty_like(y_total)
    plot_test = np.empty_like(y_total)
    plot_train[:] = np.nan
    plot_val[:] = np.nan
    plot_test[:] = np.nan

    # Tentukan indeks akhir masing-masing bagian
    idx_train_end = len_train
    idx_val_end = idx_train_end + len_val

    # Assign prediksi TRAIN
    plot_train[0:len(y_train_pred)] = y_train_pred.ravel()

    # Assign prediksi VALIDASI
    plot_val[idx_train_end:idx_train_end + len(y_val_pred)] = y_val_pred.ravel()

    # Assign prediksi TEST
    plot_test[idx_val_end:idx_val_end + len(y_test_pred)] = y_test_pred.ravel()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_total, label='Data Asli', color='black')
    plt.plot(plot_train, label='Prediksi Train', color='blue')
    plt.plot(plot_val, label='Prediksi Val', color='orange')
    plt.plot(plot_test, label='Prediksi Test', color='red')
    plt.axvline(x=idx_train_end, color='green', linestyle='--', label='Split Train/Val')
    plt.axvline(x=idx_val_end, color='purple', linestyle='--', label='Split Val/Test')
    plt.title(f"Plot Prediksi - {jenis}")
    plt.xlabel("Time (day)")
    plt.ylabel(jenis)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_{jenis}.png")
    plt.close()

# ----------------------------
# FUNGSI: Training per jenis
# ----------------------------
def train_model_for_jenis(jenis, list_deret, config):
    params = config[jenis]
    look_back = params["look_back"]
    lstm_units = params["lstm_units"]
    epochs = params["epochs"]

    # Split list_deret jadi train, val, test (72%, 18%, 10%)
    n_total = len(list_deret)
    n_train = int(n_total * 0.72)
    n_val = int(n_total * 0.18)

    train_deret = list_deret[0:8]
    val_deret = list_deret[8:10]
    test_deret = list_deret[10:11]

    X_all = []

    # Siapkan data train
    X_train_all, y_train_all = [], []
    scalers_train = []
    for deret in train_deret:
        deret = deret.reshape(-1, 1)
        scaler_train = MinMaxScaler()
        deret_scaled = scaler_train.fit_transform(deret)
        scalers_train.append(scaler_train)

        X, y = create_dataset(deret_scaled, look_back)
        X_train_all.append(X)
        y_train_all.append(y)
        X, y = create_dataset(deret, look_back)
        X_all.append(X)
    X_train = np.concatenate(X_train_all)
    y_train = np.concatenate(y_train_all)

    # Siapkan data validasi (fit scaler baru dari val_deret)
    X_val_all, y_val_all = [], []
    scalers_val = []
    for deret in val_deret:
        deret = deret.reshape(-1, 1)
        scaler_val = MinMaxScaler()
        deret_scaled = scaler_val.fit_transform(deret)
        scalers_val.append(scaler_val)

        X, y = create_dataset(deret_scaled, look_back)
        X_val_all.append(X)
        y_val_all.append(y)
        X, y = create_dataset(deret, look_back)
        X_all.append(X)
    X_val = np.concatenate(X_val_all)
    y_val = np.concatenate(y_val_all)

    # Siapkan data test (fit scaler baru dari test_deret)
    X_test_all, y_test_all = [], []
    scalers_test = []
    for deret in test_deret:
        deret = deret.reshape(-1, 1)
        scaler_test = MinMaxScaler()
        deret_scaled = scaler_test.fit_transform(deret)  # fit pada test deret
        scalers_test.append(scaler_test)

        X, y = create_dataset(deret_scaled, look_back)
        X_test_all.append(X)
        y_test_all.append(y)
        X, y = create_dataset(deret, look_back)
        X_all.append(X)
    X_test = np.concatenate(X_test_all)
    y_test = np.concatenate(y_test_all)

    # Reshape untuk LSTM
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_val = X_val.reshape((X_val.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    # MODEL
    model = Sequential()
    for i in range(params.get("layers", 1)):
        return_sequences = i < params["layers"] - 1  # True jika bukan layer terakhir
        if i == 0:
            model.add(LSTM(lstm_units, return_sequences=return_sequences, input_shape=(look_back, 1)))
        else:
            model.add(LSTM(lstm_units, return_sequences=return_sequences))
        if params.get("dropout", 0) > 0:
            model.add(Dropout(params["dropout"]))

    model.add(Dense(1))  # output layer
    model.compile(optimizer='adam', loss='mse')


    history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=8,
    shuffle=False,
    verbose=1
    )

    # VISUALISASI LOSS
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curve - {jenis}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'loss_{jenis}.png')
    plt.close()

    # INVERS TRAIN
    y_train_pred = model.predict(X_train)
    y_train_pred_inv = []
    y_train_true_inv = []
    start_idx = 0
    for deret, scaler_train in zip(train_deret, scalers_train):
        deret = deret.reshape(-1, 1)
        deret_scaled = scaler_train.transform(deret)
        X_tmp, y_tmp = create_dataset(deret_scaled, look_back)
        n = len(y_tmp)
        y_pred_tmp = y_train_pred[start_idx:start_idx+n]
        y_train_pred_inv.append(scaler_train.inverse_transform(y_pred_tmp))
        y_train_true_inv.append(scaler_train.inverse_transform(y_tmp.reshape(-1, 1)))
        start_idx += n
    y_train_pred_inv = np.vstack(y_train_pred_inv)
    y_train_true_inv = np.vstack(y_train_true_inv)

    # INVERS VAL
    y_val_pred = model.predict(X_val)
    y_val_pred_inv = []
    y_val_true_inv = []
    start_idx = 0
    for deret, scaler_val in zip(val_deret, scalers_val):
        deret = deret.reshape(-1, 1)
        deret_scaled = scaler_val.transform(deret)
        X_tmp, y_tmp = create_dataset(deret_scaled, look_back)
        n = len(y_tmp)
        y_pred_tmp = y_val_pred[start_idx:start_idx+n]
        y_val_pred_inv.append(scaler_val.inverse_transform(y_pred_tmp))
        y_val_true_inv.append(scaler_val.inverse_transform(y_tmp.reshape(-1, 1)))
        start_idx += n
    y_val_pred_inv = np.vstack(y_val_pred_inv)
    y_val_true_inv = np.vstack(y_val_true_inv)

    # INVERS TEST
    y_test_pred = model.predict(X_test)
    inverse_preds = []
    inverse_truths = []
    start_idx = 0
    for deret, scaler_test in zip(test_deret, scalers_test):
        deret = deret.reshape(-1, 1)
        deret_scaled = scaler_test.transform(deret)  # Transform tanpa fit ulang
        X_tmp, y_tmp = create_dataset(deret_scaled, look_back)
        n = len(y_tmp)
        y_pred_tmp = y_test_pred[start_idx:start_idx+n]
        y_pred_inv = scaler_test.inverse_transform(y_pred_tmp)
        y_true_inv = scaler_test.inverse_transform(y_tmp.reshape(-1, 1))
        inverse_preds.append(y_pred_inv)
        inverse_truths.append(y_true_inv)
        start_idx += n
    y_pred_total = np.vstack(inverse_preds)
    y_true_total = np.vstack(inverse_truths)

    # HITUNG METRIK
    rmse = np.sqrt(mean_squared_error(y_true_total, y_pred_total))
    mae = mean_absolute_error(y_true_total, y_pred_total)
    print(f"[{jenis}] RMSE (skala asli): {rmse:.4f}, MAE (skala asli): {mae:.4f}")

    # Plot prediksi
    train_deret = [series[look_back:] for series in train_deret]
    val_deret = [series[look_back:] for series in val_deret]
    test_deret = [series[look_back:] for series in test_deret]
        # Gabungkan seluruh data asli dari test_deret pertama (bisa ganti jika ingin)
    sample_all = np.concatenate([np.concatenate(train_deret), np.concatenate(val_deret), np.concatenate(test_deret)]).reshape(-1, 1)

    plot_predictions(
        jenis,
        sample_all,
        y_train_pred_inv,
        y_val_pred_inv,
        y_pred_total, 
        look_back,
        len(y_train_pred_inv),
        len(y_val_pred_inv)
    )
    
    # Simpan model
    model.save(f"model_{jenis}.h5")
    print(f"Model untuk '{jenis}' berhasil disimpan.")

# ----------------------------
# FUNGSI: Simulasi data 
# ----------------------------
import pandas as pd

def load_data_per_jenis():
    # Membaca file Excel
    xls = pd.ExcelFile('dataset_time_series.xlsx')

    data_per_jenis = {
        "Temperature (°C)": [],
        "Water Table (meter)": [],
        "Soil Moisture (%)": [],
        "Rainfall (milimeter)": []
    }

    # Iterasi per sheet di Excel
    for sheet_name in xls.sheet_names:
        sheet = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Masukkan data dari setiap kolom jenis ke dalam data_per_jenis
        data_per_jenis["Temperature (°C)"].append(sheet["Temperature (°C)"].values)
        data_per_jenis["Water Table (meter)"].append(sheet["Water Table (meter)"].values)
        data_per_jenis["Soil Moisture (%)"].append(sheet["Soil Moisture (%)"].values)
        data_per_jenis["Rainfall (milimeter)"].append(sheet["Rainfall (milimeter)"].values)

    return data_per_jenis

# ----------------------------
# MAIN
# ----------------------------
def main():
    data_per_jenis = load_data_per_jenis()
    for jenis, list_deret in data_per_jenis.items():
        print(f"\n[INFO] Training model untuk: {jenis}")
        train_model_for_jenis(jenis, list_deret, config)

if __name__ == "__main__":
    main()


