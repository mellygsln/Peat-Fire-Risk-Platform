import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Baca dataset
df1 = pd.read_excel("dataset_predict_fire.xlsx")

# Pisahkan fitur dan label
X = df1[['Temperature (°C)', 'Water Table (meter)', 'Soil Moisture (%)', 'Rainfall (milimeter)', 'Latitude', 'Longitude']]
y = df1['Label']

# Bagi data setelah distandarkan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#SMOTE Oversampling
#smote = SMOTE(random_state=42)
#X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Buat dan latih model SVM (menggunakan probabilitas=True untuk AUC)
model = SVC(kernel='linear', probability=True, C=1, gamma=0.1, random_state=42, class_weight={0: 1, 1: 5} )
model.fit(X_train, y_train)

# Prediksi
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_train_prob = model.predict_proba(X_train)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]

# Evaluasi
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train, y_train_prob)
test_auc = roc_auc_score(y_test, y_test_prob)

# Tambahan evaluasi dengan F1-score
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Akurasi Training : {train_acc:.4f}")
print(f"Akurasi Testing  : {test_acc:.4f}")
print(f"AUC Training     : {train_auc:.4f}")
print(f"AUC Testing      : {test_auc:.4f}")
print(f"F1 Score Training: {train_f1:.4f}")
print(f"F1 Score Testing : {test_f1:.4f}")

print("\nClassification Report (Testing Data):")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix Plot untuk Data Testing
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Not Fire Prone", "Fire Prone"], 
            yticklabels=["Not Fire Prone", "Fire Prone"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - SVM (Testing Data)")
plt.tight_layout()
plt.show()

# Visualisasi ROC dan Thresholding (simulasi boosting round)
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

plt.figure(figsize=(10, 5))

# Plot ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr_train, tpr_train, label=f'Train AUC = {train_auc:.2f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {test_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.show()

# Simpan model
joblib.dump(model, "svm_model.pkl")


new_data = pd.DataFrame([{
    'Temperature (°C)': 28.9,
    'Water Table (meter)': -0.819,
    'Soil Moisture (%)': 31.472,
    'Rainfall (milimeter)': 0.014,
    'Latitude': -2.289,
    'Longitude': 113.886
}])
#new_data = scaler.transform(new_data)
# Prediksi kelas
new_pred = model.predict(new_data)[0]

# Prediksi probabilitas
new_proba = model.predict_proba(new_data)[0][1]  # Probabilitas kelas 1 (kebakaran terjadi)

# Tampilkan hasil
print(f"\nPrediksi untuk data baru: {'Terjadi kebakaran' if new_pred == 1 else 'Tidak terjadi kebakaran'}")
print(f"Probabilitas terjadinya kebakaran: {new_proba:.4f}")
