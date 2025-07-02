import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix, 
    roc_curve, 
    f1_score, 
    classification_report
)

# 1. Load Data
df = pd.read_excel("dataset_predict_fire.xlsx")

X = df[['Temperature (°C)', 'Water Table (meter)', 'Soil Moisture (%)', 'Rainfall (milimeter)', 'Latitude', 'Longitude']]
y = df['Label']

# 2. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Model Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features=6,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    class_weight={0: 1, 1: 35}
)

model.fit(X_train, y_train)

# 4. Prediksi dan Evaluasi
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_train_prob = model.predict_proba(X_train)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train, y_train_prob)
test_auc = roc_auc_score(y_test, y_test_prob)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Akurasi Training: {train_acc:.4f}")
print(f"Akurasi Testing : {test_acc:.4f}")
print(f"AUC Training    : {train_auc:.4f}")
print(f"AUC Testing     : {test_auc:.4f}")
print(f"F1 Score Training: {train_f1:.4f}")
print(f"F1 Score Testing: {test_f1:.4f}")

print("\nClassification Report (Testing):")
print(classification_report(y_test, y_test_pred))

# 5. Confusion Matrix Plot
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Not Fire Prone", "Fire Prone"], 
            yticklabels=["Not Fire Prone", "Fire Prone"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()

# 6. ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC Curve (ROC = {test_auc:.2f})")
plt.plot([0,1], [0,1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC Curve - Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Feature Importance Plot
importances = model.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(6,4))
sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


import joblib

# Simpan model ke file
joblib.dump(model, "randomforest_model.pkl")

new_data = pd.DataFrame([{
    'Temperature (°C)': 28.09,
    'Water Table (meter)': -0.819,
    'Soil Moisture (%)': 31.472,
    'Rainfall (milimeter)': 0.014,
    'Latitude': -2.289,
    'Longitude': 113.886
}])

# Prediksi kelas
new_pred = model.predict(new_data)[0]

# Prediksi probabilitas
new_proba = model.predict_proba(new_data)[0][1]  # Probabilitas kelas 1 (kebakaran terjadi)

# Tampilkan hasil
print(f"\nPrediksi untuk data baru: {'Terjadi kebakaran' if new_pred == 1 else 'Tidak terjadi kebakaran'}")
print(f"Probabilitas terjadinya kebakaran: {new_proba:.4f}")

