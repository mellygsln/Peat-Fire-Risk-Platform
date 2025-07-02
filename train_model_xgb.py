import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. Load Data
df1 = pd.read_excel("dataset_predict_fire.xlsx")

X = df1[['Temperature (°C)', 'Water Table (meter)', 'Soil Moisture (%)', 'Rainfall (milimeter)', 'Latitude', 'Longitude']]
y = df1['Label']

# 2. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Model XGBoost
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=10,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=10,
    scale_pos_weight=10,
    verbosity=1
)

# 4. Fit model
model.fit(X_train, y_train, 
          eval_set=[(X_train, y_train), (X_test, y_test)], 
          verbose=True)

# 5. Prediksi dan Evaluasi
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 6. AUC Training & AUC Testing
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# 7. F1 Scores
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

# 8. Classification Report
class_report = classification_report(y_test, y_test_pred)

# 10b. Confusion Matrix (Testing Data)
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Fire Prone", "Fire Prone"],
            yticklabels=["Not Fire Prone", "Fire Prone"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - XGBoost (Testing Data)")
plt.tight_layout()
plt.show()

# 9. Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Print evaluation metrics
print(f"Akurasi Training: {train_acc:.4f}")
print(f"Akurasi Testing : {test_acc:.4f}")
print(f"AUC Training    : {train_auc:.4f}")
print(f"AUC Testing     : {test_auc:.4f}")
print(f"F1 Score Training: {train_f1:.4f}")
print(f"F1 Score Testing: {test_f1:.4f}")
print("\nClassification Report (Testing):")
print(class_report)

# 10. Plot Logloss, AUC, dan Accuracy
results = model.evals_result()
epochs = len(results['validation_0']['logloss'])

train_preds = [model.predict_proba(X_train, iteration_range=(0, i))[:, 1] for i in range(1, epochs + 1)]
test_preds = [model.predict_proba(X_test, iteration_range=(0, i))[:, 1] for i in range(1, epochs + 1)]

train_auc_curve = [roc_auc_score(y_train, pred) for pred in train_preds]
test_auc_curve = [roc_auc_score(y_test, pred) for pred in test_preds]

train_acc_curve = [accuracy_score(y_train, pred > 0.5) for pred in train_preds]
test_acc_curve = [accuracy_score(y_test, pred > 0.5) for pred in test_preds]

# Plot Logloss
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(results['validation_0']['logloss'], label='Training Logloss')
plt.plot(results['validation_1']['logloss'], label='Testing Logloss')
plt.xlabel('Boosting Round')
plt.ylabel('Logloss')
plt.title('Logloss')
plt.legend()

# Plot AUC
plt.subplot(1, 3, 2)
plt.plot(train_auc_curve, label='Train AUC')
plt.plot(test_auc_curve, label='Test AUC')
plt.xlabel('Boosting Round')
plt.ylabel('ROC')
plt.title('ROC Curve')
plt.legend()

# Plot Accuracy
plt.subplot(1, 3, 3)
plt.plot(train_acc_curve, label='Train Accuracy')
plt.plot(test_acc_curve, label='Test Accuracy')
plt.xlabel('Boosting Round')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 11. Save model
joblib.dump(model, "xgboost_model.pkl")


# 12. Prediksi Data Baru (1 baris input)
new_data = pd.DataFrame([{
    'Temperature (°C)': 28.9,
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
