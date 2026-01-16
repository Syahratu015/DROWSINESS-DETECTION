import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load Dataset
print("Membaca dataset...")
df = pd.read_csv('dataset_mata.csv')

# Cek data kosong
if df.isnull().values.any():
    print("WARNING: Ada data kosong! Membersihkan...")
    df = df.dropna()

print(f"Total Data: {len(df)} baris")
print(f"Jumlah Label 0 (Melek): {len(df[df['label'] == 0])}")
print(f"Jumlah Label 1 (Ngantuk): {len(df[df['label'] == 1])}")

# 2. Split Features (X) & Label (y)
# Kita pakai 3 fitur: ear_left, ear_right, ear_avg
X = df[['ear_left', 'ear_right', 'ear_avg']]
y = df['label']

# 3. Split Training & Testing (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Inisialisasi Model Naive Bayes (Gaussian)
model = GaussianNB()

# 5. Training
print("Sedang melatih model...")
model.fit(X_train, y_train)

# 6. Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"AKURASI MODEL: {accuracy * 100:.2f}%")
print("-" * 30)
print("\nLaporan Detail:")
print(classification_report(y_test, y_pred, target_names=['Melek', 'Ngantuk']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Simpan Model
model_filename = "drowsiness_model.pkl"
joblib.dump(model, model_filename)
print(f"\nModel berhasil disimpan ke: {model_filename} 🧠")
print("Siap dipakai di aplikasi utama!")
