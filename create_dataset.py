import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

# --- KONFIGURASI ---
OUTPUT_FILE = "dataset_mata.csv"
TARGET_SAMPLES = 1000  # Jumlah data per kelas (1000 baris Awake, 1000 baris Drowsy)

# Inisialisasi MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Indeks Mata
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def calculate_ear(eye_landmarks, img_w, img_h):
    coords = []
    for point in eye_landmarks:
        coords.append(np.array([point.x * img_w, point.y * img_h]))
    d_A = euclidean_distance(coords[1], coords[5])
    d_B = euclidean_distance(coords[2], coords[4])
    d_C = euclidean_distance(coords[0], coords[3])
    ear = (d_A + d_B) / (2.0 * d_C)
    return ear

def record_data(label_name, label_code, cap):
    data_points = []
    print(f"\n--- PERSIAPAN REKAM: {label_name} ---")
    print("Tekan 'SPACE' untuk mulai merekam...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # Tampilkan instruksi
        cv2.putText(frame, f"Mode: {label_name} (Label {label_code})", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 1)
        cv2.putText(frame, "Tekan SPASI untuk mulai", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow("Data Collector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    print(f"MEREKAM {label_name}...")
    counter = 0
    
    while counter < TARGET_SAMPLES:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_inv = [face_landmarks.landmark[i] for i in LEFT_EYE]
                right_inv = [face_landmarks.landmark[i] for i in RIGHT_EYE]
                
                ear_left = calculate_ear(left_inv, w, h)
                ear_right = calculate_ear(right_inv, w, h)
                ear_avg = (ear_left + ear_right) / 2.0
                
                # Simpan data: [ear_kiri, ear_kanan, ear_rata2, label]
                row = [ear_left, ear_right, ear_avg, label_code]
                data_points.append(row)
                counter += 1

                # Visualisasi Progress
                cv2.putText(frame, f"Merekam: {counter}/{TARGET_SAMPLES}", (30, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        # Tampilkan Frame
        cv2.putText(frame, f"ACTION: {label_name}", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 1)
        cv2.imshow("Data Collector", frame)
        cv2.waitKey(1) # Penting agar frame update
    
    print(f"Selesai merekam {label_name}.")
    return data_points

# --- MAIN ---
cap = cv2.VideoCapture(0)

# 1. Rekam Data MELEK (Kode 0)
data_awake = record_data("MELEK NORMAL", 0, cap)

# Jeda sebentar
print("Istirahat 3 detik...")
time.sleep(3)

# 2. Rekam Data NGANTUK (Kode 1)
data_drowsy = record_data("MATA TERTUTUP/NGANTUK", 1, cap)

cap.release()
cv2.destroyAllWindows()

# 3. Simpan ke CSV
print(f"Menyimpan ke {OUTPUT_FILE}...")
with open(OUTPUT_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Header
    writer.writerow(['ear_left', 'ear_right', 'ear_avg', 'label'])
    # Isi Data
    writer.writerows(data_awake)
    writer.writerows(data_drowsy)

print(f"Berhasil! Dataset tersimpan dengan total {len(data_awake) + len(data_drowsy)} baris.")
