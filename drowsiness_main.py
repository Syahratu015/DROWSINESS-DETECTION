import cv2
import mediapipe as mp
import numpy as np

# --- 1. SETUP ---

# Indeks Titik Mata (Landmark Indices)
LEFT_EYE = [33, 160, 158, 133, 153, 144] 
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Fungsi Menghitung Jarak Euclidean (Pythagoras)
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Fungsi Menghitung Eye Aspect Ratio (EAR)
def calculate_ear(eye_landmarks, img_w, img_h):
    # Konversi landmaks (normalized) ke pixel (x, y)
    coords = []
    for point in eye_landmarks:
        coords.append(np.array([point.x * img_w, point.y * img_h]))
    
    # Referensi Titik EAR:
    # P1 (Kiri)   = index 0
    # P2 (Atas1)  = index 1
    # P3 (Atas2)  = index 2
    # P4 (Kanan)  = index 3
    # P5 (Bawah2) = index 4
    # P6 (Bawah1) = index 5

    # Hitung Jarak Vertikal (Kelopak Atas ke Bawah)
    d_A = euclidean_distance(coords[1], coords[5]) 
    d_B = euclidean_distance(coords[2], coords[4]) 
    
    # Hitung Jarak Horizontal (Ujung ke Ujung)
    d_C = euclidean_distance(coords[0], coords[3]) 

    # Rumus Dasar EAR
    ear = (d_A + d_B) / (2.0 * d_C)
    return ear


# Prioritaskan Import Manual (HACK untuk Windows Version Conflict)
try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    from mediapipe.python.solutions import face_mesh
    mp_face_mesh = face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,     
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 2. MAIN LOOP ---
cap = cv2.VideoCapture(0)

print("Tahap 3 Dimulai: Menghitung EAR...")

# --- CONFIG & VARIABEL ---
# Berapa frame mata harus tertutup agar dianggap ngantuk?
# Asumsi Webcam 30 FPS. Jadi 2 detik = 30 * 2 = 60 frame.
DROWSY_LIMIT = 50  # Kita set 50 frame (sedikit kurang dari 2 detik biar responsif)
COUNTER = 0       # Menghitung frame berturut-turut

while True:
    ret, frame = cap.read()
    if not ret: break

    # Flip & Convert RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    img_h, img_w = frame.shape[:2]

    # Default status
    status_text = "Active"
    color_status = (0, 255, 0) # Hijau
    
    # Reset alert text variable untuk frame ini
    alert_message = "" 

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Ambil objek landmark mata kiri & kanan
            left_eye_landmarks  = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE]

            # Hitung EAR
            left_ear  = calculate_ear(left_eye_landmarks, img_w, img_h)
            right_ear = calculate_ear(right_eye_landmarks, img_w, img_h)
            avg_ear = (left_ear + right_ear) / 2.0

            # --- LOGIKA TIMER (COUNTER) ---
            threshold = 0.20
            
            if avg_ear < threshold:
                # Mata Tertutup -> Tambah Counter
                COUNTER += 1
                eye_status = "Eyes Closed"
                
                # Jika counter melebihi batas (sudah > 2 detik)
                if COUNTER >= DROWSY_LIMIT:
                    status_text = "Drowsiness Detected!" # Teks Status Panjang
                    color_status = (0, 0, 255) # Merah
                else:
                    # Belum sampai limit, status masih 'Active' tapi warna kuning (Warning)
                    status_text = "Eyes Closing..."
                    color_status = (0, 255, 255) # Kuning
            else:
                # Mata Terbuka -> Reset Counter
                COUNTER = 0
                eye_status = "Eyes Open"
                status_text = "Active"
                color_status = (0, 255, 0) # Hijau

            # --- VISUALISASI ---
            
            # 1. Bounding Box
            h, w, _ = frame.shape
            x_min, x_max, y_min, y_max = w, 0, h, 0
            for pt in face_landmarks.landmark:
                x, y = int(pt.x * w), int(pt.y * h)
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y
            
            padding = 20
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

            # Gambar Kotak
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_status, 1)
            
            # Siku Corner
            L, t = 30, 5
            cv2.line(frame, (x_min, y_min), (x_min + L, y_min), color_status, t)
            cv2.line(frame, (x_min, y_min), (x_min, y_min + L), color_status, t)
            cv2.line(frame, (x_max, y_min), (x_max - L, y_min), color_status, t)
            cv2.line(frame, (x_max, y_min), (x_max, y_min + L), color_status, t)
            cv2.line(frame, (x_min, y_max), (x_min + L, y_max), color_status, t)
            cv2.line(frame, (x_min, y_max), (x_min, y_max - L), color_status, t)
            cv2.line(frame, (x_max, y_max), (x_max - L, y_max), color_status, t)
            cv2.line(frame, (x_max, y_max), (x_max, y_max - L), color_status, t)

            # 2. Titik Mata
            for point in left_eye_landmarks + right_eye_landmarks:
                px, py = int(point.x * w), int(point.y * h)
                cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

            # 3. Panel Info
            overlay = frame.copy()
            # Lebarkan kotak background (450px) agar muat teks panjang
            cv2.rectangle(overlay, (10, 10), (450, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, f"Status: {status_text}", (30, 90), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color_status, 2)
            
    # Tampilkan Frame
    cv2.imshow('Drowsiness Check - Tahap 3', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
