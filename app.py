import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# --- 1. SETUP PAGE CONFIG ---
st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="👁️",
    layout="wide"
)

# --- 2. CSS STYLING (Clean & Modern) ---
st.markdown("""
    <style>
    /* Background Utama: Putih Soft enak di mata */
    .stApp {
        background-color: #F8F9FA;
        color: #212529;
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Helvetica', sans-serif;
        font-size: 3rem;
        color: #2C3E50;
        text-align: center;
        font-weight: 700;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tombol Start/Stop */
    div.stButton > button {
        border-radius: 12px;
        height: 50px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    /* Styling Card untuk Video */
    .video-container {
        border: 2px solid #E9ECEF;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def calculate_ear(eye_landmarks, img_w, img_h):
    coords = []
    for point in eye_landmarks:
        coords.append(np.array([point.x * img_w, point.y * img_h]))
    
    # EAR Formula (Vertikal / Horizontal)
    d_A = euclidean_distance(coords[1], coords[5]) 
    d_B = euclidean_distance(coords[2], coords[4]) 
    d_C = euclidean_distance(coords[0], coords[3]) 
    ear = (d_A + d_B) / (2.0 * d_C)
    return ear

# Load MediaPipe (Saved in cache to avoid reloading)
@st.cache_resource
def load_mediapipe():
    try:
        mp_face_mesh = mp.solutions.face_mesh
    except AttributeError:
        from mediapipe.python.solutions import face_mesh
        mp_face_mesh = face_mesh
    return mp_face_mesh

# Indeks Mata
LEFT_EYE = [33, 160, 158, 133, 153, 144] 
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# --- 4. MAIN INTERFACE ---
st.markdown('<div class="main-header">👁️ Drowsiness Detection System</div>', unsafe_allow_html=True)

# Initialize Session State untuk tombol Start/Stop
if 'run' not in st.session_state:
    st.session_state['run'] = False

def start_detection():
    st.session_state['run'] = True

def stop_detection():
    st.session_state['run'] = False

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # Tombol UI yang lebih bagus
    col_start, col_stop = st.columns(2)
    with col_start:
        st.button("START", on_click=start_detection, type="primary", use_container_width=True)
    with col_stop:
        st.button("STOP", on_click=stop_detection, use_container_width=True)

    st.divider()
    
    st.write("**Konfigurasi Sensitivitas:**")
    EAR_THRESHOLD = st.slider("EAR Threshold", 0.15, 0.40, 0.20, 0.01)
    DROWSY_LIMIT = st.slider("Durasi Ngantuk (Frame)", 10, 100, 50, 5)
    
    st.info(f"""
    **Status Sistem:**
    {'✅ AKTIF' if st.session_state['run'] else '⛔ NON-AKTIF'}
    """)
    
    input_source = st.selectbox("Sumber Kamera", ["Webcam Default", "Eksternal Camera"])
    source_index = 0 if input_source == "Webcam Default" else 1

# --- 5. CENTERED VIDEO LAYOUT ---
# Membuat 3 kolom. Video akan ditaruh di kolom tengah (col2) agar center.
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    # Placeholder video diletakkan di tengah
    video_placeholder = st.empty()
    
    # Jika sistem mati, tampilkan Frame Hitam (TV Off Effect)
    if not st.session_state['run']:
        # Buat gambar hitam kosong ukuran 640x480
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Tulis teks di tengah
        cv2.putText(black_frame, "SYSTEM STANDBY", (160, 240), cv2.FONT_HERSHEY_DUPLEX, 1.2, (100, 100, 100), 2)
        cv2.putText(black_frame, "Press START to Activate", (190, 280), cv2.FONT_HERSHEY_PLAIN, 1.2, (80, 80, 80), 1)
        
        video_placeholder.image(black_frame, channels="RGB", use_container_width=True, caption="Kamera Non-Aktif")

# --- 6. LOGIC LOOP ---
if st.session_state['run']:
    mp_face_mesh = load_mediapipe()
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(source_index)
    
    COUNTER = 0

    while cap.isOpened() and st.session_state['run']:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membaca kamera/Kamera terputus.")
            stop_detection()
            break

        # Flip & Convert Logic
        frame = cv2.flip(frame, 1)
        # Note: Kita simpan original frame untuk drawing BGR
        
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w, _ = frame.shape
        
        # Default var
        status_text = "Active"
        color_status = (0, 255, 0)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # Extract Landmarks
                left_eye_landmarks  = [face_landmarks.landmark[i] for i in LEFT_EYE]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE]

                # Calc EAR
                left_ear  = calculate_ear(left_eye_landmarks, w, h)
                right_ear = calculate_ear(right_eye_landmarks, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                # Logic Timer
                if avg_ear < EAR_THRESHOLD:
                    COUNTER += 1
                    eye_status = "Eyes Closed"
                    if COUNTER >= DROWSY_LIMIT:
                        status_text = "Drowsiness Detected!"
                        color_status = (0, 0, 255) # Merah (BGR untuk OpenCV)
                    else:
                        status_text = "Eyes Closing..."
                        color_status = (0, 255, 255) # Kuning (BGR)
                else:
                    COUNTER = 0
                    eye_status = "Eyes Open"
                    status_text = "Active"
                    color_status = (0, 255, 0) # Hijau (BGR)

                # --- DRAWING (Opencv uses BGR) ---
                draw_color = color_status # Sudah format BGR

                # 1. Bounding Box
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

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), draw_color, 1)

                # Fancy Corners
                L, t = 30, 5
                cv2.line(frame, (x_min, y_min), (x_min + L, y_min), draw_color, t)
                cv2.line(frame, (x_min, y_min), (x_min, y_min + L), draw_color, t)
                cv2.line(frame, (x_max, y_min), (x_max - L, y_min), draw_color, t)
                cv2.line(frame, (x_max, y_min), (x_max, y_min + L), draw_color, t)
                cv2.line(frame, (x_min, y_max), (x_min + L, y_max), draw_color, t)
                cv2.line(frame, (x_min, y_max), (x_min, y_max - L), draw_color, t)
                cv2.line(frame, (x_max, y_max), (x_max - L, y_max), draw_color, t)
                cv2.line(frame, (x_max, y_max), (x_max - L, y_max), draw_color, t)

                # 2. Eye Dots
                for point in left_eye_landmarks + right_eye_landmarks:
                    px, py = int(point.x * w), int(point.y * h)
                    cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

                # 3. Panel Info
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (450, 120), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, f"Status: {status_text}", (30, 90), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, draw_color, 2)

        # Convert BGR to RGB for Streamlit Display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display to Streamlit (Centered in col2)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
    cap.release()

