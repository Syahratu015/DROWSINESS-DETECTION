import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib

# --- 1. SETUP PAGE CONFIG ---
st.set_page_config(
    page_title="Drowsiness Detection (AI Powered)",
    page_icon="🤖",
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
    
    d_A = euclidean_distance(coords[1], coords[5]) 
    d_B = euclidean_distance(coords[2], coords[4]) 
    d_C = euclidean_distance(coords[0], coords[3]) 
    ear = (d_A + d_B) / (2.0 * d_C)
    return ear

# Load MediaPipe
@st.cache_resource
def load_mediapipe():
    try:
        mp_face_mesh = mp.solutions.face_mesh
    except AttributeError:
        from mediapipe.python.solutions import face_mesh
        mp_face_mesh = face_mesh
    return mp_face_mesh

# Load AI Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('drowsiness_model.pkl')
        return model
    except Exception as e:
        st.error(f"Model error: {e}")
        return None

# Indeks Mata
LEFT_EYE = [33, 160, 158, 133, 153, 144] 
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# --- 4. MAIN INTERFACE ---
st.markdown('<div class="main-header">🤖 Drowsiness Detection (AI)</div>', unsafe_allow_html=True)

# Initialize Session State
if 'run' not in st.session_state:
    st.session_state['run'] = False

def start_detection():
    st.session_state['run'] = True

def stop_detection():
    st.session_state['run'] = False

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    col_start, col_stop = st.columns(2)
    with col_start:
        st.button("START", on_click=start_detection, type="primary", use_container_width=True)
    with col_stop:
        st.button("STOP", on_click=stop_detection, use_container_width=True)

    st.divider()
    
    st.success("✅ Model AI Aktif: Gaussian Naive Bayes")
    
    # Threshold Slider DIHAPUS, ganti dengan timer saja
    DROWSY_LIMIT = st.slider("Durasi Toleransi (Frame)", 10, 100, 40, 5)
    
    st.info(f"""
    **Cara Kerja AI:**
    1. Mengukur mata.
    2. Bertanya ke Model Naive Bayes.
    3. Jika Model bilang 'Ngantuk' > {DROWSY_LIMIT} frame -> Alarm.
    """)
    
    input_source = st.selectbox("Sumber Kamera", ["Webcam Default", "Eksternal Camera"])
    source_index = 0 if input_source == "Webcam Default" else 1

# --- 5. CENTERED VIDEO LAYOUT ---
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    video_placeholder = st.empty()
    prob_placeholder = st.empty() # Bar probabilitas
    
    if not st.session_state['run']:
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(black_frame, "AI SYSTEM STANDBY", (160, 240), cv2.FONT_HERSHEY_DUPLEX, 1.0, (100, 100, 100), 2)
        video_placeholder.image(black_frame, channels="RGB", use_container_width=True)

# --- 6. LOGIC LOOP ---
if st.session_state['run']:
    mp_face_mesh = load_mediapipe()
    model = load_model() # Load Model
    
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(source_index)
    COUNTER = 0

    while cap.isOpened() and st.session_state['run']:
        ret, frame = cap.read()
        if not ret:
            stop_detection()
            break

        frame = cv2.flip(frame, 1)
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w, _ = frame.shape
        
        status_text = "Active"
        color_status = (0, 255, 0)
        prediction_text = "Unknown"
        probability = 0.0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_inv  = [face_landmarks.landmark[i] for i in LEFT_EYE]
                right_inv = [face_landmarks.landmark[i] for i in RIGHT_EYE]

                left_ear  = calculate_ear(left_inv, w, h)
                right_ear = calculate_ear(right_inv, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                # --- AI PREDICTION ---
                # Input features sesuai saat training: [left, right, avg]
                features = np.array([[left_ear, right_ear, avg_ear]])
                
                # Prediksi: 0 = Melek, 1 = Ngantuk
                pred_label = model.predict(features)[0]
                pred_proba = model.predict_proba(features)[0] # [prob_0, prob_1]
                
                # Ambil probabilitas Ngantuk (index 1)
                drowsy_prob = pred_proba[1] 

                # --- LOGIC TIMER DENGAN HASIL AI ---
                if pred_label == 1: # Jika AI bilang Ngantuk
                    COUNTER += 1
                    eye_status = "Eyes Closed"
                    probability = drowsy_prob # Untuk progress bar
                    
                    if COUNTER >= DROWSY_LIMIT:
                        status_text = "Drowsiness Detected!"
                        color_status = (0, 0, 255)
                    else:
                        status_text = "Analyzing..."
                        color_status = (0, 255, 255)
                else:
                    COUNTER = 0
                    status_text = "Active"
                    color_status = (0, 255, 0)
                    probability = drowsy_prob 

                # --- VISUALISASI ---
                draw_color = color_status
                
                # Kotak Wajah
                x_min, x_max, y_min, y_max = w, 0, h, 0
                for pt in face_landmarks.landmark:
                    x, y = int(pt.x * w), int(pt.y * h)
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y
                
                pad = 20
                cv2.rectangle(frame, (max(0, x_min-pad), max(0, y_min-pad)), (min(w, x_max+pad), min(h, y_max+pad)), draw_color, 1)

                # Info Panel (Lebih Besar)
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (450, 150), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                # Tampilkan keyakinan AI
                cv2.putText(frame, f"AI Conf: {drowsy_prob*100:.1f}% Drowsy", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(frame, f"Status: {status_text}", (30, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, draw_color, 2)

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
        # Update Probability Bar di luar video
        with col2:
            if 'drowsy_prob' in locals():
                prob_placeholder.progress(float(drowsy_prob), text=f"Tingkat Kantuk: {drowsy_prob*100:.1f}%")

    cap.release()

