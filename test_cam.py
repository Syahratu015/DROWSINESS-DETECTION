import cv2

# Inisialisasi webcam (biasanya indeks 0 atau 1)
cap = cv2.VideoCapture(0)

# Cek apakah kamera berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak dapat mengakses kamera.")
    exit()

print("Kamera berhasil dibuka! Tekan tombol 'q' di keyboard untuk keluar.")

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    
    # Jika frame tidak terbaca, berhenti
    if not ret:
        print("Gagal membaca frame.")
        break
    
    # Flip frame secara horizontal (1) agar gerakan kanan-kiri sesuai
    frame = cv2.flip(frame, 1)

    # Tampilkan frame di window bernama 'Test Kamera'
    cv2.imshow('Test Kamera', frame)

    # Tunggu input tombol selama 1ms. Jika 'q' ditekan, break loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua window
cap.release()
cv2.destroyAllWindows()
