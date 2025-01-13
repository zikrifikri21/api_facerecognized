import cv2
import os

# Muat model yang sudah dilatih
model_path = os.path.join(os.path.dirname(__file__), '../models/face_recognizer.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Muat classifier wajah
cascade_path = os.path.join(os.path.dirname(__file__), '../haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

# Direktori dataset
dataset_path = os.path.join(os.path.dirname(__file__), '../dataset/')

# Membaca folder dan mengambil nama orang (folder) yang ada dalam dataset
names = {}
for i, person_name in enumerate(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        names[i] = person_name  # ID (indeks) dan nama orang

# Akses kamera
camera = cv2.VideoCapture(1)

def classify_face():
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Tidak dapat membaca frame dari kamera.")
            break

        # Ubah gambar menjadi grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_region)

            # Tampilkan ID dan Nama yang terdeteksi jika confidence di bawah threshold
            if confidence < 100:  # Lower confidence value means higher confidence in prediction
                name = names.get(label, "Tidak Dikenali")
            else:
                name = "Tidak Dikenali"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(frame, f"{name} - ID: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Tampilkan frame
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    classify_face()
