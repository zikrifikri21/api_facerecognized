import cv2
import os

class CaptureDataset:
    def __init__(self):
        pass

    def capture(self):
        # Menyiapkan folder dataset
        dataset_folder = os.path.join(os.path.dirname(__file__), '../dataset/')
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        # Mengambil nama orang
        person_name = input("Masukkan nama orang: ").strip()
        person_folder = os.path.join(dataset_folder, person_name)

        # Membuat folder untuk orang jika belum ada
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        # Get the current image count in the folder
        existing_images = [f for f in os.listdir(person_folder) if f.endswith('.jpg')]
        img_count = len(existing_images) + 1

        # Inisialisasi kamera
        camera = cv2.VideoCapture(1)
        face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        print("Mulai mengambil gambar wajah, tekan 'q' untuk berhenti.")

        new_images_count = 0

        while True:
            _, frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_ref.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                if new_images_count >= 20:
                    break
                face = gray[y:y + h, x:x + w]
                img_path = os.path.join(person_folder, f"img{img_count}.jpg")
                cv2.imwrite(img_path, face)
                img_count += 1
                new_images_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q") or new_images_count >= 20:
                break

        camera.release()
        cv2.destroyAllWindows()
        print(f"Dataset untuk {person_name} telah disimpan di {person_folder}")
        
if __name__ == "__main__":
    CaptureDataset().capture()
