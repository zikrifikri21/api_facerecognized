import os
import shutil
import cv2
import numpy as np

class FaceDatasetManager:
    def __init__(self, dataset_path='app/dataset/', model_path='models/face_recognizer.xml'):
        self.dataset_path = dataset_path
        self.model_path = model_path
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def copy_images_to_dataset(self, source_folder, person_name):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        target_folder = os.path.join(self.dataset_path, person_name)
        os.makedirs(target_folder, exist_ok=True)

        for file_name in os.listdir(source_folder):
            source_path = os.path.join(source_folder, file_name)
            target_path = os.path.join(target_folder, file_name)

            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = cv2.imread(source_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            face = gray[y:y+h, x:x+w]
                            cv2.imwrite(target_path, face)
                            print(f"Wajah disalin: {file_name} ke {target_folder}")
                            break
                    else:
                        print(f"Tidak ada wajah pada {file_name}, dilewati.")
                except Exception as e:
                    print(f"Gagal memproses {file_name}: {e}")

        print(f"Semua gambar yang valid telah disalin ke {target_folder}.")
        return target_folder

    def insert_image(self, image_path, person_name):
        print(f"Memproses gambar: {image_path}")

        # Load the face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Create the target folder if it does not exist
        target_folder = os.path.join(self.dataset_path, person_name)
        os.makedirs(target_folder, exist_ok=True)

        # Determine the next image number
        existing_files = [
            f for f in os.listdir(target_folder)
            if f.startswith('img') and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if existing_files:
            # Extract numbers from file names and find the next number
            existing_numbers = []
            for f in existing_files:
                try:
                    existing_numbers.append(int(f.split('_')[1].split('.')[0]))
                except (IndexError, ValueError):
                    continue
            next_number = max(existing_numbers) + 1 if existing_numbers else 1
        else:
            next_number = 1

        # Get the file extension of the image
        file_extension = os.path.splitext(image_path)[1]
        target_path = os.path.join(target_folder, f'img_{next_number}{file_extension}')

        if file_extension.lower() in ['.png', '.jpg', '.jpeg']:
            try:
                # Read the image and convert it to grayscale
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        # Extract and save the detected face
                        face = gray[y:y+h, x:x+w]
                        cv2.imwrite(target_path, face)
                        print(f"Wajah disalin: {os.path.basename(target_path)} ke {target_folder}")
                        return {
                            'status': 'success',
                            'message': f"Wajah disalin: {os.path.basename(target_path)} ke {target_folder}"
                        }
                else:
                    print(f"Tidak ada wajah pada {os.path.basename(image_path)}, dilewati.")
                    return {
                        'status': 'failed',
                        'message': f"Tidak ada wajah pada {os.path.basename(image_path)}, dilewati."
                    }
            except Exception as e:
                print(f"Gagal memproses {os.path.basename(image_path)}: {e}")
                return {
                    'status': 'error',
                    'message': f"Gagal memproses {os.path.basename(image_path)}: {e}"
                }

        return {
            'status': 'failed',
            'message': f"File {os.path.basename(image_path)} bukan gambar yang valid."
        }
    def train_new_dataset(self):
        print("Membaca data dari folder dataset...")
        images, labels, names = self.load_images_from_folder(self.dataset_path)

        if not images or not labels:
            print("Dataset kosong atau tidak valid.")
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("Melatih model baru...")
        recognizer.train(images, np.array(labels))
        recognizer.write(self.model_path)
        print(f"Model baru telah disimpan ke {self.model_path}")

        names_path = self.model_path.replace('.xml', '_names.txt')
        with open(names_path, 'w') as f:
            for key, value in names.items():
                f.write(f"{key}:{value}\n")
        print(f"Daftar nama disimpan ke {names_path}")

    def load_images_from_folder(self, folder):
        images = []
        labels = []
        names = {}
        current_id = 0

        for person_name in os.listdir(folder):
            person_path = os.path.join(folder, person_name)
            if os.path.isdir(person_path):
                print(f"Memproses folder: {person_name}")
                if person_name not in names.values():
                    names[current_id] = person_name
                    current_id += 1

                label = list(names.keys())[list(names.values()).index(person_name)]

                for file_name in os.listdir(person_path):
                    file_path = os.path.join(person_path, file_name)
                    try:
                        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            images.append(img)
                            labels.append(label)
                    except Exception as e:
                        print(f"Error membaca file {file_name}: {e}")

        return images, labels, names

if __name__ == "__main__":
    manager = FaceDatasetManager()
    print("=== Upload Gambar dari Folder untuk Dataset Baru ===")
    source_folder = input("Masukkan path folder gambar: ")
    person_name = input("Masukkan nama orang untuk dataset baru: ")

    if not os.path.exists(source_folder):
        print("Folder sumber tidak ditemukan.")
        exit()

    target_folder = manager.copy_images_to_dataset(source_folder, person_name)
    print(f"Dataset baru untuk {person_name} telah dibuat di {target_folder}.")

    print("Apakah Anda ingin melatih model baru sekarang? (y/n)")
    choice = input("> ").lower()

    if choice == 'y':
        manager.train_new_dataset()
    else:
        print("Proses selesai. Anda dapat melatih model nanti.")
