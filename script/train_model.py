import cv2
import os
import numpy as np

class FaceRecognizer:
    # def __init__(self, dataset_path='dataset/', model_path='models/face_recognizer.xml'):
    def __init__(self, dataset_path='../dataset/', model_path='../models/face_recognizer.xml'):
        self.dataset_path = dataset_path
        self.model_path = model_path
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

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

    def train_or_update_model(self, new_dataset=True):
        print("Membaca data dari folder dataset...")
        images, labels, names = self.load_images_from_folder(self.dataset_path)

        if not images or not labels:
            print("Tidak ada gambar ditemukan di dataset.")
            return

        if not new_dataset and os.path.exists(self.model_path):
            print("Menambahkan data ke model yang sudah ada...")
            self.recognizer.read(self.model_path)

            existing_images, existing_labels, existing_names = self.load_images_from_folder(self.dataset_path)

            images.extend(existing_images)
            labels.extend(existing_labels)

            for key, value in existing_names.items():
                if value not in names.values():
                    names[key] = value
        else:
            print("Membuat model baru...")

        print("Melatih model...")
        self.recognizer.train(images, np.array(labels))
        self.recognizer.write(self.model_path)
        print(f"Model disimpan ke {self.model_path}")

        names_path = self.model_path.replace('.xml', '_names.txt')
        with open(names_path, 'w') as f:
            for key, value in names.items():
                f.write(f"{key}:{value}\n")
        print(f"Daftar nama disimpan ke {names_path}")

if __name__ == "__main__":
    face_recognizer = FaceRecognizer()
    print("1. Buat dataset baru")
    print("2. Tambah ke dataset yang sudah ada")
    choice = input("Pilih opsi (1/2): ")

    if choice == '1':
        face_recognizer.train_or_update_model(new_dataset=True)
    elif choice == '2':
        face_recognizer.train_or_update_model(new_dataset=False)
    else:
        print("Pilihan tidak valid. Program dihentikan.")
