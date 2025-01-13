from flask import request, jsonify
import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Path model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/face_recognizer.xml')
# Load model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def recognize():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    # Simpan gambar sementara
    image_path = 'temp.jpg'
    file.save(image_path)

    # Proses gambar
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Muat nama dari file
    names = {}
    names_file_path = os.path.join(os.path.dirname(__file__), '../models/face_recognizer_names.txt')
    with open(names_file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(':')
            names[int(key)] = value

    results = []
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_region)
        name = names.get(label, "Tidak Dikenal")
        accuracy = 100 - confidence
        if confidence > 100:
            results.append({'name': "Tidak Dikenal", 'label': -1, 'confidence': confidence, 'accuracy': 0})
        else:
            results.append({'name': name, 'label': label, 'confidence': confidence, 'accuracy': accuracy})


    os.remove(image_path)
    return jsonify({'faces': results})

if __name__ == '__main__':
    recognize()