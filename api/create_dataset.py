from flask import request, jsonify
from script.upload_from_folder import FaceDatasetManager
import os
import shutil

def get_incremented_filename(directory, filename):
    base, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}{counter}{extension}"
        counter += 1
    return new_filename

def create():
    person_name = request.form.get('name')
    images = request.files.getlist('images')

    if not person_name or not images:
        return jsonify({'error': 'Name and images are required'}), 400

    manager = FaceDatasetManager(dataset_path='/app/dataset/')
    source_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '/app/')), 'dataset/temp_upload')
    os.makedirs(source_folder, exist_ok=True)

    feedback = []
    for image in images:
        new_filename = get_incremented_filename(source_folder, image.filename)
        image_path = os.path.join(source_folder, new_filename)
        image.save(image_path)
        result = manager.insert_image(image_path, person_name)
        feedback.append(result)

    shutil.rmtree(source_folder)

    manager.train_new_dataset()

    return jsonify({'message': f'Dataset for {person_name} added and model updated successfully!', 'feedback': feedback})

if __name__ == '__main__':
    create()
