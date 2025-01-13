from flask import Flask
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.create_dataset import create
from api.recognize import recognize

app = Flask(__name__)

# Endpoint untuk mengenali wajah
@app.route('/recognize', methods=['POST'])
def recognize_face():
    return recognize()

@app.route('/add-dataset', methods=['POST'])
def add_dataset():
    return create()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
