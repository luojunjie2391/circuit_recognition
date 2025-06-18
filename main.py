from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from img_prec import img_recognition
app = Flask(__name__)
CORS(app)


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = img_recognition(img)
    return result


if __name__ == '__main__':
    app.run(debug=True, port=8000)
