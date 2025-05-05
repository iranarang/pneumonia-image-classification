from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model 
model = tf.keras.models.load_model('/models/best_pneumonia_model_tf215.keras')

@app.route('/summary', methods=['GET'])
def model_info():
    return jsonify({
        "version": "v1",
        "name": "pneumonia-detection-cnn",
        "description": "A CNN that classifies chest X-ray images as normal vs. pneumonia",
        "accuracy": 1.0 
    })

def preprocess_input(pil_img):
    """
    Resize, normalize, and reshape PIL image to the model's expected input shape.
    """
    img = pil_img.convert('RGB').resize((150, 150))
    arr = np.array(img)
    return arr.reshape(1, 150, 150, 3)        # add batch dimension

@app.route('/inference', methods=['POST']) # used chatGPT to fix resizing issues and print the probabilities
def classify_pneumonia():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "The `image` field is required"}), 400

    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        data = preprocess_input(pil_img)
        preds = model.predict(data)
        class_idx = np.argmax(preds, axis=1)[0]
        label = "pneumonia" if class_idx == 1 else "normal"
        return jsonify({"prediction": label, "probabilities": preds.tolist()})
    except Exception as e:
        return jsonify({
            "error": "Could not process the `image` field",
            "details": str(e)
        }), 400

@app.route('/best-hyperparameters', methods=['GET'])
def best_hyperparameters():
    return jsonify({
        "Conv blocks": 2,
        "Filters (first block)": 64,
        "Dense units": 192,
        "Dropout": 0.2,
        "Optimizer": "rmsprop",
        "Learning rate": 0.001
    })


if __name__ == '__main__':
    # Serve on all interfaces, port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
