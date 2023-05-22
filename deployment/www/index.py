import os
from pathlib import Path
from flask import Flask, render_template, jsonify, request, abort
import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient

app = Flask(__name__)

ROOT_PATH = Path(__file__).parents[2]
MODEL_PATH = ROOT_PATH / 'models' / 'final_model.h5'

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'MCC': MatthewsCorrelationCoefficient(num_classes=3, name='MCC')}
    )

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        abort(404)
    file = request.files['file']
    path = os.path.join('temps/', file.filename)
    file.save(path)
    
    label = 'Une erreur'
    class_value = get_image_label(path)
    if class_value == 0:
        label = 'une pneumonie bactérienne.'
    elif class_value == 1:
        label = 'absolument rien !'
    elif class_value == 2:
        label = 'une pneumnie virale.'

    return jsonify(
        text=label
    )
    
def image_to_tensor(path):
    image = tf.keras.utils.load_img(
        path,
        color_mode="rgb",
        target_size=(512,512)
        )
    numpy_image = np.array(image)
    
    return tf.convert_to_tensor([numpy_image])
    
def get_image_label(path):
    input = image_to_tensor(path)
    
    output = model.predict(input)
    
    return np.argmax(output, axis=1)
    