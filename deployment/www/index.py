from flask import Flask, render_template, jsonify, request, abort
import os

app = Flask(__name__)


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
    return jsonify(
        type=""
    )