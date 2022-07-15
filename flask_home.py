from flask import Flask, render_template, request, url_for, redirect, flash
import os
from werkzeug.utils import secure_filename
import predictions
from PIL import Image



app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST', 'GET'])
def upload_image():
    if request.method == 'POST':
        image = request.files['input_image']
        if 'input_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        if image.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if image:
            image_name = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))
            image_prediction = predictions.make_prediction(image_name)
            return render_template('prediction.html', uploaded_image=image_name, predicted_output=image_prediction)   
        image_prediction = predictions.make_prediction(image_name)
    return render_template('prediction.html')

@app.route('/display/<filename>')
def display_image(filename=''):
    return redirect(url_for('static',filename='uploads/'+ filename))

if __name__ == '__main__':
    app.run(debug=True)