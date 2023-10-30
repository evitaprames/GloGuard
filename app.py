import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model = tf.keras.models.load_model('densenet201.h5')

labels = ['Seborrheic Keratosis', 'Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Melanoma', 'Vascular Lesion']

def model_predict(img_path, model):
    
    # Read the image
    img = image.load_img(img_path, target_size=(75, 75))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.densenet.preprocess_input(img)

    # Predict the class of the image
    prediction = model.predict(img)
    prediction = np.argmax(prediction, axis=1)[0]
    label = labels[prediction]

    return label


@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('home.html')

@app.route('/tools')
def tools():
    # Main page
    return render_template('index.html')

@app.route('/about')
def about():
    # Main page
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=False)
