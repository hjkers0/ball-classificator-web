import numpy as np
from flask import Flask, request, render_template
import pickle
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER= os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = pickle.load(open("../ball_classificator_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    class_names = ['American Football', 'Baseball', 'Basketball', 'Billiard Ball', 'Bowling Ball',
                'Cricket Ball', 'Football', 'Golf Ball', 'Hockey Ball', 'Hockey Put',
                'Rugby Ball', 'Shuttle Cock', 'Table Tennis Ball', 'Tennis Ball', 'Volley Ball']

    uploaded_img = request.files['uploaded-file']
    img_ball = Image.open(uploaded_img).resize((180,180))

    img_balon_array = tf.keras.utils.img_to_array(img_ball)
    img_balon_array = tf.expand_dims(img_balon_array, 0)

    prediccion = model.predict(img_balon_array)
    score = tf.nn.softmax(prediccion[0])

    return render_template(
        "index.html", 
        prediction_text= "\n Esta imagen se parece a {} con un {:.2f} porciento de confianza."
        .format(class_names[np.argmax(score)], 100 * np.max(score)),
        uploaded_img = img_ball
    )


if __name__ == "__main__":
    app.run(debug=True)