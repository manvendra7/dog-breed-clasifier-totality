import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model

from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
    global model
    model = load_model('resenetModel.h5')
    print(" * Model loaded!")
    
def preprocess_image(image, target_size=(224,224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    x = preprocess_input(image)
    return x
    
        
print(" * Loading Keras model...")
get_model()

@app.route("/predict",methods=['POST'])
def predict():
    print(request.json)
    data = request.json
    #print('This is message',message)
    encoded = data['image']
    bencoded = bytes(encoded, 'utf-8')
    decoded =  base64.decodebytes(bencoded)
    image = Image.open(io.BytesIO(decoded))
#     image_result = open('deer_dog.jpg', 'wb')
#     image_result.write(decoded)
#     image = Image.load('deer_dog.jpg')
    processed_image = preprocess_image(image, target_size=(224, 224))
    prediction = model.predict(processed_image).tolist()
    pred_label = np.argmax(model.predict(processed_image))
    
    label_dict = {0: 'beagle',
                  1: 'chihuahua',
                  2: 'doberman',
                  3: 'french_bulldog',
                  4: 'golden_retriever',
                  5: 'malamute',
                  6: 'pug',
                  7: 'saint_bernard',
                  8: 'scottish_deerhound',
                  9: 'tibetan_mastiff'}
    
    response = {
        'prediction': {
            label_dict[pred_label]: prediction[0][pred_label],
#             'chihuahua': prediction[0][1],
#             'doberman': prediction[0][2],
#             'french_bulldog' : prediction[0][3],
#             'golden_retriever' : prediction[0][4],
#             'malamute' : prediction[0][5],
#             'pug' : prediction[0][6],
#             'saint_bernard' : prediction[0][7],
#             'scottish_deerhound' : prediction[0][8],
#             'tibetan_mastiff' : prediction[0][9]
        }
    }
    return jsonify(response)
