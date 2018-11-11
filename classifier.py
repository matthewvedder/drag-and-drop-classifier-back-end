from flask import Flask
from flask import request
from flask_cors import CORS
from flask import jsonify
from fastai import *
from fastai.vision import *
from io import BytesIO

app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

@app.route("/")

@app.route('/classifier', methods=['POST', 'GET'])
def classifier():
    if request.method == 'POST':
        pred = predict(request.data)
        return jsonify({ 'prediction': pred })
    else:
        return 'classifier'

def predict(img_data):
    classes = ['evans', 'hemsworth', 'pine', 'pratt']
    data = ImageDataBunch.single_from_classes('./', classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(data, models.resnet34)
    learn.load('stage-3')
    # img = Image.frombytes('RGBA', (224,224), img_data)
    img = open_image(BytesIO(img_data))
    pred_class, pred_idx, outputs = learn.predict(img)
    return pred_class
