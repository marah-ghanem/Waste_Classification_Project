import os
import numpy as np
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from flask import Flask, request, render_template
from keras_preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from flask import Flask, request, render_template, send_file
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation


app = Flask(__name__)

def create_model():
    new_height = 224
    new_width=224

    VGG16_model = VGG16(input_shape=(new_height,new_width,3),include_top=False,weights="imagenet")
    for layer in VGG16_model.layers:
        layer.trainable=False

    model=Sequential()
    model.add(VGG16_model)
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(BatchNormalization())
    model.add(Dense(1024,kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1,activation='sigmoid'))
    model.load_weights('lastw.h5')

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',metrics=[AUC(name='auc')],optimizer=optimizer)

    return model

app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained model
model = create_model()

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    uploaded_file = None
    prediction = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return render_template('draft.html')

        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            uploaded_file = file.filename

            # Classify the uploaded image

            img = load_img(filename, target_size=(224,224))
            img = img_to_array(img)
            img = img / 255
            img = np.expand_dims(img,axis=0)

            def predict_prob(number):
                return [number[0],1-number[0]]
            
            my_model = create_model()
            answer = np.array(list(map(predict_prob, my_model.predict(img))))
            if answer[0][0] > 0.5:
                prediction="Recycle waste"
            else:
                prediction="Organic waste"

    return render_template('index.html', uploaded_file=uploaded_file, prediction=prediction)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)


@app.route('/index')
def index():
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(port=3000,debug=True)
