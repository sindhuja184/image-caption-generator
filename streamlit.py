import streamlit as st
import numpy as np
from PIL import Image
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model


with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

caption_model = load_model("best_model_epochs_30.keras", compile=False)


vgg = VGG16()
vgg = Model(inputs = vgg.input, outputs = vgg.layers[-2].output)


max_length = 34

def get_features(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)
    features = vgg.predict(image, verbose= 0)
    return features

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

     
        image_input = image.reshape((1, 4096))

        yhat = model.predict([image_input, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat)
        if word is None:
            break

        in_text += ' ' + word
        if word == 'endseq':
            break

    return in_text.replace('startseq', '').replace('endseq', '').strip()
    


st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Upload an Image", type = ['jpg'])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = 'Uploaded image')

    if st.button("Generate Caption"):
        with st.spinner('Generating Caption'):
            features = get_features(image)
            caption = predict_caption(caption_model, features, tokenizer, max_length)
        st.markdown(f"**Predicted Caption:** {caption}")
        

