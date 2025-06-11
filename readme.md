# Image Caption Generator
An AI-powered web app that generates descriptive captions for uploaded images using deep learning and natural language processing. Built with Streamlit, VGG16 for feature extraction, and a trained caption generation model.

## Model Overview

- Feature Extractor: Pre-trained VGG16 model (excluding final classification layer)

- Caption Generator: Trained on Flickr8k dataset using a sequence model (e.g., LSTM + embedding)

- Tokenizer: Used to convert words to sequences for generation


## Demo
Check out the demo [here](https://drive.google.com/file/d/1DWIbuS3GSIcodTu2DXhBE2HwRSw61iSi/view?usp=sharing)

## SetUp Instrauctions
1. Clone the repo 
```bash 
git clone https://github.com/sindhuja184/image-caption-generator.git
cd image-caption-generator
```

2. Create and activate virtual environment
```bash 
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```
3. Install depedencies
```bash
pip install -r requirements.txt
```
4. Run streamlit App
```bash
streamlit run streamlit_app.py
```


## Model Architecture
- Feature Extractor: Pre-trained VGG16 (without top layer)

- Caption Generator: LSTM-based decoder

- Training Dataset: Flickr8k

- Tokenization: Keras tokenizer saved as tokenizer.pkl

## How It Works
- The uploaded image is resized to 224x224 and preprocessed using VGG16.

- VGG16 extracts a 4096-dimensional feature vector.

- This vector is passed to the LSTM-based decoder along with the tokenizer to predict a caption.

- The result is shown in the Streamlit interface.