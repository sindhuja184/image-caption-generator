# Image Caption Generator
An AI-powered web app that generates descriptive captions for uploaded images using deep learning and natural language processing. Built with Streamlit, VGG16 for feature extraction, and a trained caption generation model.

## Model Overview

- Feature Extractor: Pre-trained VGG16 model (excluding final classification layer)

- Caption Generator: Trained on Flickr8k dataset using a sequence model (e.g., LSTM + embedding)

- Tokenizer: Used to convert words to sequences for generation


## Demo
Check out the demo [here](https://drive.google.com/file/d/1DWIbuS3GSIcodTu2DXhBE2HwRSw61iSi/view?usp=sharing)

## SetUp Instructions
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

## Check Out BLEU Scores for various epochs

1. Epochs = 15
![Epochs 15](<WhatsApp Image 2025-06-10 at 23.36.17_b2f868d4.jpg>)

2. Epochs = 20
![alt text](<WhatsApp Image 2025-06-10 at 23.52.32_931903be.jpg>)

3. Epochs = 30
![](<WhatsApp Image 2025-06-11 at 00.15.20_90a1a20c.jpg>)

The model used is the epoch 30 model.

## Challenges Faced

- Feature Extraction: Choosing the right CNN architecture and extracting meaningful features without losing spatial context.

- Sequence Modeling: Training the LSTM model to generate fluent and contextually relevant captions word-by-word.
 
- Evaluation Metrics: Implementing and interpreting BLEU scores to effectively evaluate caption quality.

- Multimodal Fusion: Successfully integrating visual features with textual sequence generation.

- Data Preprocessing: Cleaning and tokenizing captions consistently, managing padding and sequence lengths.

- Large File Handling: Managing large .h5, .pkl files and avoiding Git push issues (handled via Git LFS).

## Scope for improvement
- Integrate Transformer Models: Replace LSTM with Transformer-based architectures for better context handling.

- Add Attention Mechanism: Use attention layers to help the model focus on relevant parts of the image.

- Multilingual Captioning: Expand the system to generate captions in multiple languages.

- Enhanced Evaluation Metrics: Include CIDEr and METEOR alongside BLEU for more accurate caption assessment.
 