# Plant Disease Classification using Hybrid CNN-Vision Transformer

## Overview
This project implements a hybrid CNN-Vision Transformer model for plant disease classification using data scraped from iNaturalist website.

## Data Collection
The training data is scraped from the iNaturalist website using the provided scraping tool.

**Steps:**
- Use the `get_data.py` file located in the `dataset/` directory
- You can modify the list of diseases you want to work with based on image availability on iNaturalist
- The script will automatically download and organize the plant disease images

## Model Architecture
The model combines CNN feature extraction with Vision Transformer attention mechanisms:

- **CNN Backbone**: Extracts local features from plant images (3 convolutional layers)
- **Patch Embedding**: Converts CNN feature maps into sequence format for transformer processing
- **Vision Transformer**: Processes image patches using multi-head self-attention mechanism
- **Classification Head**: Final layer that predicts the specific disease type

The model has approximately 27.7M parameters and processes 384x384 pixel images.

## Installation
First, install pixi package manager:
```bash
curl -fsSL https://pixi.sh/install.sh | bash 
# Install project dependencies
pixi install
```
## Configuration
Before training, configure your model settings in the config.py file:

Number of disease classes
Batch size and learning rate
Model architecture parameters (embedding dimensions, number of layers, etc.)
Training epochs and other hyperparameters

## Training
To train the model with your configuration:
```bash
python train_pipeline.py
```
The training script will:

- Load your configured settings
- Preprocess the scraped data
- Train the hybrid CNN-ViT model
- Save checkpoints and training results

## Usage Summary

- Run python dataset/get_data.py to scrape plant disease data from iNaturalist
- Edit disease list in the scraping script based on available images
- Configure model parameters in config.py
- Start training with python train_pipeline.py


