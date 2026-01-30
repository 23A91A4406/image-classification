# Image Classification System using Transfer Learning

## Project Overview
This project implements an end-to-end image classification pipeline using transfer learning. A pre-trained ResNet50 model is fine-tuned on a custom multi-class image dataset and deployed as a containerized REST API. The project follows standard MLOps practices, covering data preprocessing, model training, evaluation, and production-ready deployment using Docker.


## Objective
- Build an image classification system using transfer learning
- Apply data preprocessing and augmentation
- Fine-tune a pre-trained CNN model (ResNet50)
- Evaluate the trained model using standard metrics
- Deploy the model as a REST API using FastAPI and Docker.


## Model & Dataset
- **Model**: ResNet50 (pre-trained on ImageNet)
- **Technique**: Transfer Learning
- **Dataset**: Caltech-101
- **Number of Classes**: 10
- **Framework**: PyTorch


## Project Structure
image-classification/
│
├── data/
│ ├── train/
│ └── val/
│
├── model/
│ └── image_classifier.pth
│
├── results/
│ └── metrics.json
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│ ├── api.py
│ └── config.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md


## Pipeline Steps

### Data Preprocessing
- Automatically downloads the Caltech-101 dataset
- Selects 10 image classes
- Splits data into training (80%) and validation (20%)
- Organizes data into PyTorch-compatible directory structure

### Data Augmentation
Applied during training:
- RandomResizedCrop
- RandomHorizontalFlip
- RandomRotation


### Model Training
- Loads a pre-trained ResNet50 model
- Freezes convolutional layers
- Replaces final classification layer for 10 classes
- Trains the model on the custom dataset
- Saves trained model to `model/image_classifier.pth`


### Model Evaluation
The evaluation script computes:
- Accuracy
- Weighted Precision
- Weighted Recall
- Confusion Matrix

#### Results are saved in:
results/metrics.json


### REST API
The trained model is deployed using FastAPI.

#### Available Endpoints:
- GET /health → Health check endpoint
- POST /predict → Image classification endpoint

#### Sample Response:
{
  "predicted_class": "airplanes",
  "confidence": 0.99
}

### Docker & Deployment
The application is fully containerized using Docker and managed with Docker Compose.

### Environment Variables (.env.example)
API_PORT=8000
MODEL_PATH=model/image_classifier.pth

### Run the Application
docker-compose up --build

### The API will be available at:
http://localhost:8001

## Dependencies
All required dependencies are listed in requirements.txt.

## Conclusion
This project demonstrates a complete workflow for building, evaluating, and deploying a deep learning image classification model using transfer learning and Docker. It reflects real-world MLOps practices used in production environments.