# ML web application


Predictive model with fastapi that accepts a .csv dataset.
Tools: Tensorflow
     - Keras
     - Numpy
     - Pandas
     - FastApi
     - scikit-learn
     
Accuracy predict MLP 70%       

Tests, deployment, MLP in progress

## Requirmentes

- Python 3.9+
- FastAPI
- Uvicorn
- Docker

## Installation

1. Repository cloning: `git clone https://github.com/Hallexz/Predict-models.git`
2. Go to the project directory: `cd Predict-models`
3. Installing dependencies: `pip install -r requirements.txt`

## Запуск приложения

1. Starting the server: `uvicorn src.web_app:app --reload`
2. Open in browser: `uvicorn app.app:app --host 0.0.0.0 --port 8080`

Docker:
1. `docker build -t ml-app .`
2. `docker run -p 80:80 ml-app`
