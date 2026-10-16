# FastAPI-ML-Project
Machine Learning API using FastAPI
## API Endpoints

### GET /
Response: `{"message": "My ML API is running!"}`

### GET /health
Checks if the API is running.
Response: `{"status": "ok"}`

### POST /predict
Request: `{"text": "hello"}`
Response: `{"input": "hello", "prediction": "placeholder"}`

## How to Run
```
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs to test.
## Week 1 Progress
- Set up Python, VS Code, Git and GitHub
- Created virtual environment and installed FastAPI, uvicorn, pydantic
- Created project structure (models/, data/, requirements.txt)
- Built GET /, GET /health and POST /predict endpoints
- Tested endpoints using Swagger UI and PowerShell
- Documented the API in README
## Week 2 Progress
- Added scikit-learn model (LogisticRegression) in models.py
- Added /train endpoint using pandas to load data/sample.csv
- Added /predict endpoint with pydantic validation and numpy
- Added error handling and logging for training and prediction
- Tested endpoints using Swagger UI