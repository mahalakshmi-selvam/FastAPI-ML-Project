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
## Week 3 Progress
- Restructured project into models/, routes/, utils/ folders
- Added schemas.py with pydantic request/response models
- Built POST /train, POST /predict and GET /evaluate endpoints
- Added train/test split and metrics (accuracy, precision, recall, F1)
- Added error handling and logging
## Live Demo
https://fastapi-ml-project-hxnj.onrender.com/docs

## Week 4 Progress
- Added CSV dataset upload endpoint (POST /upload)
- Added hyperparameter tuning with GridSearchCV
- Added JWT login with admin/user roles and role-based access
- Added rate limiting and IP blocking
- Added Prometheus metrics at /metrics (can be visualized with Grafana)
- Added automated tests (pytest) and GitHub Actions
- Deployed on Render with automatic HTTPS and auto-deploy on push
## Week 5 Progress
- Project structure with separate models/, routes/, utils/ folders
- Prediction endpoint with pydantic validation and model loading
- OAuth2 password flow with JWT tokens
- Role-based access control: admin can train/upload, users can predict/evaluate
- HTTPS enabled through Render deployment
- Tested authentication and authorization using Swagger UI and pytest