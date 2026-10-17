from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def get_token(username, password):
    r = client.post("/login", data={"username": username, "password": password})
    assert r.status_code == 200
    return r.json()["access_token"]


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_login_wrong_password():
    r = client.post("/login", data={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_predict_needs_token():
    r = client.post("/predict", json={"hours_studied": 8, "attendance": 90})
    assert r.status_code == 401


def test_admin_train_and_predict():
    token = get_token("admin", "admin123")
    r = client.post("/train", headers=auth(token))
    assert r.status_code == 200
    r = client.post(
        "/predict",
        json={"hours_studied": 8, "attendance": 90},
        headers=auth(token),
    )
    assert r.status_code == 200
    assert r.json()["passed"] is True
    
    
def test_user_cannot_train():
    token = get_token("user", "user123")
    r = client.post("/train", headers=auth(token))
    assert r.status_code == 403