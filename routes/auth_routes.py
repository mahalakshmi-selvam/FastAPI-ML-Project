from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from utils.auth import authenticate_user, create_access_token

router = APIRouter(tags=["Auth"])


@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(form.username, user["role"])
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}