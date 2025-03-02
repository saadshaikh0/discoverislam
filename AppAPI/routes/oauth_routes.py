from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import get_db
from services.auth_service import get_google_auth_url, get_google_token, get_google_user_info
from dao import get_user_by_email, create_user

router = APIRouter(prefix="/oauth", tags=["OAuth"])

@router.get("/login")
def login():
    """Redirects users to Google OAuth login page."""
    return {"auth_url": get_google_auth_url()}

@router.get("/callback")
def callback(code: str, db: Session = Depends(get_db)):
    """Handles Google OAuth callback, fetches user info, and stores email in the database."""
    token_response = get_google_token(code)
    if "access_token" not in token_response:
        raise HTTPException(status_code=400, detail="OAuth Token Exchange Failed")

    user_info = get_google_user_info(token_response["access_token"])
    print("User INFO: ",user_info)
    if "email" not in user_info:
        raise HTTPException(status_code=400, detail="Email not found in OAuth response")

    # ✅ Store user email & name in the database
    user = get_user_by_email(db, user_info["email"])
    if not user:
        user = create_user(db, user_info["email"])

    return {
        "access_token": token_response["access_token"],
        "token_type": "Bearer",
        "user_email": user.user_email
    }
