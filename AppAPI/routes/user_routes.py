from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import OAuth2AuthorizationCodeBearer
from sqlalchemy.orm import Session
from database import get_db
import dao, schemas
from services.auth_service import get_google_user_info

router = APIRouter(prefix="/user", tags=["User Query"])

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/auth",  # ✅ Required
    tokenUrl="https://oauth2.googleapis.com/token"  # ✅ Required
)

@router.post("/query")
async def receive_user_query(
    request: schemas.QueryRequest,
    db: Session = Depends(get_db),
    token: str = Security(oauth2_scheme)  # 👈 Require OAuth Token
):
    # ✅ Step 1: Validate OAuth Token with Google
    user_info = get_google_user_info(token)

    # ✅ Step 2: Ensure Google Returns Email
    if "email" not in user_info:
        raise HTTPException(status_code=401, detail="Invalid token or user info not found")

    # ✅ Step 4: Check if User Exists in DB
    user = dao.get_user_by_email(db, user_info["email"])

    if user_info["email"] != request.user_email:
        raise HTTPException(status_code=404, detail="User not found")

    # ✅ Step 4: Create a Chat Session with a Session Name (First Query)
    session_id, user_session_id = dao.create_chat_session(db, user.user_email, request.query)

    query_id = dao.store_user_query(db, session_id, user.user_email, user_session_id, request.query)

    return {
        "session_id": session_id,
        "user_session_id": user_session_id,
        "query_id": query_id,
        "message": request.query
    }