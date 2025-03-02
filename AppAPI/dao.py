from sqlalchemy.orm import Session
from models import User, UserChatSession, UserQuery 
import schemas

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.user_email == email).first()

def create_user(db: Session, email: str):
    """Create a user if not exists."""
    user = db.query(User).filter(User.user_email == email).first()
    if not user:
        user = User(user_email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

def create_chat_session(db: Session, user_email: str, query: str):
    """
    Creates a new chat session and assigns a manual user_session_id.
    """
    user = db.query(User).filter(User.user_email == user_email).first()
    if not user:
        raise ValueError("User not found")

    # ✅ Find last user_session_id for this user
    last_session = db.query(UserChatSession)\
                     .filter(UserChatSession.user_id == user_email)\
                     .order_by(UserChatSession.user_session_id.desc())\
                     .first()

    next_user_session_id = 1 if last_session is None else last_session.user_session_id + 1

    # ✅ Create new session with first query as session name
    session = UserChatSession(user_id=user_email, user_session_id=next_user_session_id, session_name=query)
    db.add(session)
    db.commit()
    db.refresh(session)

    return session.session_id, session.user_session_id

def store_user_query(db: Session, session_id: int, user_email: str, user_session_id: int, query: str):
    user_query = UserQuery(session_id=session_id, user_email=user_email, user_session_id=user_session_id, query=query)
    db.add(user_query)
    db.commit()
    db.refresh(user_query)
    return user_query.id

def update_query_response(db: Session, query_id: int, response: str):
    query = db.query(UserQuery).filter(UserQuery.id == query_id).first()
    if query:
        query.response = response
        db.commit()
