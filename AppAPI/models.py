from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "user_details"

    user_email = Column(String(255), primary_key=True, index=True)
    chat_sessions = relationship("UserChatSession", back_populates="user")

class UserChatSession(Base):
    __tablename__ = "user_chat_sessions"
    
    session_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("user_details.user_email"))
    user_session_id = Column(Integer, nullable=False)
    session_name = Column(String(255), nullable=False)

    user = relationship("User", back_populates="chat_sessions")
    queries = relationship("UserQuery", back_populates="session")

class UserQuery(Base):
    __tablename__ = "user_queries"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("user_chat_sessions.session_id"))
    user_email = Column(String(255), ForeignKey("user_details.user_email"))  # ✅ New column
    user_session_id = Column(Integer, nullable=False)  # ✅ New column
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=True)

    session = relationship("UserChatSession", back_populates="queries")
