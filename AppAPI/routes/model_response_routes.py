from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
import os
import sys
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import get_db
from dao import update_query_response

router = APIRouter(prefix="/modelresponse", tags=["Model Response"])

# # ✅ Predefined response for testing
# PREDEFINED_RESPONSE = """
# ### **Zakat Eligibility Criteria (Based on Islamic Principles)**  

# Zakat is an obligatory charitable contribution in Islam, typically 2.5% of a Muslim’s qualifying wealth, given to specific beneficiaries as outlined in the **Quran (Surah At-Tawbah 9:60)**. The eight eligible categories for Zakat are:

# 1. **The Poor (Al-Fuqara)** – Those who have little wealth and struggle to meet basic needs.
# 2. **The Needy (Al-Masakin)** – Those in extreme poverty, unable to support themselves.
# 3. **Zakat Collectors (Al-Amilina ‘Alayha)** – Individuals appointed to collect and distribute Zakat.
# 4. **New Muslims (Al-Mu’allafatu Qulubuhum)** – Those whose hearts need strengthening in faith, including new converts.
# 5. **Those in Bondage (Fir-Riqab)** – Slaves and captives needing freedom.
# 6. **Debtors (Al-Gharimin)** – Those burdened with debts they cannot repay.
# 7. **In the Cause of Allah (Fi Sabilillah)** – Those striving for the cause of Islam, including religious education and defense.
# 8. **Wayfarers (Ibnus-Sabil)** – Stranded travelers in need of financial assistance.

# ### **Key References**:
# - **Quran, Surah At-Tawbah (9:60)** – Specifies Zakat recipients.
# - **Sahih Muslim 987** – Emphasizes the obligation of Zakat and its proper distribution.

# Would you like help calculating Zakat for your wealth?
# """

@router.websocket("/stream/{query_id}")
async def stream_response(websocket: WebSocket, query_id: int, db: Session = Depends(get_db)):
    await websocket.accept()

    response_text = ""
    
    # ✅ Fetch query details
    query = db.query(UserQuery).filter(UserQuery.id == query_id).first()
    if not query:
        await websocket.send_text("Query not found.")
        await websocket.close()
        return

    user_email = query.user_email  # ✅ Retrieve user email
    user_session_id = query.user_session_id  # ✅ Retrieve session ID
    print(query.query)
    
    try:
        # ✅ Call LLM and Stream Response in Real-Time from generate_response_stream()
        async for chunk in generate_response_stream(query.query):
            await websocket.send_text(chunk)  # ✅ Send chunk immediately
            response_text += chunk

        # ✅ Save full response in DB after streaming completes
        update_query_response(db, query_id, response_text.strip())

        await websocket.send_text("Response Completed ✅")  # Notify frontend
    except WebSocketDisconnect:
        print(f"Client disconnected while streaming response for query {query_id}")
