from uuid import uuid4, UUID
from typing import Any, Dict, Optional

# The in-memory session store
session_store: Dict[UUID, Any] = {}

def create_session(state: Any) -> UUID:
    session_id = str(uuid4())
    state['session_id'] = session_id
    session_store[session_id] = state
    return session_id

def get_session(session_id: UUID) -> Optional[Any]:
    return session_store.get(session_id)

def get_all_sessions() -> Dict[UUID, Any]:
    return session_store.copy()

def update_session(session_id: UUID, new_state: Any) -> bool:
    if session_id in session_store:
        session_store[session_id] = new_state
        return True
    return False

def delete_session(session_id: UUID) -> bool:
    if session_id in session_store:
        del session_store[session_id]
        return True
    return False
