from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain_openai import AzureChatOpenAI

from models.userstate import State
from models.chatmodel import ChatModel
from session_store import create_session, get_session, update_session, delete_session, get_all_sessions
from utils.executor import execute_node
from routes.nvidiaa2f import a2f_router

app = FastAPI()


app.include_router(a2f_router, tags=["a2f"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    model=os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini"),
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    temperature=0.9,
)

@app.get("/session/")
async def init_session():
    state = State(
        name=None,
        service_type=None,
        check_in={},
        ticket_booking={},
        amount=0,
        history=[],
        retry_count=0
    )
    state = execute_node("collect_name", llm, state)
    session_id = create_session(state)
    return {"session_id": str(session_id), "state": state}

@app.post("/chat/")
# async def chat(session_id: str, user_input: str):
async def chat(ChatModel: ChatModel):
    session_id = ChatModel["session_id"]
    user_input = ChatModel["user_input"]
    session = get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    state = session

    current_node = state.get("current_node", "collect_name")
    state = execute_node(current_node, llm, state, user_input)
    next_node = state.get("current_node")
    state = execute_node(next_node, llm, state)
    update_session(session_id, state)
    return {"session_id": session_id, "state": state}

@app.get("/get_all_sessions/")
async def get_all_sessins():
    sessions = get_all_sessions()
    return sessions

