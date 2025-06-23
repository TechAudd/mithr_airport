from typing_extensions import TypedDict


class ChatModel(TypedDict):
    session_id: str
    user_input: str