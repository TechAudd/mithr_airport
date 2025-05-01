from typing_extensions import TypedDict


class State(TypedDict):
    name: str
    service_type: str
    check_in: dict
    ticket_booking: dict
    history: list[dict]
    retry_count: int
    current_node: str
