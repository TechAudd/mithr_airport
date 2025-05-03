from pprint import pprint as pp
from langgraph.graph import END
import json

with open("conf/mock.json", "r") as file:
    MOCK_DATA = json.load(file)


def collect_name_router(state):
    if state.get("collect_name_result") is None:
        return "collect_name"
    return "service_choice"


def service_choice_router(state):
    if state.get("service_choice_result") is None:
        return "service_choice"
    service = state.get("service_type")
    if service == "check_in":
        return "check_in_booking_node"
    elif service == "book_ticket":
        return "book_ticket_node"
    elif service == "general_query":
        return "general_query_node"
    else:
        print("Bot: Please specify with what do you need assistance with.")
        return "service_choice"


def check_in_booking_router(state):
    booking_details = state.get("check_in", {}).get("booking_details")
    if not booking_details or any(value is None for value in booking_details.values()):
        return "check_in_booking_node"
    MOCK = MOCK_DATA['passengers']
    passenger_details = MOCK.get(booking_details.get("ticket_no"), {})
    if passenger_details:
        state["check_in"]["passenger_details"] = passenger_details
    else:
        return END
    return "check_in_passport_node"


def check_in_passport_router(state):
    passport_details = state.get("check_in", {}).get("passport_details")
    if not passport_details or any(value is None for value in passport_details.values()):
        return "check_in_passport_node"
    return "seat_preference_node"


def seat_preference_router(state):
    seat_preference = state.get("check_in", {}).get("passenger_details", {}).get("seat_no")
    if seat_preference:
        return "luggage_checkin_node"
    return "seat_preference_node"
