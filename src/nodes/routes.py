from pprint import pprint as pp

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
    return "check_in_visual_node"

def check_in_visual_router(state):
    aadhar_details = state.get("check_in", {}).get("aadhar_details")
    if not aadhar_details or any(value is None for value in aadhar_details.values()):
        return "check_in_visual_node"
    pp(state)
    return "general_query_node"

