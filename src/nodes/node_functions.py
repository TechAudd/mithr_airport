from langchain_core.messages import AIMessage, HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from pprint import pprint
import json
import sys
import pdb

from utils.vlm_extraction import extract_details_with_vllm
from utils.helper import ask_llm_for_question, extract_field_and_refusal_with_json
from utils.tts import botspeak
from models.userstate import State

with open("conf/fields.json", "r") as file:
    FIELDS = json.load(file)

with open("conf/mock.json", "r") as file:
    MOCK_DATA = json.load(file)

with open("conf/output.json", "r") as file:
    FLIGHT_DATA = json.load(file)

def collect_field(llm, state, field, options=None, greeting=False, node=None, retry_count=None, optional=False):
    if type(state) is not dict:
        state = state[0]
    if retry_count is None:
        retry_count = state.get("retry_count", 0)
    history = state.get("history", [])
    field_desc = FIELDS[field]["description"]
    field_desc += f" from the following options: {str(options)}" if options else ""
    question = ask_llm_for_question(llm, field, field_desc, state, retry_count, greeting)
    botspeak(question)
    user_input = input("You: ")
    history.append(AIMessage(content=question))
    history.append(HumanMessage(content=user_input))
    value, refused = extract_field_and_refusal_with_json(llm, field, field_desc, user_input)
    if refused or not value or (value == "None" and not optional):
        retry_count = state.get("retry_count", 0) + 1
        if retry_count == 2:
            botspeak("This information is required to proceed further and chat will terminate if not provided. Could you please provide it?")
            history.append(AIMessage(content="This information is required to proceed. Could you please provide it?"))
        elif retry_count > 2:
            botspeak("Exiting the chat as the information is required to proceed further.")
            sys.exit(0)
        return {**state, "retry_count": retry_count, "history": history}, None
    if node:
        state[node][field] = value
        return {**state, "retry_count": 0, "history": history}, value
    else:
        return {**state, field: value, "retry_count": 0, "history": history}, value


def collect_field_visual(llm, state, node, field):
    if type(state) is not dict:
        state = state[0]
    retry_count = state.get("retry_count", 0)
    history = state.get("history", [])
    field_desc = FIELDS[field]["description"]
    question = ask_llm_for_question(llm, field, field_desc, state, retry_count)
    botspeak(question)
    image_path = input("Please provide the path to the image: ")
    history.append(AIMessage(content=question))
    details = extract_details_with_vllm(image_path, data_format=FIELDS[field]["data_format"])
    history.append(HumanMessage(content=str(details)))
    history.append(HumanMessage(content=f"Here are my {field} details converted to json."))
    if not details or any(value is None for value in details.values()):
        retry_count = state.get("retry_count", 0) + 1
        if retry_count == 2:
            bot_dialog = "This information is required to proceed further. Could you please recapture the image properly?"
            botspeak(bot_dialog)
            history.append(AIMessage(content=bot_dialog))
        elif retry_count > 2:
            botspeak("Exiting the chat as the information is required to proceed further.")
            sys.exit(0)
        return {**state, "retry_count": retry_count, "history": history}, None
    state[node][field] = details
    return {**state, "retry_count": 0, "history": history}, True


def handle_node_entry(state: State, node_name: str) -> State:
    # print(f"Entering node: {node_name}")
    # print(f"Current state: {state}")
    if state.get("current_node") != node_name:
        return {**state, "retry_count": 0, "current_node": node_name}
    return state


def collect_name(llm, state):
    state = handle_node_entry(state, "collect_name")
    new_state, result = collect_field(llm, state, "name", greeting=True)
    new_state["collect_name_result"] = result
    return new_state


def service_choice(llm, state):
    state = handle_node_entry(state, "service_choice")
    new_state, result = collect_field(llm, state, "service_type")
    new_state["service_choice_result"] = result
    new_state["service_value"] = (new_state.get("service") or "").lower()
    return new_state


def check_in_booking(llm, state):
    state = handle_node_entry(state, "check_in_booking_node")
    new_state, result = collect_field_visual(llm, state, "check_in", "booking_details")
    return new_state


def check_in_passport(llm, state):
    state = handle_node_entry(state, "check_in_passport_node")
    type = state.get("check_in", {}).get("passenger_details", {}).get("type")
    if type == "domestic":
        new_state, result = collect_field_visual(llm, state, "check_in", "aadhar_details")
    else:
        new_state, result = collect_field_visual(llm, state, "check_in", "passport_details")
    return new_state


def seat_preference(llm, state):
    state = handle_node_entry(state, "seat_preference_node")
    if not state['check_in']['passenger_details']['seat_no']:
        avilable_seats = MOCK_DATA['seats_data']['available']
        new_state, result = collect_field(llm, state, "seat_no", avilable_seats)
        selected_seat = result
        if selected_seat is not None and selected_seat in avilable_seats:
            new_state["check_in"]["passenger_details"]["seat_no"] = result
        else:
            history = state.get("history", [])
            history.append(AIMessage(content="Please select a valid seat from the available options."))
            state["history"] = history
        return new_state
    return state

def check_destination(llm, state):
    cities = list(FLIGHT_DATA.keys())
    destination = state["ticket_booking"]["destination"]
    if destination.lower() in cities:
        return destination.lower()
    else:
        cities_str = ", ".join(cities) if isinstance(cities, list) else str(cities)
        if isinstance(destination, (tuple, list)):
            destination = destination[0]
        prompt = ChatPromptTemplate.from_template(
            "The user has provided a destination '{destination}' "
            "which does not resemble any of the available cities in my list: {cities_str}. "
            "Check if the user has provided a valid destination. "
            "Determine if the user's input corresponds to any city in the list, even if it's an abbreviation or shorthand. "
            "If it matches, return the full correct city name from the list. "
            "If it does not correspond to any city in the list, return None.\n"
            "{format_instructions}"
        )


        response_schemas = [
            ResponseSchema(name="correct_destination", description="Corrected destination")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        chain = prompt | llm | output_parser

        response = chain.invoke({
            "destination":destination,
            "cities_str":cities_str,
            "format_instructions":format_instructions
        })
        if response["correct_destination"] is None:
            return None
        elif response["correct_destination"].lower() in cities:
            return response["correct_destination"].lower()
        else:
            return None


def book_ticket(llm, state):
    state = handle_node_entry(state, "book_ticket_node")
    state, result = collect_field(llm, state, "ticket_type", node="ticket_booking")
    new_state, result = collect_field(llm, state, "destination", node="ticket_booking")
    destination = check_destination(llm, new_state)
    if destination is None:
        new_state["ticket_booking"]["destination"] = None
        bot_message = "My sincere apologies but currently we don't opearate at that destination. Is there anything else I can help you with?"
        botspeak(bot_message)
    else:
        flights = FLIGHT_DATA[destination]
        botspeak("Your options are listed below")
        pprint(flights)
        selection = input("Please select a flight from the above options: ")
        flight, type = selection.split(".")
        if type.lower() == "o":
            price = flights[flight]['price']['oneway']
        elif type.lower() == "r":
            price = flights[flight]['price']['roundtrip']
        new_state["ticket_booking"]["flight"] = flights[flight]
        new_state.setdefault("check_in", {}).setdefault("passenger_details", {})["type"] = "domestic"
        new_state['check_in']['passenger_details']['seat_no'] = None
        new_state["amount"] += price
    return new_state