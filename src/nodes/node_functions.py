from langchain_core.messages import AIMessage, HumanMessage
import json
import sys

from utils.vlm_extraction import extract_details_with_vllm
from utils.helper import ask_llm_for_question, extract_field_and_refusal_with_json
from models.userstate import State

with open("conf/fields.json", "r") as file:
    FIELDS = json.load(file)


def collect_field(llm, state, field):
    if type(state) is not dict:
        state = state[0]
    retry_count = state.get("retry_count", 0)
    history = state.get("history", [])
    field_desc = FIELDS[field]["description"]
    question = ask_llm_for_question(llm, field, field_desc, history, retry_count)
    print("Bot:", question)
    user_input = input("You: ")
    history.append(AIMessage(content=question))
    history.append(HumanMessage(content=user_input))
    value, refused = extract_field_and_refusal_with_json(llm, field, field_desc, user_input)
    if refused or not value:
        retry_count = state.get("retry_count", 0) + 1
        if retry_count == 2:
            print("Bot: This information is required to proceed further and chat will terminate if not provided. Could you please provide it?")
            history.append(AIMessage(content="This information is required to proceed. Could you please provide it?"))
        elif retry_count > 2:
            print("Bot: Exiting the chat as the information is required to proceed further.")
            sys.exit(0)
        return {**state, "retry_count": retry_count, "history": history}, None
    return {**state, field: value, "retry_count": 0, "history": history}, True


def collect_field_visual(llm, state, node, field):
    if type(state) is not dict:
        state = state[0]
    retry_count = state.get("retry_count", 0)
    history = state.get("history", [])
    field_desc = FIELDS[field]["description"]
    question = ask_llm_for_question(llm, field, field_desc, history, retry_count)
    print("Bot:", question)
    image_path = input("Please provide the path to the image: ")
    history.append(AIMessage(content=question))
    details = extract_details_with_vllm(image_path, data_format=FIELDS[field]["data_format"])
    history.append(HumanMessage(content=str(details)))
    if not details or any(value is None for value in details.values()):
        retry_count = state.get("retry_count", 0) + 1
        if retry_count == 2:
            bot_dialog = "This information is required to proceed further. Could you please recapture the image properly?"
            print(f"Bot: {bot_dialog}")
            history.append(AIMessage(content=bot_dialog))
        elif retry_count > 2:
            print("Bot: Exiting the chat as the information is required to proceed further.")
            sys.exit(0)
        return {**state, "retry_count": retry_count, "history": history}, None
    state[node][field] = details
    return {**state, "retry_count": 0, "history": history}, True


def handle_node_entry(state: State, node_name: str) -> State:
    if state.get("current_node") != node_name:
        return {**state, "retry_count": 0, "current_node": node_name}
    return state


def collect_name(llm, state):
    state = handle_node_entry(state, "collect_name")
    new_state, result = collect_field(llm, state, "name")
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


def check_in_visual(llm, state):
    state = handle_node_entry(state, "check_in_visual_node")
    new_state, result = collect_field_visual(llm, state, "check_in", "aadhar_details")
    return new_state
