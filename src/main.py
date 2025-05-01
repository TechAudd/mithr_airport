import os
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from langsmith import traceable
from functools import partial

import pdb
from pprint import pprint as pp

from nodes.routes import (
    service_choice_router, 
    collect_name_router, 
    check_in_booking_router,
    check_in_visual_router)
from models.userstate import State
from misc.visualise import generate_mermaid_code, visualize_workflow
from nodes.node_functions import collect_name, service_choice, check_in_booking, check_in_visual


os.environ["LANGCHAIN_PROJECT"] = "MITHR"


llm = ChatOpenAI(
    model="Qwen/Qwen2-VL-7B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base="http://10.45.100.6:8000/v1",
    max_tokens=128,
    temperature=1.0,
)

def book_ticket(llm, state):
    return state

def general_query(llm, state):
    return state

workflow = StateGraph(State)
workflow.add_node("collect_name", partial(collect_name, llm))
workflow.add_node("service_choice", partial(service_choice, llm))
workflow.add_node("check_in_booking_node", partial(check_in_booking, llm))
workflow.add_node("check_in_visual_node", partial(check_in_visual, llm))
workflow.add_node("book_ticket_node", partial(book_ticket, llm))
workflow.add_node("general_query_node", partial(general_query, llm))
workflow.set_entry_point("collect_name")

workflow.add_edge(START, "collect_name")
workflow.add_conditional_edges("collect_name", collect_name_router, ["collect_name", "service_choice"])
workflow.add_conditional_edges(
    "service_choice",
    service_choice_router,
    ["service_choice", "check_in_booking_node", "book_ticket_node", "general_query_node"]
)
workflow.add_edge("book_ticket_node", "general_query_node")
workflow.add_conditional_edges(
    "check_in_booking_node",
    check_in_booking_router,
    ["check_in_booking_node", "check_in_visual_node"]
)
workflow.add_conditional_edges(
    "check_in_visual_node",
    check_in_visual_router,
    ["check_in_visual_node", "general_query_node"]
)
workflow.add_edge("check_in_visual_node", "general_query_node")

compiled_workflow = workflow.compile()

state = State(
    name=None,
    service_type=None,
    check_in={},
    ticket_booking={},
    history=[],
    retry_count=0
)
# compiled_workflow.get_graph().draw_png("workflow_graph.png")

@traceable(run_type="chain", name="BookingWorkflow")
def run_workflow(state):
    return compiled_workflow.invoke(state)

run_workflow(state)