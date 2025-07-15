from nodes import node_functions, routes


def get_func_and_router(node_name):
    base = node_name.replace("_node", "")
    func = getattr(node_functions, base, None)
    router = getattr(routes, f"{base}_router", None)
    return func, router


def execute_node(node_name, llm, state, user_input=None):
    func, router = get_func_and_router(node_name)
    next_node = None

    if func:
        state = func(llm, state, user_input)
        if user_input and router:
            next_node = router(state)

    state["current_node"] = next_node if next_node else node_name
    return state
