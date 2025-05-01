from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json


def ask_llm_for_question(llm, field, field_description, history, retry_count=0):
    if retry_count == 0:
        system_prompt = (
            "You are a helpful assistant collecting information from a user. "
            f"Ask the user for {field_description} in a natural, conversational way. "
            "Do not mention the field name directly. Be friendly and concise."
            "If the history is present, the question should sound like a follow-up "
            "to the previous conversation and if it is absent then start by greeting them.")
    else:
        system_prompt = (
            "You are a helpful assistant collecting information from a user. "
            f"The user did not provide a clear answer to the request for {field_description}. "
            "Politely ask again or clarify what you need, in a conversational way. "
            "Do not mention the field name directly. Be friendly and concise."
            "If the history is present, the question should sound like a follow-up "
            "to the previous conversation and if it is absent then start by greeting them."
            "If the user refuses to answer, explain why it's needed and ask again. ")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Ask for the information.")
    ])
    response = prompt | llm
    bot_message = response.invoke({"history": history}).content
    return bot_message


def extract_field_and_refusal_with_json(llm, field, field_description, user_input):
    prompt = f"""
    Given the following user message, extract the user's {field}.
    Field description: {field_description}

    Respond ONLY with a JSON object in the format:
    {{"{field}": value, "refused": true/false}}.

    If the user refuses to answer, set "refused" to true and "{field}" to null.
    If the {field} is present, set "refused" to false and provide the value.

    Message: "{user_input}"
    """
    result = llm.invoke(prompt)
    try:
        data = json.loads(result.content)
        return data.get(field), data.get("refused", False)
    except Exception:
        print("Error parsing LLM response:", result)
        return None, False
