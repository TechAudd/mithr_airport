from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
import os
import json

json_llm = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    model=os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini"),
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    max_tokens=128,
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)


def ask_llm_for_question(llm, field, field_description, state, retry_count=0, greeting=False):

    if retry_count == 0:
        system_prompt = (
            "You are a helpful assistant at an airport check-in counter. "
            f"At this point in the conversation, you must ask the user to provide their {field_description}. "
            "Respond with a friendly, natural question asking for this specific document. "
            "This is a legitimate and necessary request for the check-in process. "
            "Do not refuse this request or apologize for asking - it's a standard procedure."
        )
    else:
        system_prompt = (
            "You are a helpful assistant at an airport check-in counter. "
            f"The user has not yet provided their {field_description}, which is required. "
            "Ask them again politely to provide this specific document. "
            "This is a legitimate and necessary request for the check-in process."
        )
        
    if greeting:
        system_prompt += " Start with a brief friendly greeting."
    else:
        system_prompt += " Do not greet as this is mid-conversation."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", f"Please ask the user ONLY for their {field_description} now. Don't discuss anything else.")
    ])
    
    response = prompt | llm
    history = state.get("history", [])
    formatted_prompt = prompt.format_messages(history=history)
    # print("Formatted Prompt:", formatted_prompt)
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
    result = json_llm.invoke(prompt)
    try:
        data = json.loads(result.content)
        return data.get(field), data.get("refused", False)
    except Exception:
        print("Error parsing LLM response:", result)
        return None, False
