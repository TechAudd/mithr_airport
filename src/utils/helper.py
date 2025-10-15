from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
import os
import json
import time

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


def ask_llm_for_question(llm, field, field_description, state, retry_count=0, greeting=False, conversation_history=None, small_talk_response=None):
    if conversation_history is None:
        conversation_history = []
    else:
        conversation_history = conversation_history[-5:]
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

    if small_talk_response:
        print(f"Small talk response in ask_llm_for_question: {small_talk_response}")
        system_prompt += (
            "\n\nThe user just made a small talk comment. Your friendly short reply to that is:\n"
            f"'{small_talk_response}'\n\n"
            "Begin your entire response with this small talk reply. Then, transition smoothly using a phrase like "
            "'By the way', 'Also', or 'Whenever you're ready' into politely asking for the required document. "
            "Do NOT include a separate greeting as the small talk reply itself should serve as the friendly opening. "
            "Your full response should be a single natural, coherent message."
        )
    system_prompt += "\nIf user asks for some other information give a funny response without offending them. BUT make sure to answer the question alongside asking for the required information."
    
    system_prompt += "\n\nTime now: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    system_prompt += "\n\nConversation history:\n" + "\n".join([f"User: {msg.content}" if msg.type == "human" else f"Bot: {msg.content}" for msg in conversation_history])
    system_prompt += "\n\nUse the conversation history to make the question more personal. Make sure that adding personal touches doesnt make it sound unnatural or forced, add them only when needed."
    system_prompt += " Check if the generated question sounds natural and human if attached to the conversation history."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
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
