from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

load_dotenv()

# llm = AzureChatOpenAI(
#         deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
#         model=os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini"),
#         api_key=os.environ["AZURE_OPENAI_API_KEY"],
#         azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#         api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
#         max_tokens=128,
#         temperature=0.0,
#     )

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"), max_tokens=128, temperature=0.0)


classification_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a text classifier. Return true if the following message is small talk, else false. "
        "Anything unrelated to the main task of airport check-in is considered small talk. "
        "Respond only with 'true' or 'false', nothing else."
    ),
    HumanMessagePromptTemplate.from_template("{text_to_classify}")
])

def is_small_talk(text: str) -> bool:
    prompt_messages = classification_prompt.format_prompt(text_to_classify=text).to_messages()
    response = llm.invoke(prompt_messages)
    text_resp = response.content.strip().lower()
    return text_resp == "true"

def human_small_talk(user_input: str, chat_history: list) -> str:
    system_message = (
        "You are a friendly and personable receptionist at an airport check-in counter. "
        "You engage naturally with travelers including making small talk when appropriate. "
        "Never reveal you are an AI or mention AI. "
        "Use context from the conversation and any relevant information retrieved from the vector database to respond. "
        "Be warm, professional, and approachable."
    )
    message_history = ChatMessageHistory()
    for msg in chat_history:
        message_history.add_message(msg)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=message_history
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False
    )
    result = conversation_chain.invoke({"input": user_input})
    return result.get("response", "")