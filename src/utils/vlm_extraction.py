import base64
import os
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")
    return image_b64


def extract_details_with_vllm(image_path, data_format):
    image_b64 = encode_image_to_base64(image_path)
    prompt = "Extract the following details as a proper JSON object. If any of these fields are not present, set them to null."
    image_url = f"data:image/png;base64,{image_b64}"

    response_schemas = [ResponseSchema(name=field['name'], description=field["description"]) for field in data_format]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = f"{prompt}\n{format_instructions}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # llm = ChatOpenAI(
    #     model="Qwen/Qwen2-VL-7B-Instruct",
    #     openai_api_key="EMPTY",
    #     openai_api_base="http://10.45.100.6:8000/v1",
    #     max_tokens=128,
    #     temperature=0.0,
    # )
    llm = AzureChatOpenAI(
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        model=os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-4-vision-preview"),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        max_tokens=128,
        temperature=0.0,
    )

    result = llm.invoke(messages)
    try:
        structured_response = output_parser.parse(result.content)
    except Exception:
        structured_response = {field["name"]: None for field in data_format}
    return structured_response
