import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ratelimiter import RateLimitMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel
from xml.etree import ElementTree as ET
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv(override=True)

DEFAULT_PORT = 8080
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER")
LLM_MODEL = os.getenv("LLM_MODEL")
SERVER_ENV = os.getenv("SERVER_ENV", "production")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", f"http://localhost:{DEFAULT_PORT}")
TEMPERATURE = 0.8
MAX_TOKENS_PER_SECTION = 4096

if LLM_PROVIDER == "openai":
    API_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_LLM_MODEL = "gpt-4o"
elif LLM_PROVIDER == "openrouter":
    API_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_LLM_MODEL = "anthropic/claude-3.5-sonnet"

app = FastAPI()

if SERVER_ENV == "production":
    # Rate limiting middleware if this ever gets pushed to production.
    app.add_middleware(
        RateLimitMiddleware,
        limit=5,
        window=60,
    )

# CORS configuration
if os.getenv("SERVER_ENV") == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Only allow requests from the frontend origin in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[os.getenv("FRONTEND_ORIGIN")],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=API_BASE_URL)

# Define the model for the incoming topic request
class Topic(BaseModel):
    topic: str


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def get_valid_xml_response(client, system_prompt, user_prompt):
    """

    Get a valid XML response from the LLM.
    Retry up to 3 times if the response is not valid.

    """
    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS_PER_SECTION,
    )

    outliner_response = response.choices[0].message.content

    print("\nOutliner Response: -------------------------------")
    print(outliner_response)

    # Extract XML content between <knowledge> tags
    start_tag = "<knowledge>"
    end_tag = "</knowledge>"
    start_index = outliner_response.find(start_tag)
    end_index = outliner_response.rfind(end_tag)

    if start_index != -1 and end_index != -1:
        xml_content = outliner_response[start_index : end_index + len(end_tag)]
    else:
        print("XML tags not found, retrying...")
        raise ValueError("XML tags not found in the response")

    # Attempt to parse XML
    try:
        root = ET.fromstring(xml_content)
        return root, xml_content
    except ET.ParseError:
        print("XML parsing failed, retrying...")
        raise  # This will trigger a retry


@app.post("/api/generate")
async def generate_knowledge(topic: Topic):
    """

    Generates the long-form article about a topic.

    """
    from prompts import (
        construct_outliner_system_prompt,
        construct_outliner_user_prompt,
        construct_knowledge_generator_system_prompt,
        construct_knowledge_generator_user_prompt,
    )

    print("\n--------------------------------")
    print("Beginning knowledge generation...")

    outliner_system_prompt = construct_outliner_system_prompt()
    outliner_user_prompt = construct_outliner_user_prompt(topic.topic)

    try:
        root, outliner_response = await get_valid_xml_response(
            client, outliner_system_prompt, outliner_user_prompt
        )
    except Exception as e:
        return {
            "error": f"Failed to get valid XML response after 3 attempts: {str(e)}",
            "content": "",
        }

    # Extract topic
    topic = root.find("topic").text

    # Extract sections and create prompts
    section_user_prompts = []
    for section in root.find("sections").findall("section"):
        title = section.find("title").text
        section_user_prompts.append(
            construct_knowledge_generator_user_prompt(title, outliner_response)
        )

    # Now we have a list of prompts in section_user_prompts
    # We'll use these to generate content for each section
    knowledge_generator_system_prompt = construct_knowledge_generator_system_prompt()

    # Function for generating content for a section...
    # allows us to generate content in parallel.
    async def generate_section_content(index, prompt):
        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": knowledge_generator_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS_PER_SECTION,
                temperature=TEMPERATURE,
            )
            return index, response.choices[0].message.content
        except Exception as e:
            return index, f"Error: {str(e)}"

    tasks = [
        generate_section_content(i, prompt) for i, prompt in enumerate(section_user_prompts)
    ]

    section_contents = await asyncio.gather(*tasks)

    # Check if any sections had errors
    errors = [content for _, content in section_contents if content.startswith("Error:")]
    if errors:
        error_message = "; ".join(errors)
        return {
            "error": f"Error generating content: {error_message}",
            "content": "",
        }

    # Sort the results by index to maintain original order
    sorted_section_contents = sorted(section_contents, key=lambda x: x[0])

    # Extract just the content, discarding the index
    final_content = [content for _, content in sorted_section_contents]

    # Combine all section contents into a single string
    combined_content = "\n\n".join(final_content)

    return {"content": combined_content, "outline": outliner_response}


@app.get("/")
async def read_index():
    """

    Serves the frontend.

    """
    return FileResponse("index.html")


# -----------------------------------------
#
# Server startup
#
# -----------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", DEFAULT_PORT))
    reload = os.getenv("SERVER_ENV") != "production"

    uvicorn.run("main:app", host=host, port=port, reload=reload)
