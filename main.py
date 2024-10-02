import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from ratelimiter import RateLimitMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel
from xml.etree import ElementTree as ET

load_dotenv(override=True)

DEFAULT_PORT = 8080
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER")
LLM_MODEL = os.getenv("LLM_MODEL")
SERVER_ENV = os.getenv("SERVER_ENV", "production")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", f"http://localhost:{DEFAULT_PORT}")
TEMPERATURE = 0.8
MAX_TOKENS = 4096

if LLM_PROVIDER == "openai":
    API_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_LLM_MODEL = "gpt-4o"
elif LLM_PROVIDER == "openrouter":
    API_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_LLM_MODEL = "anthropic/claude-3.5-sonnet"

app = FastAPI()


# Add the rate limiting middleware
# app.add_middleware(
#     RateLimitMiddleware,
#     limit=5,
#     window=60,
# )

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


@app.post("/api/generate")
async def generate_knowledge(topic: Topic):
    from prompts import (
        construct_outliner_system_prompt,
        construct_outliner_user_prompt,
        construct_knowledge_generator_system_prompt,
        construct_knowledge_generator_user_prompt,
    )

    outliner_system_prompt = construct_outliner_system_prompt()
    outliner_user_prompt = construct_outliner_user_prompt(topic.topic)

    try:
        # outliner_response = """"""

        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": outliner_system_prompt},
                {"role": "user", "content": outliner_user_prompt},
            ],
        )

        outliner_response = response.choices[0].message.content

        # Parse the XML string
        root = ET.fromstring(outliner_response)

        # Extract topic
        topic = root.find("topic").text

        # Extract sections and create prompts
        section_prompts = []
        for section in root.find("sections").findall("section"):
            title = section.find("title").text
            section_prompts.append(
                f"Write a detailed section about '{title}' for the topic '{topic}'."
            )

        # Now you have a list of prompts in section_prompts
        # You can use these to generate content for each section
        for section_prompt in section_prompts:
            print("\nSection Prompt: ", section_prompt)

    except ET.ParseError as e:
        return {"error": f"XML parsing error: {str(e)}", "content": outliner_response}
    except Exception as e:
        return {"error": str(e), "content": ""}

    return {"content": outliner_response, "section_prompts": section_prompts}


@app.get("/")
async def read_index():
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
