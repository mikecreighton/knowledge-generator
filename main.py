import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from ratelimiter import RateLimitMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel

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
    # This is where you'll implement the LLM call to generate content
    # For now, we'll return a placeholder response
    generated_content = f"# Knowledge about {topic.topic}\n\nThis is where the generated content about {topic.topic} would appear."
    return {"content": generated_content}


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
