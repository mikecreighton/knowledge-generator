# Knowledge Generator
A simple tool for generating information about a topic to be fed into [NotebookLM](https://notebooklm.google.com/).

## Description

This is a simple tool for generating a small knowledge base for a given topic, intended to be used as a "Source" for NotebookLM, since it needs source documents to work with. This should be especially useful for the Audio Overview feature, where maybe you want to learn about a topic, but don't have any source material to start with.

Outputs are formatted in [Markdown](https://www.markdownguide.org/). When you click the "Copy" button, the Markdown will be copied to your clipboard. You can then paste that into a plain text document, into NotebookLM as copied text, or into a Google Doc to be imported into NotebookLM.

Also, you can just read it, becuase it'll be a nice, organized document about the topic.

_Note_: Large language models (LLMs), which generate the content, are prone to hallucinations. This means that the content may not be accurate or complete. It is up to you to fact check the content for accuracy. Also, because of the nature of how LLMs are trained, each LLM has a different knowledge cut-off point, so you may find that recent information is missing from the content.

_Another Note_: This was hastily put together because I wanted to help my kids with resources for some of their interests, so it'll probably have bugs or I'll have overlooked something.

## Requirements

- Python 3.10+
- An [OpenAI API key](https://openai.com/index/openai-api/) or an [OpenRouter API key](https://openrouter.ai/)

## Running the app locally

1. Copy the `.env.template` file to `.env` and set the appropriate environment variables.

2. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the app:

```bash
python main.py
```

4. Open your browser and navigate to `http://localhost:8080`.
