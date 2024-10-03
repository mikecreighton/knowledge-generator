# knowledge-generator
A simple tool for generating information about a topic to be fed into NotebookLM.

## Description

This is a simple tool for generating a small knowledge base for a given topic, intended to be used as a "Source" for <a href="https://notebooklm.google.com/" target="_blank">NotebookLM</a>, since it needs source documents to work with. This should be especially useful for the Audio Overview feature, where maybe you want to learn about a topic, but don't have any source material to start with.

Outputs are formatted in <a href="https://www.markdownguide.org/" target="_blank"></a>Markdown</a>. When you click the "Copy" button, the Markdown will be copied to your clipboard. You can then paste that into a plain text document, into NotebookLM as copied text, or into a Google Doc, which should translate the Markdown into the appropriate formatting.

## Requirements

- Python 3.10+
- An OpenAI API key or an OpenRouter API key

## Running the app locally

1. Copy the `.env.example` file to `.env` and set the appropriate environment variables.

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