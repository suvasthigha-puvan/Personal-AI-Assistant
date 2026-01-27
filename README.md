# Personal AI Assistant

A customizable AI assistant that runs on your data. This tool uses RAG (Retrieval Augmented Generation) to answer questions based on your own documents (PDFs and Text files).

## Features

- **Custom Knowledge Base**: Drop your resumes, portfolios, or documentation into a folder.
- **Configurable Persona**: Change the assistant's name and identity easily.
- **Quality Control**: Uses a secondary AI model to critique and improve responses before showing them to users.
- **Lead Capture**: Records user emails and unknown questions.

## Setup Instructions

### 1. Prerequisites
- Python 3.10 or higher installed.

### 2. Installation
1.  Clone this repository or download the files.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration
1.  Rename `.env.example` to `.env`.
2.  Open `.env` in a text editor and fill in your details:
    -   `GOOGLE_API_KEY`: Get this from Google AI Studio.
    -   `OPENAI_API_KEY`: Get this from OpenAI Platform (used for evaluating responses).
    -   `ASSISTANT_NAME`: (Optional) Change this to your name. default: "My AI Assistant"
    -   `KNOWLEDGE_DIR`: (Optional) Change the folder name where you put your docs. default: "knowledge_base"

### 4. Add Your Data
1.  Create a folder named `knowledge_base` (or whatever you set in `.env`).
2.  Add your PDF files (e.g., `resume.pdf`) and Text files (`.txt`) to this folder.
3.  The assistant will automatically read and index these files when it starts.

### 5. Run the Application
Run the script:
```bash
python app.py
```
A browser window should open with your AI assistant interface.

## Project Structure
- `app.py`: The main application code.
- `requirements.txt`: List of libraries needed.
- `.env`: (You create this) Stores your API keys. **DO NOT SHARE THIS FILE.**
- `.gitignore`: Ensures your secrets are not uploaded to GitHub.
