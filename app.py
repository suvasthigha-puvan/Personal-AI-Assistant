from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv(override=True)

# Configuration
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "My AI Assistant")
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge_base")

# Initialize Clients
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=GOOGLE_API_KEY)
real_openai = OpenAI(api_key=OPENAI_API_KEY)

# --- Tool Functions ---
def push(text):
    """Sends a notification via Pushover if credentials are set."""
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if token and user:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": token,
            "user": user,
            "message": text,
        })
    else:
        print(f"Push Notification (Simulated): {text}")

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

tools = [
    {
        "type": "function", 
        "function": {
            "name": "record_user_details",
            "description": "Record user interest and email",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "name": {"type": "string"},
                    "notes": {"type": "string"}
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Record questions you can't answer",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"]
            }
        }
    }
]

class Assistant:
    def __init__(self):
        self.name = ASSISTANT_NAME
        self.chroma_client = chromadb.Client()
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.chroma_client.create_collection(name="knowledge_base", embedding_function=self.embed_fn)
        self.load_documents()

    def load_documents(self):
        documents = []
        ids = []
        
        # Ensure knowledge directory exists
        if not os.path.exists(KNOWLEDGE_DIR):
            os.makedirs(KNOWLEDGE_DIR)
            print(f"Created knowledge directory at: {KNOWLEDGE_DIR}")
            print("Please add PDF or TXT files to this folder to build your knowledge base.")
            return

        print(f"Scanning for documents in: {KNOWLEDGE_DIR}")
        
        for filename in os.listdir(KNOWLEDGE_DIR):
            file_path = os.path.join(KNOWLEDGE_DIR, filename)
            
            if filename.endswith(".pdf"):
                try:
                    reader = PdfReader(file_path)
                    pdf_text = ""
                    for page in reader.pages:
                        pdf_text += page.extract_text() + "\n"
                    chunks = [c for c in pdf_text.split("\n\n") if len(c) > 20]
                    for i, chunk in enumerate(chunks):
                        documents.append(chunk)
                        ids.append(f"{filename}_{i}")
                except Exception as e:
                    print(f"Error reading PDF {filename}: {e}")

            elif filename.endswith(".txt"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        chunks = [c for c in content.split("\n\n") if len(c) > 20]
                        for i, chunk in enumerate(chunks):
                            documents.append(chunk)
                            ids.append(f"{filename}_{i}")
                except Exception as e:
                    print(f"Error reading text file {filename}: {e}")
        
        if documents:
            self.collection.add(documents=documents, ids=ids)
            print(f"RAG: Loaded {len(documents)} knowledge chunks.")
        else:
            print("No documents found to index.")

    def query_knowledge_base(self, query):
        if self.collection.count() == 0:
            return "No knowledge base loaded."
        results = self.collection.query(query_texts=[query], n_results=3)
        return "\n\n".join(results['documents'][0])

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}")
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self, context):
        prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        prompt += f"\n\n## Relevant Context:\n{context}\n"
        prompt += f"\nAlways stay in character as {self.name}."
        return prompt
    

    def evaluate_response(self, question, context, answer):
        eval_prompt = f"""
        You are an expert editor evaluating an AI assistant's response for {self.name}.
        
        USER QUESTION: "{question}"
        RELEVANT FACTS FROM DATABASE: "{context}"
        AI RESPONSE TO EVALUATE: "{answer}"
        
        CRITERIA:
        1. ACCURACY: If the question is professional, does it match the facts? 
        2. PERSONA: Is the response professional?
        3. BOUNDARIES: If the user asks personal/irrelevant questions (family, etc.), the AI SHOULD politely refuse and steer back to professional topics.

        SPECIAL INSTRUCTION:
        If the AI response says "I'm here to discuss my professional background" (or something similar) in response to a personal question, this is a "PASS". 
        Do NOT fail the response for not having "facts" if the question was personal.

        Respond ONLY with the word "PASS" if it's good, or a brief explanation of what's wrong if it's "FAIL".
        """
        
        try:
            response = real_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a quality control judge."},
                        {"role": "user", "content": eval_prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return "PASS (Evaluation Error)"

    def chat(self, message, history):
        retrieved_context = self.query_knowledge_base(message)
        messages = [{"role": "system", "content": self.system_prompt(retrieved_context)}] + history + [{"role": "user", "content": message}]
        
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            try:
                response = gemini.chat.completions.create(model="gemini-1.5-flash", messages=messages, tools=tools)
            except Exception as e:
                return f"Error communicating with AI service: {e}"
            
            # Handle Tool Calls
            if response.choices[0].finish_reason == "tool_calls":
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(response_message)
                messages.extend(results)
                continue 
            
            # Evaluate Response
            answer_content = response.choices[0].message.content
            evaluation = self.evaluate_response(message, retrieved_context, answer_content)
            
            if "PASS" in evaluation.upper():
                return answer_content
            else:
                attempts += 1
                messages.append({"role": "assistant", "content": answer_content})
                messages.append({"role": "system", "content": f"CRITIQUE: {evaluation}. Please rewrite your previous answer to fix these issues."})
                print(f"Correction Attempt {attempts}: {evaluation}")

        return "I'm having trouble phrasing this correctly. Please contact me directly."

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY is not set. The AI will not function correctly.")
        
    ai_assistant = Assistant()
    
    with gr.Blocks(title=f"{ASSISTANT_NAME}") as demo:
        gr.Markdown(f"# {ASSISTANT_NAME}")
        
        with gr.Tab("Chat"):
            gr.ChatInterface(ai_assistant.chat, type="messages")
            
        with gr.Tab("Knowledge Base Status"):
            # Button to refresh and show what's in the DB
            try:
                initial_data = ai_assistant.collection.get(include=['documents', 'metadatas'])
            except:
                initial_data = {"status": "Database empty or not initialized"}
                
            db_view = gr.JSON(value=initial_data)
            refresh_btn = gr.Button("Refresh View")
            refresh_btn.click(lambda: ai_assistant.collection.get(include=['documents', 'metadatas']), outputs=db_view)

    demo.launch()
