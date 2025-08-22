import os
import json
import re
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv
import hashlib
import time

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
FAISS_DIR = "career_faiss_index"
KNOWLEDGE_BASE_DIR = "knowledge_base"
LEARNED_KNOWLEDGE_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "learned_knowledge")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SIMILARITY_THRESHOLD = 0.85  # A high threshold for finding exact or near-exact matches

if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Ensure the directory for learned knowledge exists
os.makedirs(LEARNED_KNOWLEDGE_DIR, exist_ok=True)

# --- Helper Functions ---
def load_vectorstore():
    """Loads the FAISS vector store from the local directory."""
    try:
        print("Loading FAISS vector store...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS vector store loaded successfully.")
        return vectorstore
    except Exception as e:
        if "No such file or directory" in str(e) or "could not be deserialized" in str(e):
            raise RuntimeError(
                f"❌ Error loading FAISS index from {FAISS_DIR}. "
                f"Please run 'python ingest.py' first to create the index."
            )
        else:
            raise RuntimeError(f"❌ An unexpected error occurred: {str(e)}")

def correct_spelling(text: str) -> str:
    """Corrects common misspellings related to careers and technology."""
    corrections = {
        "careear": "career",
        "scintist": "scientist",
        "data scienctist": "data scientist",
        "programer": "programmer",
        "softwere": "software",
        "devoloper": "developer",
        "engeneer": "engineer",
        "desiner": "designer",
        "markiting": "marketing",
        "finence": "finance",
        "bussiness": "business",
        "analystt": "analyst"
    }
    
    words = text.lower().split()
    corrected_words = [corrections.get(word, word) for word in words]
    return " ".join(corrected_words)

def parse_kb_text(text: str):
    """Parses raw text from the knowledge base into a structured dictionary."""
    data = {
        "career": None,
        "description": None,
        "skills_required": [],
        "learning_path": [],
        "future_scope": None
    }
    
    # Use regular expressions to find sections
    career_match = re.search(r'Career:\s*(.*?)(?:\nDescription:|$)', text, re.DOTALL)
    if career_match:
        data['career'] = career_match.group(1).strip()

    description_match = re.search(r'Description:\s*(.*?)(?:\nSkills Required:|$)', text, re.DOTALL)
    if description_match:
        data['description'] = description_match.group(1).strip()

    skills_match = re.search(r'Skills Required:\s*(.*?)(?:\nLearning Path:|$)', text, re.DOTALL)
    if skills_match:
        skills_text = skills_match.group(1).strip()
        if skills_text:
            data['skills_required'] = [s.strip() for s in re.split(r'[,\n]', skills_text) if s.strip()]

    learning_match = re.search(r'Learning Path:\s*(.*?)(?:\nFuture Scope:|$)', text, re.DOTALL)
    if learning_match:
        learning_text = learning_match.group(1).strip()
        if learning_text:
            data['learning_path'] = [l.strip() for l in re.split(r'[,\n]', learning_text) if l.strip()]

    future_scope_match = re.search(r'Future Scope:\s*(.*)', text, re.DOTALL)
    if future_scope_match:
        data['future_scope'] = future_scope_match.group(1).strip()

    return data

def is_parsed_data_empty(data: dict):
    """Checks if all fields in the parsed data dictionary are empty or None."""
    return (
        not data.get("career") and
        not data.get("description") and
        not data.get("skills_required") and
        not data.get("learning_path") and
        not data.get("future_scope")
    )

try:
    vectorstore = load_vectorstore()
except RuntimeError as e:
    vectorstore = None
    print(e)

# Initialize the LLM and RetrievalQA chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
qa_chain = None
if vectorstore:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff"
    )

# --- FastAPI App ---
app = FastAPI(
    title="AI 360 Career Mentor Bot",
    description="A career guidance chatbot with RAG and a self-expanding knowledge base."
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
async def home():
    return {
        "message": "✅ AI 360 Career Mentor Bot is running!",
        "status": "Ready",
        "usage": "Visit /docs for API documentation."
    }

@app.get("/chat", tags=["Chat"])
async def chat(query: str = Query(..., description="Your career-related question.")):
    """
    Responds using the knowledge base first. If no relevant info is found,
    it falls back to Gemini and adds the response to the knowledge base.
    """
    try:
        if not vectorstore or not qa_chain:
            raise HTTPException(status_code=500, detail="FAISS index not loaded. Please run ingest.py.")
        
        corrected_query = correct_spelling(query)
        print(f"Original query: '{query}' -> Corrected query: '{corrected_query}'")

        # --- NEW: Check for existing high-similarity learned documents first ---
        print("Checking for high-similarity learned documents...")
        docs_and_scores = vectorstore.similarity_search_with_score(corrected_query, k=1)
        
        if docs_and_scores and docs_and_scores[0][1] > SIMILARITY_THRESHOLD:
            doc, score = docs_and_scores[0]
            if doc.metadata.get("source") == "gemini_api":
                print(f"✅ Found a high-similarity learned document with score: {score:.2f}")
                parsed_response = parse_kb_text(doc.page_content)
                return {
                    "query": query,
                    "response": parsed_response,
                    "source": "knowledge_base (direct match)",
                    "note": "Answer found in the existing knowledge base."
                }
            else:
                print(f"⚠️ Found high-similarity initial document ({score:.2f}), but its not a 'learned' document.")

        # --- OLD LOGIC (runs if no high-similarity match is found or if it's not a learned doc) ---
        print(f"Attempting to answer from knowledge base for query: '{corrected_query}'")
        response_text = qa_chain.invoke({"query": corrected_query})["result"]
        
        negative_phrases = [
            "don't know", "not able to answer", "no answer", "not find an answer", 
            "not specifically mentioned", "do not have information on", "i am unable to provide information"
        ]
        
        is_negative_response = any(phrase in response_text.lower() for phrase in negative_phrases)
        parsed_response = parse_kb_text(response_text)
        
        if is_negative_response or is_parsed_data_empty(parsed_response):
            print(f"⚠️ KB response was unhelpful for '{corrected_query}'. Falling back to Gemini.")

            format_prompt = f"""
            You are a helpful and detailed career guidance assistant.
            Your task is to provide comprehensive information about a specific career path based on the user's query.
            You must ONLY provide the JSON object. Do not include any other text, preambles, or markdown formatting (e.g., ```json).
            If a piece of information is not available, you should use `null` or an empty array.
            
            If the user's query is a question, you must still try to fit the answer into this JSON structure. 
            If the query is too vague to determine a specific career, use your general knowledge to fill in the fields with as much detail as possible.

            {{
                "career": "string | null",
                "description": "string | null",
                "skills_required": ["string"],
                "learning_path": ["string"],
                "future_scope": "string | null"
            }}

            Question: {corrected_query}
            """
            
            gemini_response_content = llm.invoke(format_prompt).content
            gemini_response_content = gemini_response_content.replace('```json', '').replace('```', '').strip()
            
            parsed_json = None
            try:
                parsed_json = json.loads(gemini_response_content)
                knowledge_text = f"Career: {parsed_json.get('career')}\n"
                knowledge_text += f"Description: {parsed_json.get('description')}\n"
                knowledge_text += "Skills Required:\n" + "\n".join(parsed_json.get('skills_required', [])) + "\n"
                knowledge_text += "Learning Path:\n" + "\n".join(parsed_json.get('learning_path', [])) + "\n"
                knowledge_text += f"Future Scope: {parsed_json.get('future_scope')}"
                
                print("✅ Successfully parsed Gemini's JSON response.")

            except json.JSONDecodeError:
                knowledge_text = gemini_response_content
                print("❌ Gemini returned invalid JSON. Saving raw text instead.")
            
            try:
                file_hash = hashlib.md5(corrected_query.encode()).hexdigest()
                file_name = f"{file_hash[:8]}_{int(time.time())}.txt"
                file_path = os.path.join(LEARNED_KNOWLEDGE_DIR, file_name)
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(knowledge_text)
                
                print(f"✅ New knowledge for '{corrected_query}' saved to file: {file_path}")

                new_doc = Document(page_content=knowledge_text, metadata={"question": corrected_query, "source": "gemini_api"})
                vectorstore.add_documents([new_doc])
                vectorstore.save_local(FAISS_DIR)
                print(f"✅ New knowledge for '{corrected_query}' added to FAISS.")

            except Exception as save_e:
                print(f"An error occurred while saving the file or updating FAISS: {save_e}")
                return {
                    "query": query,
                    "response": parsed_json or knowledge_text,
                   
                }

            return {
                "query": query,
                "response": parsed_json or knowledge_text,
               
            }
        else:
            print(f"✅ Relevant documents found for '{corrected_query}'. Answering from knowledge base.")
            return {
                "query": query,
                "response": parsed_response
            }
               

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e), "message": "An internal error occurred while processing your request."}