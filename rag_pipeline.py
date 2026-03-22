from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
import tempfile
import os

# ── Step 1: Load PDF ──────────────────────────────────────────
def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()

# ── Step 2: Split into chunks ─────────────────────────────────
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return splitter.split_documents(documents)

# ── Step 3: Create embeddings + vector store ──────────────────
def create_vector_store(chunks):
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5"
    )
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)
    return vector_store

# ── Step 4: Analyse resume vs job description ─────────────────
def analyse_resume_jd(
    resume_vector_store,
    jd_text: str
):
    # Search resume for relevant skills
    resume_results = resume_vector_store.similarity_search(
        jd_text, k=5
    )
    resume_content = "\n".join(
        [doc.page_content for doc in resume_results]
    )

    # Build LLM
    model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

    # Build prompt
    prompt = f"""
You are an expert career coach and resume analyser.

RESUME CONTENT:
{resume_content}

JOB DESCRIPTION:
{jd_text}

Please analyse and provide:

1. MATCH SCORE: Give a % match score (0-100%)

2. MATCHING SKILLS: List skills/experience from 
   resume that match the job description

3. MISSING SKILLS: List skills required in JD 
   that are NOT in the resume

4. RESUME IMPROVEMENTS: Give 3 specific bullet 
   points to add/improve in the resume to better 
   match this JD

5. OVERALL VERDICT: Should this person apply? 
   Yes/No and why in 2 sentences.

Be specific and actionable. Use bullet points.
"""

    response = model.invoke(prompt)
    return response.content