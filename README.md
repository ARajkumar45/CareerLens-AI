# 🎯 CareerLens AI — Resume vs JD Career Coach

AI-powered career coaching platform that analyses 
your resume against a job description and generates 
a personalised learning roadmap.

## Features
- 📊 Match score — how well your resume fits the JD
- ❌ Skill gap analysis — what's missing
- 📅 Personalised daily schedule — based on YOUR time
- 📈 Market trends — real-time via Tavily search
- 🗺️ Week by week roadmap — to close the gaps
- 💰 Salary prediction — realistic ranges
- 📥 Download report — as text file

## Tech Stack
- LangChain — RAG pipeline
- NVIDIA LLM API — llama-3.3-70b-instruct
- ChromaDB — vector store
- Streamlit — frontend UI
- Tavily — real-time market intelligence
- LangSmith — tracing and observability

## Setup
1. Clone the repo
2. Copy example.env to .env
3. Add your API keys to .env
4. Install dependencies:
pip install langchain langchain-community 
langchain-nvidia-ai-endpoints chromadb 
streamlit tavily-python python-dotenv pypdf

5. Run:
streamlit run app.py

## Built By
Arumugam Raj Kumar — transitioning into 
GenAI Engineering
GitHub: github.com/ARajkumar45
LinkedIn: linkedin.com/in/rajkumar45
