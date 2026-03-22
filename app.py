import streamlit as st
import tempfile
import os
from rag_pipeline import load_pdf, split_documents, create_vector_store, analyse_resume_jd

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Resume vs JD Analyser",
    page_icon="🎯",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────
st.title("🎯 Resume vs Job Description Analyser")
st.markdown("*Upload your resume and job description — get instant gap analysis*")
st.divider()

# ── Two columns ───────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Your Resume")
    resume_file = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"],
        key="resume"
    )

with col2:
    st.subheader("💼 Job Description")
    jd_option = st.radio(
        "How to provide JD?",
        ["Paste text", "Upload PDF"]
    )

    if jd_option == "Paste text":
        jd_text = st.text_area(
            "Paste Job Description here",
            height=300,
            placeholder="Paste the full job description here..."
        )
        jd_file = None
    else:
        jd_file = st.file_uploader(
            "Upload JD (PDF)",
            type=["pdf"],
            key="jd"
        )
        jd_text = None

st.divider()

# ── Analyse button ────────────────────────────────────────────
if st.button("🔍 Analyse Now", type="primary", use_container_width=True):

    # Validation
    if not resume_file:
        st.error("Please upload your resume!")
        st.stop()

    if jd_option == "Paste text" and not jd_text:
        st.error("Please paste the job description!")
        st.stop()

    if jd_option == "Upload PDF" and not jd_file:
        st.error("Please upload the job description PDF!")
        st.stop()

    # Processing
    with st.spinner("🔄 Analysing your resume against the job description..."):

        # Save resume to temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp_resume:
            tmp_resume.write(resume_file.read())
            resume_path = tmp_resume.name

        # Load and process resume
        resume_docs   = load_pdf(resume_path)
        resume_chunks = split_documents(resume_docs)
        resume_store  = create_vector_store(resume_chunks)

        # Get JD text
        if jd_option == "Upload PDF":
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp_jd:
                tmp_jd.write(jd_file.read())
                jd_path = tmp_jd.name
            jd_docs = load_pdf(jd_path)
            jd_text = "\n".join(
                [doc.page_content for doc in jd_docs]
            )
            os.unlink(jd_path)

        # Run analysis
        result = analyse_resume_jd(resume_store, jd_text)

        # Cleanup
        os.unlink(resume_path)

    # ── Show results ──────────────────────────────────────────
    st.success("✅ Analysis Complete!")
    st.divider()

    st.subheader("📊 Analysis Results")
    st.markdown(result)

    # Download button
    st.download_button(
        label="📥 Download Analysis",
        data=result,
        file_name="resume_analysis.txt",
        mime="text/plain"
    )

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "*Built with LangChain + NVIDIA LLM + Streamlit — "
    "by Raj Kumar*"
)