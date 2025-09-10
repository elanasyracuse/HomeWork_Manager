import os
import io
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file (if present)
load_dotenv()

# ---------- helpers --------- 
def read_pdf(file_obj) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz) if available,
    otherwise fall back to PyPDF2. No hard dependency—imports are inside.
    """
    data = file_obj.read()  # read bytes once

    # Try PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for p in doc:
            pages.append(p.get_text("text"))
        doc.close()
        return "\n".join(pages).strip()
    except Exception:
        pass

    # Try PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception:
        pass

    return ""  # if neither works or extraction fails

def clear_doc_state():
    for k in ("document_text", "document_name"):
        if k in st.session_state:
            del st.session_state[k]

# ---------- UI ----------
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
)

# Read API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("⚠️ No API key found. Please add it to your `.env` file as `OPENAI_API_KEY=your_key_here`")
else:
    # Model choice — include all 4 requested
    model = st.selectbox(
        "Choose a model",
        ["gpt-3.5-turbo", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"],
        index=1,
    )

    # Create OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Only allow .txt and .pdf
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # If user removed the file, purge all doc data immediately
    if not uploaded_file:
        clear_doc_state()

    # If a file is present, read it fresh each time (no lingering access)
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'txt':
            document = uploaded_file.read().decode(errors="ignore")
            st.session_state["document_text"] = document
            st.session_state["document_name"] = uploaded_file.name
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
            if not document:
                st.error(
                    "Couldn't extract text from this PDF. "
                    "Install PyMuPDF (fitz) or PyPDF2 in your environment and try again."
                )
                clear_doc_state()
            else:
                st.session_state["document_text"] = document
                st.session_state["document_name"] = uploaded_file.name
        else:
            st.error("Unsupported file type.")
            clear_doc_state()

    # Ask the question
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder='Example: "Is this course hard?"',
        disabled=("document_text" not in st.session_state),
    )

    # If we have a doc and a question, call the model
    if ("document_text" in st.session_state) and question:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Here's a document: {st.session_state['document_text']}\n\n---\n\n{question}"
                ),
            }
        ]

        # Stream the response
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)
