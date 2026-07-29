
import streamlit as st
import pdfplumber
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# -----------------------------
# Load AI Models
# -----------------------------

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)


# -----------------------------
# PDF Reader
# -----------------------------

def read_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text



# -----------------------------
# Create Vector Database
# -----------------------------

def create_index(text):

    chunks = []

    size = 500

    for i in range(0,len(text),size):

        chunks.append(
            text[i:i+size]
        )


    embeddings = embed_model.encode(chunks)


    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )


    index.add(
        np.array(embeddings)
    )


    return index,chunks



# -----------------------------
# Retrieve Context
# -----------------------------

def search_document(
        query,
        index,
        chunks):


    query_vector = embed_model.encode(
        [query]
    )


    distance, ids = index.search(
        np.array(query_vector),
        3
    )


    result = ""

    for i in ids[0]:

        result += chunks[i]+"\n"


    return result



# -----------------------------
# AI Answer
# -----------------------------

def ask_ai(question,context):


    prompt=f"""

Answer the question using only
the document information.

Context:

{context}


Question:

{question}

"""


    answer = generator(
        prompt,
        max_length=300
    )


    return answer[0]["generated_text"]



# -----------------------------
# Streamlit UI
# -----------------------------


st.title(
    "⚖️ AI Legal Document Assistant"
)


uploaded_file = st.file_uploader(
    "Upload Legal PDF",
    type="pdf"
)


if uploaded_file:


    st.success(
        "Document Uploaded"
    )


    text = read_pdf(
        uploaded_file
    )


    index,chunks = create_index(
        text
    )


    st.subheader(
        "Document Summary"
    )


    summary_prompt=f"""

Summarize this legal document:

{text[:4000]}

"""


    summary = generator(
        summary_prompt,
        max_length=400
    )


    st.write(
        summary[0]["generated_text"]
    )



    st.subheader(
        "Ask Questions"
    )


    question = st.text_input(
        "Enter your question"
    )


    if question:


        context = search_document(
            question,
            index,
            chunks
        )


        answer = ask_ai(
            question,
            context
        )


        st.write(answer)
