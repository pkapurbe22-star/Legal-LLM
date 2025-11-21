import streamlit as st
import os
import re
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DOCS_FOLDER = r"C:\Users\Prachee\OneDrive\Desktop\commerical _courts"
HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN
@st.cache_resource(show_spinner=False)
def get_rag_chain():
    if not os.path.exists(DOCS_FOLDER):
        st.error(f"Folder not found: {DOCS_FOLDER}")
        st.stop()

    pdf_loader = DirectoryLoader(
        DOCS_FOLDER, glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    txt_loader = DirectoryLoader(
        DOCS_FOLDER, glob="**/*.txt", loader_cls=TextLoader
    )
    all_docs = pdf_loader.load() + txt_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, chunk_overlap=1000
    )
    chunks = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
        model="deepseek-ai/DeepSeek-R1",
        temperature=0.0,
        max_tokens=1024,
    )

    template = """
You are a legal assistant for commercial court matters.
Answer questions strictly using the context from the documents.

Chat history:
{chat_history}

Document context:
{context}

Current user question:
{question}

Guidelines:
- Use ONLY the information that appears in the context.
- If the answer is not clearly supported by the context, say:
  "I’m not sure based on the provided documents."
- Do NOT invent facts, case names, or legal conclusions that are not in the context.
- If the question is vague (e.g., "what?" or "explain more"), say what you
  need the user to clarify, based only on the context.
- If asked hello or how are you you can respond accordinly i.e. by saying hello hope you are having a good day.

Provide a clear, concise answer. Use bullet points where helpful.
"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    question_selector = itemgetter("question")

    chain = (
        {
            "context": question_selector | retriever | format_docs,
            "question": question_selector,
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

st.title("Legal Query")

if "messages" not in st.session_state:
    st.session_state.messages = []

rag_chain = get_rag_chain()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask your question here...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    history_str = ""
    for m in st.session_state.messages[:-1]:
        speaker = "User" if m["role"] == "user" else "Assistant"
        history_str += f"{speaker}: {m['content']}\n"

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                raw_response = rag_chain.invoke(
                    {
                        "question": query,
                        "chat_history": history_str,
                    }
                )
                clean_response = re.sub(
                    r"<think>.*?</think>", "", raw_response, flags=re.DOTALL
                ).strip()

                st.markdown(clean_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": clean_response}
                )
            except Exception as e:
                st.error(f"Error processing request: {e}")
