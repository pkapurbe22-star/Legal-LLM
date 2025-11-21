# Legal-LLM
Legal Query Assistant (RAG-Based)

This is a simple Retrieval-Augmented Generation (RAG) application built using Streamlit, LangChain, FAISS, and DeepSeek-R1.
It allows users to ask legal questions, and the system answers strictly based on the content of the PDF/TXT documents stored locally.

🚀 Features

Chat-style interface using Streamlit

Loads PDF and TXT files from a local folder

Splits documents into chunks for efficient retrieval

Creates embeddings using MiniLM-L6-v2

Stores vectors using FAISS

Sends context to DeepSeek-R1 via HuggingFace Router

Removes <think> blocks to keep responses clean

Maintains chat history for smooth conversation flow

