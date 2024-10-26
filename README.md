#LearnSphere
LearnSphere is a smart study companion designed to enhance students' learning experiences by enabling them to query and summarize documents.

#Key Features:
AI-Powered Chatbot: An interactive chatbot that can answer questions across various subjects.
Document Processing: Functionality to generate Q&A and summaries from uploaded PDFs.
User-Friendly Interface: An intuitive interface for engaging with the chatbot and processing documents.
Application Components:
Frontend: Built using Streamlit, allowing users to interact with the chatbot, upload documents, and view responses.
Backend: Handles PDF processing, text chunking, embedding generation, and query answering using AI models.
Storage: Utilizes FAISS for storing and retrieving document embeddings.
Rationale Behind the Design:
Streamlit: Offers a simple way to create interactive web applications.
FAISS: Efficiently manages vector storage and similarity searches.
AI Models: Leverages Google’s Gemini for natural language understanding and document processing.
# Prerequisites:
Before you start, ensure you have the following:

Python: Install Python 3.x.
Libraries: Install the required libraries using pip install.
Google Cloud API Key: Obtain your API key from Google Cloud and set it up in a .env file.
Made by Team Ligc Lrods at Mumbai Hack 2024.
