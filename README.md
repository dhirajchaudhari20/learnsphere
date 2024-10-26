# LearnSphere
LearnSphere is an smart study companion that lets students query and summarize documents to enhance their learning experience.The educational application includes the following key features:
- AI-Powered Chatbot: An interactive chatbot capable of answering questions across various subjects.
- Document Processing: Functionality to generate Q&A and summaries from uploaded PDFs.
- User-Friendly Interface: An intuitive interface for interacting with the chatbot and processing documents.

## The application consists of:

 - Frontend: Built using Streamlit, allowing users to interact with the chatbot, upload documents, and view responses.
 - Backend: Handles PDF processing, text chunking, embedding generation, and query answering using AI models.
 - Storage: Uses FAISS for storing and retrieving document embeddings.
   
## Rationale Behind the Design
- Streamlit: Provides a simple way to create interactive web applications.
- FAISS: Efficiently handles vector storage and similarity search.
- AI Models: Utilize Google’s Gemini for natural language understanding and document processing.
## Prerequisites
Before you start, ensure you have the following:

 - Python: Install Python 3.x.
 - Libraries: Install required libraries using pip install.
 - Google Cloud API Key: Obtain your API key from Google Cloud and set it up in a .env file.
