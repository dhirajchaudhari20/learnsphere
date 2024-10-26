import os
import speech_recognition as sr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    st.error("API Key not found. Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# Read all PDF files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text() or ""
            if extracted_text:  # Check if text is not empty
                text += extracted_text + "\n"  # Add a newline for better chunking
    return text.strip()

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Adjusted chunk sizes for better handling
    return splitter.split_text(text)

# Get embeddings for each chunk
def get_vector_store(chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def get_conversational_chain():
    prompt_template = """
    You are an AI assistant. Answer the user's question based on the context provided. 
    If the answer is not available, say: "Answer is not available in the context."
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        # If no documents found, get response from Gemini API
        if not docs:  
            gemini_response = get_gemini_response(user_question)  
            return {"output_text": [gemini_response]}

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        # If no response found in the PDFs, use Gemini API
        if not response['output_text'].strip():
            gemini_response = get_gemini_response(user_question)
            return {"output_text": [gemini_response]}

        return response
    except Exception as e:
        st.error(f"Error processing your question: {e}")
        return None

def get_gemini_response(question):
    """Fetch answer from Gemini API."""
    try:
        prompt = f"Please provide a detailed answer for the question: {question}"
        response = genai.generate_text(prompt=prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error fetching response from Gemini API: {e}")
        return "Answer is not available."

def speech_to_text():
    """Convert speech input to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for your question...")
        audio = r.listen(source)
        try:
            question = r.recognize_google(audio)
            st.success(f"You asked: {question}")
            return question
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service.")
            return None

def main():
    st.set_page_config(page_title="LearnSphere", page_icon="🤖", layout="wide")

    # Custom CSS for better aesthetics
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f2f2f2;
        }
        .footer {
            text-align: center;
            padding: 20px;
            background-color: #f2f2f2;  /* Matches sidebar background */
            border-top: 1px solid #ccc;  /* Keeps the border */
            position: relative;  /* Ensures it's positioned relative to the main content */
            bottom: 0;  /* Positions at the bottom of the page */
            width: 100%;  /* Ensures it spans the full width */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

        # Button for speech recognition
        if st.button("Speak to Ask"):
            question = speech_to_text()
            if question:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = user_input(question)
                        if response is not None:
                            full_response = "".join(response['output_text'])
                            st.write(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Main content area for displaying chat messages
    st.title("LearnSphere: Your AI-Powered Learning Assistant")
    st.write("Welcome to the chat! You can ask questions based on the uploaded PDFs.")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        clear_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    if response is not None:
                        full_response = "".join(response['output_text'])
                        st.write(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Footer with team information
   

if __name__ == "__main__":
    main()
