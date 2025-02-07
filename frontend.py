import streamlit as st
import requests
from streamlit_chat import message  # Streamlit component for chat

# Define API base URL
API_BASE_URL = "http://127.0.0.1:5000"  # Update if Flask runs on a different host or port

st.set_page_config(page_title="PDF Chatbot", layout="centered", initial_sidebar_state="expanded")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # To store chat history

# Title and instructions
st.title("PDF Chatbot")
st.markdown("Upload a PDF and chat with it using the context of the document.")

# File upload section
st.subheader("1. Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.success("File uploaded successfully! Press the button below to process it.")
    if st.button("Process PDF"):
        with st.spinner("Processing your PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post(f"{API_BASE_URL}/upload", files=files)
            if response.status_code == 200:
                st.success("PDF processed successfully! Start chatting below.")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# Chatbot section
st.subheader("2. Chat with the PDF")
st.markdown("Type your questions in the input box below.")

# Chat input
query = st.text_input("Your Question", key="input_query")
if query and st.button("Ask", key="ask_button"):
    with st.spinner("Fetching answer..."):
        # Send query to the Flask API
        response = requests.post(f"{API_BASE_URL}/ask", json={"query": query})
        if response.status_code == 200:
            result = response.json().get("response", "I don't know.")
            # Add the query and response to the session state
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "bot", "content": result})
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    is_user = msg["role"] == "user"
    message(msg["content"], is_user=is_user, key=f"msg_{i}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Flask.")