import streamlit as st
import requests
import json

# Page Config
st.set_page_config(
    page_title="Airman AI Assistant",
    page_icon="✈️",
    layout="centered"
)

# Custom CSS for a better look
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .stMarkdown p {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("✈️ Airman AI Assistant")
st.markdown("Ask questions about aviation manuals, procedures, and regulations.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call your API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Searching manuals..."):
                # Connect to your running FastAPI server
                api_url = "http://127.0.0.1:8000/ask"
                payload = {
                    "question": prompt,
                    "top_k": 5,
                    "min_similarity": 0.1
                }
                
                response = requests.post(api_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    citations = data.get("citations", [])
                    
                    # Build the response text
                    full_response = answer
                    
                    # Append citations if available
                    if citations:
                        full_response += "\n\n**Sources:**"
                        for cit in citations:
                            doc = cit.get('document', 'Unknown')
                            page = cit.get('page', '?')
                            full_response += f"\n- *{doc}* (Page {page})"
                    
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.error(f"Error: {response.status_code} - {response.text}")
                    full_response = "Sorry, I encountered an error."

        except Exception as e:
            message_placeholder.error(f"Connection Error: {e}")
            full_response = "Could not connect to the API. Is it running?"

    # 3. Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})