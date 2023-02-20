import streamlit as st
import pandas as pd
import streamlit.components.v1 as components  # Import Streamlit
from streamlit_chat import message
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
from langchain.vectorstores import Pinecone, VectorStore
from langchain import PromptTemplate
import openai
from streamlit_chat import message
from sidebar import sidebar


st.set_page_config(
    page_title="Knowledge Base", page_icon="🐈", layout = "centered")

pinecone.init(
    api_key="d0fe562f-e3bf-4c98-8a16-f300c6f6c706",  # find at app.pinecone.io
    environment="us-east1-gcp"  # next to api key in console
)
if 'bot' not in st.session_state:
    st.session_state['bot'] = ["👋 Greetings! I'm your Compliance Assistance Trainer bot or CATbot🐈‍⬛ How may I help you today?"]

if 'user' not in st.session_state:
    st.session_state['user'] = ["Hi"]
    
if 'query' not in st.session_state:
    st.session_state['query'] = ""

if 'convo' not in st.session_state:
    st.session_state['convo'] = ["AI: 👋 Greetings! I'm your Compliance Assistance Trainer bot or CATbot🐈‍⬛ How may I help you today?"]

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings()

def load_index():
    index = "itex-hud"
    pinecone.init(
    api_key=st.secrets["PINECONE_KEY"],  # find at app.pinecone.io
    environment="us-east1-gcp"  # next to api key in console
    )
    return Pinecone.from_existing_index(index, embeddings)

with st.spinner("Connecting to OpenAI..."):
    openai.api_key = st.secrets["OPENAI_API_KEY"]

with st.spinner("Connecting to Pinecone..."):
    vectorstore = load_index()
    llm = OpenAI(model_name="text-davinci-003",n=1, temperature=0.7, max_tokens = 200)

prompt = """
You are a Compliance Assistant Trainer bot, also known as CATbot, developed for ITEX group by Chatbot Developer - Amogh Agastya. Use the following pieces of context to answer the question in a paragraph at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
#Context
{context}

Question: {question}
Helpful Answer:
"""

QA_PROMPT = PromptTemplate(
    template=prompt, input_variables=["context", "question"]
)

qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0), vectorstore, qa_prompt=QA_PROMPT)

# chat_history = [("what is the min age for assisted housing", "18 years old")]
# query = "what were we just talking about?"
# result = qa({"question": query, "chat_history": chat_history})
# print('answer', result["answer"])

def clear_text():
    st.session_state["text"] = ""
    
    
def generate_ans(user_input):
    print('generating answer... user input', user_input)
    st.session_state.user.append(user_input)
    st.session_state.convo.append("User: " + user_input)
    ctx = ('\n').join(st.session_state.convo)
    print('convo so far', st.session_state.convo)
    st.session_state.convo.append("AI: ")
    result = qa({"question": user_input, "chat_history": [(st.session_state.convo[-2:])]})
    print('res obj', result)
    print('answer', result["answer"])
    return st.session_state.bot.append(result["answer"])

def get_sources(query):
    """Gets the source documents for an answer."""

    # Get sources for the answer
    index = load_index()
    docs = index.similarity_search_with_score(query, k=5)
    contexts = []

    for i in docs:
        tuple = ()
        tuple += (i[0].page_content, i[1])
        contexts.append(tuple)
    return contexts

def main():
    # answer_col, sources_col = st.col(2)  
    st.subheader("🐈  Chat with your CATBot!")
    sidebar()
    with st.container():

        search = st.container()
        # query = st.session_state['query']
        query = search.text_input('Ask a compliance related question', value="", key="text",)
        trigger = False
        
        # search.button("Go!", key = 'go')
        if search.button("Go!") or query != "":
            with st.spinner("Thinking..."):
                # lowercase relevant lib filters
                # ask the question
                result = generate_ans(query)
                # clear_text()            
                trigger = True
                
    # with st.container():
    #     if trigger:
    #         st.markdown("#### Sources:")
    #         sources = get_sources(st.session_state['text'])
    #         for i in sources:
    #             st.write(i[0].strip())
    #             st.write('Confidence: ', i[1])

    if st.button("Reset Chat", on_click=clear_text):
        st.session_state['bot'] = ["👋 Greetings! I'm your Compliance Assistance Trainer bot or CATbot🐈 How may I help you today?"]
        st.session_state['user'] = ["Hi"]
        st.session_state['convo'] = ["AI: 👋 Greetings! I'm your Compliance Assistance Trainer bot or CATbot🐈 How may I help you today?"]
        
    if st.session_state['bot']:
            for i in range(len(st.session_state['bot'])-1, -1, -1):
                message(st.session_state["bot"][i], key=str(i), avatar_style="bottts",seed=1)
                message(st.session_state['user'][i], is_user=True, key=str(i) + '_user')

main()
