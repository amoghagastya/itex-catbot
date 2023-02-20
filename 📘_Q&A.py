import streamlit as st
from sidebar import sidebar
from langchain import OpenAI, VectorDBQA
from langchain.vectorstores import Pinecone, VectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os
import sys
from openai.embeddings_utils import get_embedding
import openai
from langchain.llms import OpenAI



pinecone.init(
    api_key=st.secrets["PINECONE_KEY"],  # find at app.pinecone.io
    environment="us-east1-gcp"  # next to api key in console
    )

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

# docsearch = Pinecone.from_existing_index(index, embeddings)       
        # qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
# qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)

def load_index():
    index = "itex-hud"
    pinecone.init(
    api_key=st.secrets["PINECONE_KEY"],  # find at app.pinecone.io
    environment="us-east1-gcp"  # next to api key in console
    )
    return Pinecone.from_existing_index(index, embeddings)

def create_context(question, max_len=3750, size="ada"):
    """
    Find most relevant context for a question via Pinecone search
    """
    index = load_index()
    docs = index.similarity_search_with_score(question, k=7)
    contexts = []

    for i in docs:
        tuple = ()
        tuple += (i[0].page_content, i[1])
        contexts.append(i[0].page_content)
    
    return "\n\n###\n\n".join(contexts)

fine_tuned_qa_model="text-davinci-003",
instruction='''You are a Q&A system designed to provide helpful information to Compliance Assistants.
Write a paragraph, addressing the user's question, and use the text below to obtain relevant information. If the user's question absolutely cannot be answered based on the context, say you don't have that information.
Be sure not to make stuff up if the information is not within the context. 
\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nParagraph long Answer:"'''
max_len=3550
size="ada"
debug=False
max_tokens=400

def get_sources(query):
    """Gets the source documents for an answer."""

    # Get sources for the answer
    index = load_index()
    docs = index.similarity_search_with_score(query, k=7)
    contexts = []

    for i in docs:
        tuple = ()
        tuple += (i[0].page_content, i[1])
        contexts.append(tuple)
    return contexts


# @st.cache(allow_output_mutation=True)
def get_answer(query: str):
    """Gets an answer to a question from a list of Documents."""
    # Get the answer
    context = create_context(query)
    print("Context:\n" + context)
    print("\n\n")
    prompt = instruction.format(context, query)
    response = llm(prompt)
    print('generated', response)
    return response

with st.spinner("Connecting to OpenAI..."):
    openai.api_key = st.secrets["OPENAI_API_KEY"]

with st.spinner("Connecting to Pinecone..."):
    index = load_index()
    llm = OpenAI(model_name="text-davinci-003",n=1, temperature=0.7, max_tokens = 400)

def clear_submit():
    st.session_state["submit"] = False

def remove_newlines(string):
    string = string.replace('\n', ' ')
    string = string.replace('\\n', ' ')
    string = string.replace('  ', ' ')
    string = string.replace('  ', ' ')
    return string

def main():
    st.header("📖HUD Manual")
    sidebar()
    query = st.text_area("Ask a question related to the document", on_change=clear_submit)
    # print('query type', type(query))
    button = st.button("Submit")
    if button or st.session_state.get("submit") or query!=None:
        if not query:
            st.error("Please enter a question!")
        else:
            st.session_state["submit"] = True
            # Output Columns
            # answer_col, sources_col = st.rows(2)    
            try:
                # with answer_col:
                with st.spinner("Retrieving contexts from Vector DB and querying OpenAI..."):
                    st.markdown("#### Answer:")
                    # st.markdown(answer["output_text"].split("SOURCES: ")[0])
                    res = get_answer(query)
                    # print(res)
                    st.write(remove_newlines(res))
                    
                st.markdown("#### Sources:")
                sources = get_sources(query)
                for i in sources:
                    st.write(i[0].strip())
                    st.write('Confidence: ', i[1])
                

                
            except Exception as e:
                print('error',e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print('oops',exc_type, fname, exc_tb.tb_lineno)
            # st.markdown()
            
main()