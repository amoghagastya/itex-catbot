import streamlit as st
from typing import Any, Dict, List
from sidebar import sidebar
from langchain import OpenAI, VectorDBQA
from langchain.vectorstores import Pinecone, VectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import pinecone
from langchain.chains.question_answering import load_qa_chain
import os
import sys

os.environ['OPENAI_API_KEY'] = "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"

st.set_page_config(page_title="Compliance Assistance Trainer", page_icon="📖", layout="wide")


def clear_submit():
    st.session_state["submit"] = False

embeddings = OpenAIEmbeddings(openai_api_key="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")

def load_index():
    index = "itex-hud"
    pinecone.init(
    api_key="d0fe562f-e3bf-4c98-8a16-f300c6f6c706",  # find at app.pinecone.io
    environment="us-east1-gcp"  # next to api key in console
    )
    return Pinecone.from_existing_index(index, embeddings)

with st.spinner("Connecting to Pinecone..."):
    index = load_index()
    index_name = "itex-hud"

# @st.cache(allow_output_mutation=True)
def search_docs(index: VectorStore, query: str) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""

    # Search for similar chunks
    docs = index.similarity_search(query, k=5)
    return docs


# @st.cache(allow_output_mutation=True)
def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs


# @st.cache(allow_output_mutation=True)
def get_answer(query: str):
    """Gets an answer to a question from a list of Documents."""
    # Get the answer
    docsearch = Pinecone.from_existing_index(index, embeddings)       
            # qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)
    answer = qa.run(query)
    return answer

    

st.header("📖HUD Manual")

sidebar()

query = st.text_area("Ask a question related to the document", on_change=clear_submit)
button = st.button("Submit")
if button or st.session_state.get("submit"):
    if not query:
        st.error("Please enter a question!")
    else:
        st.session_state["submit"] = True
        # Output Columns
        answer_col, sources_col = st.columns(2)

        docsearch = Pinecone.from_existing_index(index, embeddings)       
                # qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)
        
        sources = search_docs(index, query)
        try:
            answer = get_answer(query)
                # Get the sources for the answer
            # sources = get_sources(answer, sources)

            with answer_col:
                st.markdown("#### Answer")
                # st.markdown(answer["output_text"].split("SOURCES: ")[0])
                st.markdown(answer)

            with sources_col:
                st.markdown("#### Sources")
                # for source in sources:
                #     st.markdown(source.page_content)
                #     st.markdown(source.metadata["source"])
                #     st.markdown("---")
                    
        except Exception as e:
            print('error',e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('oops',exc_type, fname, exc_tb.tb_lineno)


