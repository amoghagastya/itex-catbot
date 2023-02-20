import streamlit as st

def sidebar():
    with st.sidebar:
        st.markdown("# About")
        st.markdown(
            "❓Ask a question about Compliance!"
            ''' The compliance assistance system will query the embedded HUD manual and will try
            to answer the question based on its context.'''
        )
