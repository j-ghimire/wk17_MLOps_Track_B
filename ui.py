import os
import requests
import streamlit as st

st.set_page_config(page_title="Support Assistant", page_icon="")
st.title("Support Routing Assistant")
st.caption("Ask a support question and receive a routed, knowledge-grounded response.")

api_url = st.sidebar.text_input("Backend URL", os.getenv("BACKEND_URL", "http://localhost:8000"))
message = st.chat_input("What do you need help with?")

if message:
    st.chat_message("user").write(message)
    try:
        response = requests.post(f"{api_url.rstrip('/')}/chat", json={"message": message}, timeout=25)
        response.raise_for_status()
        result = response.json()
        with st.chat_message("assistant"):
            st.write(result["answer"])
            st.caption(f"Route: {result['category']} | Confidence: {result['confidence']:.0%}")
            if result["degraded"]:
                st.info("Local fallback used. Connect an LLM provider for generative responses.")
            if result["sources"]:
                with st.expander("Sources"):
                    for source in result["sources"]:
                        st.write(f"**{source['title']}** ({source['score']:.0%} match)")
                        st.write(source["content"])
    except requests.RequestException as error:
        st.error(f"The backend is unavailable: {error}")
