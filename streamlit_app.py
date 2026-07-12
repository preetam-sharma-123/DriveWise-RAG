"""
streamlit_app.py
-----------------
Web interface for Drive Wise, per the "Web interface" item in the problem
statement's technology stack.

Run with:
    streamlit run streamlit_app.py

Requires GOOGLE_API_KEY to be set (see README) and the FAISS index already
built via `python vectorstore.py --build`.
"""

import streamlit as st

from rag_pipeline import answer_query
from vectorstore import BROCHURE_REGISTRY

st.set_page_config(page_title="Drive Wise", page_icon="🚗", layout="centered")

st.title("🚗 Drive Wise")
st.caption("Metadata-Aware Automotive RAG Assistant — ask questions grounded in official brochures.")

brand_model_map: dict[str, list[str]] = {}
for meta in BROCHURE_REGISTRY.values():
    brand_model_map.setdefault(meta.brand, [])
    if meta.model not in brand_model_map[meta.brand]:
        brand_model_map[meta.brand].append(meta.model)

if not brand_model_map:
    st.error("No brochures found in BROCHURE_REGISTRY. Add one in vectorstore.py and rebuild the index.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", sorted(brand_model_map.keys()))
with col2:
    model = st.selectbox("Model", sorted(brand_model_map[brand]))

selected_car = f"{brand}::{model}"
if st.session_state.get("selected_car") != selected_car:
    st.session_state["selected_car"] = selected_car
    st.session_state["history"] = []

st.divider()

for turn in st.session_state.get("history", []):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["sources"]:
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.markdown(
                        f"- **{s['brochure']}** | Section: *{s['section']}* | Version: {s['doc_version']}"
                    )
        st.caption(f"Answered in {turn['latency']:.2f}s")

question = st.chat_input(f"Ask anything about the {brand} {model}...")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the brochure..."):
            result = answer_query(brand, model, question)
        st.write(result.answer)
        if result.sources:
            with st.expander("Sources"):
                for s in result.sources:
                    st.markdown(
                        f"- **{s['brochure']}** | Section: *{s['section']}* | Version: {s['doc_version']}"
                    )
        st.caption(f"Answered in {result.latency_seconds:.2f}s")

    st.session_state["history"].append(
        {
            "question": question,
            "answer": result.answer,
            "sources": result.sources,
            "latency": result.latency_seconds,
        }
    )