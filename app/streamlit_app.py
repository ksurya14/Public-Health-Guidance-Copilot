import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import streamlit as st

from src.eval import run_eval
from src.index import index_corpus
from src.rag import answer_question
from src.utils import get_project_root, load_config


def ensure_index(strategy: str, config) -> None:
    index_dir = get_project_root() / config["paths"]["index_dir"] / strategy
    if index_dir.exists():
        return
    with st.spinner(f"Building index for {strategy} chunks..."):
        index_corpus(chunk_strategy=strategy)


st.set_page_config(page_title="Public Health Guidance Copilot", layout="wide")
config = load_config()

st.title("Public Health Guidance Copilot")
st.caption("RAG over seasonal influenza vaccination guidance with hybrid retrieval")

with st.sidebar:
    st.header("Retrieval settings")
    chunk_strategy = st.selectbox("Chunk strategy", options=["fixed", "section"], index=0)
    top_k = st.slider("Top k", min_value=3, max_value=10, value=int(config["retrieval"].get("top_k", 5)))

    bm25_w = st.slider("BM25 weight", 0.0, 1.0, value=float(config["retrieval"]["hybrid_weights"].get("bm25", 0.5)))
    dense_w = 1.0 - bm25_w
    st.write(f"Dense weight: **{dense_w:.2f}**")

    use_reranker = st.checkbox("Use reranker (if available)", value=bool(config["retrieval"].get("use_reranker", True)))

    if st.button("Run eval on sample questions"):
        with st.spinner("Running evaluation..."):
            eval_results = run_eval()
        st.success("Eval complete")
        for abl in eval_results.get("ablations", []):
            st.write(f"**{abl['name']}** — recall@k: {abl['recall_at_k']:.3f}, faithfulness: {abl['faithfulness']:.3f}")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.chat_input("Ask about influenza vaccination guidance")
if user_input:
    overrides = {
        "weights": {"bm25": bm25_w, "dense": dense_w},
        "top_k": top_k,
        "use_reranker": use_reranker,
        "chunk_strategy": chunk_strategy,
    }
    ensure_index(chunk_strategy, config)
    with st.spinner("Retrieving and generating..."):
        resp = answer_question(user_input, retrieval_overrides=overrides)
    st.session_state["history"].append((user_input, resp))

for question, resp in st.session_state["history"]:
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(resp["answer"])

    with st.expander("Retrieved contexts"):
        for ctx in resp.get("contexts", []):
            st.markdown(
                f"**{ctx['title']}** `[{ctx['doc_id']}:{ctx['chunk_id']}]` (score={ctx['score']:.3f})\n\n{ctx['text']}"
            )