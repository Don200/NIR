"""Streamlit UI for Hybrid RAG."""

import os
from typing import Optional

import streamlit as st
import requests
from streamlit_agraph import agraph, Node, Edge, Config

API_URL = os.getenv("API_URL", "http://api:8000")


def init_session_state():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "method" not in st.session_state:
        st.session_state.method = "hybrid"


def get_status() -> Optional[dict]:
    """Get API status."""
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        return response.json()
    except Exception:
        return None


def query_rag(query: str, method: str) -> Optional[dict]:
    """Query the RAG API."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query, "method": method},
            timeout=60,
        )
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def get_graph_data(limit: int = 100) -> Optional[dict]:
    """Get graph data for visualization."""
    try:
        response = requests.get(
            f"{API_URL}/graph/data",
            params={"limit": limit},
            timeout=10,
        )
        return response.json()
    except Exception as e:
        st.error(f"Graph API Error: {e}")
        return None


def render_graph_page():
    """Render graph visualization page."""
    st.title("🕸️ Knowledge Graph Visualization")

    limit = st.slider("Number of nodes to display", 10, 200, 100, 10)

    if st.button("Load Graph"):
        with st.spinner("Loading graph data..."):
            data = get_graph_data(limit=limit)

        if data and data.get("nodes"):
            st.success(f"Loaded {len(data['nodes'])} nodes and {len(data['edges'])} edges")

            nodes = [
                Node(
                    id=node_data["id"],
                    label=node_data["label"][:50],
                    size=25,
                    color="#97C2FC",
                )
                for node_data in data["nodes"]
            ]

            edges = [
                Edge(
                    source=edge_data["source"],
                    target=edge_data["target"],
                    label=edge_data["label"],
                    color="#848484",
                )
                for edge_data in data["edges"]
            ]

            config = Config(
                width=1200,
                height=800,
                directed=True,
                physics=True,
                hierarchical=False,
            )

            agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.warning("No graph data available")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Hybrid RAG",
        page_icon="🔍",
        layout="wide",
    )

    init_session_state()

    # Navigation
    page = st.sidebar.radio("Navigation", ["💬 Chat", "🕸️ Graph"], index=0)

    if page == "🕸️ Graph":
        render_graph_page()
        return

    st.title("🔍 Hybrid RAG")
    st.caption("Vector + Graph retrieval system")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        method = st.selectbox(
            "Retrieval Method",
            options=["hybrid", "vector", "graph"],
            index=0,
            help="Choose retrieval strategy",
        )
        st.session_state.method = method

        st.divider()

        # Status
        st.header("System Status")
        status = get_status()

        if status:
            col1, col2 = st.columns(2)
            with col1:
                if status["vector_indexed"]:
                    st.success(f"Vector: {status['vector_count']} chunks")
                else:
                    st.warning("Vector: Not indexed")
            with col2:
                if status["graph_indexed"]:
                    st.success("Graph: Ready")
                else:
                    st.warning("Graph: Not indexed")
        else:
            st.error("API not available")

        st.divider()
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** (score: {src['score']:.3f}, via {src['source']})")
                        st.text(src["content"][:300] + "...")
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_rag(prompt, st.session_state.method)

            if result:
                st.markdown(result["answer"])

                # Show sources
                if result["sources"]:
                    with st.expander("View Sources"):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(f"**Source {i}** (score: {src['score']:.3f}, via {src['source']})")
                            st.text(src["content"][:300] + "...")
                            st.divider()

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            else:
                st.error("Failed to get response")


if __name__ == "__main__":
    main()
