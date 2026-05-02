"""Streamlit UI for Hybrid RAG (academic-style demo)."""

import os
from typing import Optional

import streamlit as st
import requests
from streamlit_agraph import agraph, Node, Edge, Config

API_URL = os.getenv("API_URL", "http://api:8000")


METHOD_LABELS = {
    "hybrid": "Гибридный (вектор + граф)",
    "vector": "Векторный (семантический поиск)",
    "graph": "Графовый (мультихоп по связям)",
}

METHOD_DESCRIPTIONS = {
    "hybrid": (
        "Объединение результатов плотного векторного поиска и обхода графа знаний "
        "с последующей дедупликацией."
    ),
    "vector": (
        "Поиск по плотным эмбеддингам в ChromaDB. Подходит для однохоповых "
        "фактических запросов."
    ),
    "graph": (
        "Поиск по графу знаний, построенному с помощью LlamaIndex PropertyGraphIndex. "
        "Подходит для запросов, требующих рассуждения по нескольким связям."
    ),
}


CUSTOM_CSS = """
<style>
:root {
    --serif: "Charter", "PT Serif", "Georgia", "Times New Roman", serif;
    --sans:  "Inter", "Helvetica Neue", "Arial", sans-serif;
    --ink:   #1f2933;
    --muted: #52606d;
    --rule:  #cbd2d9;
    --accent:#1a365d;
    --soft:  #f7f7f4;
}

html, body, [class*="stMarkdown"], [class*="stText"] {
    color: var(--ink);
}

h1, h2, h3, h4 {
    font-family: var(--serif) !important;
    color: var(--accent) !important;
    letter-spacing: 0.005em;
}

h1 { font-weight: 700 !important; }
h2 { font-weight: 600 !important; }

.block-container {
    padding-top: 2.2rem;
    max-width: 1100px;
}

.app-header {
    border-bottom: 1px solid var(--rule);
    padding-bottom: 0.9rem;
    margin-bottom: 1.4rem;
}
.app-header .title {
    font-family: var(--serif);
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0;
}
.app-header .subtitle {
    font-family: var(--serif);
    font-style: italic;
    font-size: 1.02rem;
    color: var(--muted);
    margin-top: 0.15rem;
}
.app-header .meta {
    font-family: var(--sans);
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 0.4rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.method-card {
    border-left: 3px solid var(--accent);
    background: var(--soft);
    padding: 0.55rem 0.85rem;
    margin: 0.5rem 0 0.9rem 0;
    font-size: 0.88rem;
    color: var(--muted);
    font-family: var(--sans);
}

.source-card {
    border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    padding: 0.85rem 1rem;
    margin: 0.6rem 0;
    background: #fff;
    font-family: var(--sans);
    font-size: 0.92rem;
}
.source-card .source-topic {
    font-family: var(--serif);
    font-weight: 600;
    font-size: 1.02rem;
    color: var(--accent);
    margin-bottom: 0.15rem;
}
.source-card .source-title {
    font-family: var(--serif);
    font-style: italic;
    color: var(--ink);
    margin-bottom: 0.25rem;
    line-height: 1.35;
}
.source-card .source-authors {
    color: var(--muted);
    margin-bottom: 0.25rem;
}
.source-card .source-citation {
    color: var(--muted);
    font-size: 0.84rem;
    border-top: 1px dotted var(--rule);
    padding-top: 0.35rem;
    margin-top: 0.35rem;
    line-height: 1.4;
}
.source-card .source-meta {
    font-family: var(--sans);
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.source-card pre {
    background: var(--soft);
    border: 1px solid var(--rule);
    padding: 0.6rem 0.7rem;
    font-size: 0.84rem;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--ink);
    margin-top: 0.5rem;
}

.status-row {
    display: flex;
    justify-content: space-between;
    font-family: var(--sans);
    font-size: 0.88rem;
    padding: 0.25rem 0;
    border-bottom: 1px dotted var(--rule);
}
.status-row .ok    { color: #2f6f3e; font-weight: 600; }
.status-row .warn  { color: #8a6d1c; font-weight: 600; }
.status-row .err   { color: #8a1f1f; font-weight: 600; }

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted) !important;
    border-bottom: 1px solid var(--rule);
    padding-bottom: 0.25rem;
    margin-top: 1rem !important;
}

section[data-testid="stSidebar"] {
    background: #fbfbf8;
    border-right: 1px solid var(--rule);
}

.about-box {
    font-family: var(--sans);
    font-size: 0.85rem;
    color: var(--muted);
    line-height: 1.45;
}
.about-box b { color: var(--ink); }
</style>
"""


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "method" not in st.session_state:
        st.session_state.method = "hybrid"


def get_status() -> Optional[dict]:
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        return response.json()
    except Exception:
        return None


def query_rag(query: str, method: str) -> Optional[dict]:
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query, "method": method},
            timeout=120,
        )
        return response.json()
    except Exception as e:
        st.error(f"Сервис недоступен: {e}")
        return None


def get_graph_data(limit: int = 100) -> Optional[dict]:
    try:
        response = requests.get(
            f"{API_URL}/graph/data",
            params={"limit": limit},
            timeout=10,
        )
        return response.json()
    except Exception as e:
        st.error(f"Ошибка графа: {e}")
        return None


def render_app_header(subtitle: str):
    st.markdown(
        f"""
        <div class="app-header">
            <p class="title">Гибридная система информационного поиска</p>
            <p class="subtitle">{subtitle}</p>
            <p class="meta">Выпускная квалификационная работа &middot; Демонстрационный стенд</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_block(status: Optional[dict]):
    if status is None:
        st.markdown(
            '<div class="status-row"><span>API</span><span class="err">недоступен</span></div>',
            unsafe_allow_html=True,
        )
        return

    vec_state = (
        f'<span class="ok">проиндексирован &middot; {status["vector_count"]} фрагментов</span>'
        if status.get("vector_indexed")
        else '<span class="warn">не построен</span>'
    )
    graph_state = (
        '<span class="ok">проиндексирован</span>'
        if status.get("graph_indexed")
        else '<span class="warn">не построен</span>'
    )
    st.markdown(
        f"""
        <div class="status-row"><span>Векторный индекс</span>{vec_state}</div>
        <div class="status-row"><span>Граф знаний</span>{graph_state}</div>
        """,
        unsafe_allow_html=True,
    )


def _esc(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_source_card(idx: int, src: dict):
    meta = src.get("metadata") or {}
    topic = meta.get("topic") or ""
    display_title = meta.get("display_title") or ""
    authors = meta.get("authors") or meta.get("author") or ""
    citation = meta.get("citation") or ""
    score = src.get("score", 0.0)
    method_used = src.get("source", "")
    snippet = (src.get("content") or "").strip()

    method_ru = {"vector": "вектор", "graph": "граф"}.get(method_used, method_used)
    header_meta = (
        f"Источник №{idx} &middot; релевантность {score:.3f} &middot; ретривер: {method_ru}"
    )

    parts = [f'<div class="source-meta">{header_meta}</div>']
    if topic:
        parts.append(f'<div class="source-topic">{_esc(topic)}</div>')
    if display_title and display_title != topic:
        parts.append(f'<div class="source-title">{_esc(display_title)}</div>')
    if authors:
        parts.append(f'<div class="source-authors">Авторы: {_esc(authors)}</div>')
    if snippet:
        max_len = 600
        shown = snippet[:max_len] + ("…" if len(snippet) > max_len else "")
        parts.append(f"<pre>{_esc(shown)}</pre>")
    if citation and citation != display_title:
        parts.append(f'<div class="source-citation">{_esc(citation)}</div>')

    st.markdown(
        f'<div class="source-card">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"Источники ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            render_source_card(i, src)


def render_about_block():
    st.markdown(
        """
        <div class="about-box">
        <b>Корпус.</b> Научные работы по диагональным латинским квадратам и
        смежным комбинаторным задачам.<br/><br/>
        <b>Метод.</b> Гибридный ретривер, объединяющий плотный векторный
        поиск (ChromaDB) и обход графа знаний (LlamaIndex
        PropertyGraphIndex с динамическим извлечением триплетов).<br/><br/>
        <b>Генерация.</b> LLM поверх извлечённого контекста, цитирующая
        источники по теме и авторам.
        </div>
        """,
        unsafe_allow_html=True,
    )


ENTITY_TYPE_COLORS = {
    "LATIN_SQUARE_TYPE":      "#1a365d",
    "DIAGONAL_LATIN_SQUARE":  "#1a365d",
    "COMBINATORIAL_OBJECT":   "#2c5282",
    "ORTHOGONALITY_OBJECT":   "#2b6cb0",
    "ORTHOGONITY_OBJECT":     "#2b6cb0",
    "TRANSVERSAL_OBJECT":     "#3182ce",
    "NUMERICAL_CHARACTERISTIC": "#805ad5",
    "ENUMERATION_SEQUENCE":   "#6b46c1",
    "ALGORITHM":              "#c05621",
    "COMPUTATIONAL_PROBLEM":  "#9c4221",
    "COMPUTING_PLATFORM":     "#b7791f",
    "CONSTRAINT":             "#718096",
    "REDUCTION_TARGET":       "#4a5568",
    "THEORETICAL_RESULT":     "#2f855a",
}
DEFAULT_ENTITY_COLOR = "#4a5568"


def render_graph_page():
    render_app_header("Визуализация графа знаний")

    col_left, col_right = st.columns([3, 1])
    with col_right:
        limit = st.slider("Количество триплетов", 10, 500, 150, 10)
        load = st.button("Построить", use_container_width=True)

    if load:
        with st.spinner("Загрузка структуры графа..."):
            data = get_graph_data(limit=limit)

        if data and data.get("nodes"):
            with col_left:
                st.markdown(
                    f"<p style='font-family:var(--sans);color:var(--muted);"
                    f"font-size:0.88rem;margin-top:0.4rem;'>"
                    f"Отображено вершин: {len(data['nodes'])} &middot; "
                    f"рёбер: {len(data['edges'])}</p>",
                    unsafe_allow_html=True,
                )

            present_types = sorted({
                (n.get("properties") or {}).get("type") or ""
                for n in data["nodes"]
            } - {""})
            if present_types:
                legend_items = "".join(
                    f"<span style='display:inline-flex;align-items:center;"
                    f"margin-right:0.9rem;font-family:var(--sans);"
                    f"font-size:0.78rem;color:var(--muted);'>"
                    f"<span style='display:inline-block;width:10px;height:10px;"
                    f"border-radius:50%;background:{ENTITY_TYPE_COLORS.get(t, DEFAULT_ENTITY_COLOR)};"
                    f"margin-right:0.35rem;'></span>{t}</span>"
                    for t in present_types
                )
                st.markdown(
                    f"<div style='margin:0.4rem 0 0.6rem 0;'>{legend_items}</div>",
                    unsafe_allow_html=True,
                )

            nodes = [
                Node(
                    id=node_data["id"],
                    label=node_data["label"][:60],
                    size=20,
                    color=ENTITY_TYPE_COLORS.get(
                        (node_data.get("properties") or {}).get("type"),
                        DEFAULT_ENTITY_COLOR,
                    ),
                )
                for node_data in data["nodes"]
            ]
            edges = [
                Edge(
                    source=edge_data["source"],
                    target=edge_data["target"],
                    label=edge_data["label"],
                    color="#a0aec0",
                )
                for edge_data in data["edges"]
            ]
            config = Config(
                width=1100,
                height=720,
                directed=True,
                physics=True,
                hierarchical=False,
            )
            agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.warning("Граф не загружен или пуст. Проверьте логи API.")


def render_chat_page():
    render_app_header("Векторно-графовый ретривер для научной литературы")

    method = st.session_state.method
    st.markdown(
        f'<div class="method-card"><b>{METHOD_LABELS[method]}.</b> '
        f"{METHOD_DESCRIPTIONS[method]}</div>",
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"])

    if prompt := st.chat_input("Введите вопрос по корпусу..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Извлечение контекста и генерация ответа..."):
                result = query_rag(prompt, st.session_state.method)

            if result:
                st.markdown(result["answer"])
                render_sources(result.get("sources") or [])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources") or [],
                })
            else:
                st.error("Не удалось получить ответ от сервиса.")


def main():
    st.set_page_config(
        page_title="Гибридная система информационного поиска",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_session_state()

    with st.sidebar:
        st.markdown("### Раздел")
        page = st.radio(
            "Раздел",
            ["Диалог", "Граф знаний"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("### Стратегия поиска")
        method = st.selectbox(
            "Метод",
            options=list(METHOD_LABELS.keys()),
            format_func=lambda k: METHOD_LABELS[k],
            index=list(METHOD_LABELS.keys()).index(st.session_state.method),
            label_visibility="collapsed",
        )
        st.session_state.method = method

        st.markdown("### Состояние системы")
        render_status_block(get_status())

        st.markdown("### О системе")
        render_about_block()

        st.markdown("### Управление")
        if st.button("Очистить диалог", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if page == "Граф знаний":
        render_graph_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()
