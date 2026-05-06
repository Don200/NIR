"""Streamlit UI for Hybrid RAG (academic-style demo)."""

import json
import os
import re
import tempfile
from collections import OrderedDict
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
import requests
from pyvis.network import Network

API_URL = os.getenv("API_URL", "http://api:8000")

PREVIEW_CHARS = 240


METHOD_LABELS = {
    "hybrid": "Гибридный (вектор + граф)",
    "vector": "Векторный поиск",
    "graph": "Поиск по графу знаний",
}

METHOD_DESCRIPTIONS = {
    "hybrid": (
        "Совмещает векторный поиск и обход графа знаний, "
        "убирая повторяющиеся фрагменты."
    ),
    "vector": (
        "Семантический поиск по векторам ChromaDB. "
        "Хорошо работает для прямых фактических вопросов."
    ),
    "graph": (
        "Обход графа знаний (LlamaIndex PropertyGraphIndex). "
        "Полезен, когда ответ требует пройти по нескольким связям между понятиями."
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

.source-topic {
    font-family: var(--serif);
    font-weight: 600;
    font-size: 1.02rem;
    color: var(--accent);
    margin-bottom: 0.15rem;
}
.source-title {
    font-family: var(--serif);
    font-style: italic;
    color: var(--ink);
    margin-bottom: 0.25rem;
    line-height: 1.35;
}
.source-authors {
    color: var(--muted);
    margin-bottom: 0.25rem;
    font-family: var(--sans);
    font-size: 0.88rem;
}
.source-file {
    margin: 0.15rem 0 0.35rem 0;
    font-size: 0.78rem;
    color: var(--muted);
}
.source-file code {
    background: var(--soft);
    border: 1px solid var(--rule);
    padding: 0.05rem 0.35rem;
    border-radius: 3px;
    font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
    font-size: 0.78rem;
    color: var(--ink);
}
.source-citation {
    color: var(--muted);
    font-size: 0.84rem;
    border-top: 1px dotted var(--rule);
    padding-top: 0.35rem;
    margin-top: 0.5rem;
    line-height: 1.4;
    font-family: var(--sans);
}
.source-meta {
    font-family: var(--sans);
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
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

.triplet-block {
    background: var(--soft);
    border-left: 3px solid var(--accent);
    padding: 0.55rem 0.85rem;
    margin: 0.35rem 0 0.7rem 0;
    font-family: var(--sans);
    font-size: 0.85rem;
    line-height: 1.55;
}
.triplet-subject {
    font-weight: 700;
    color: var(--accent);
    margin: 0.25rem 0 0.15rem 0;
    font-size: 0.9rem;
}
.triplet-subject:first-child { margin-top: 0; }
.triplet-list {
    list-style: none;
    padding-left: 0.7rem;
    margin: 0;
}
.triplet-list li {
    margin: 0.15rem 0;
    color: var(--ink);
}
.rel-chip {
    display: inline-block;
    padding: 0.02rem 0.4rem;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-right: 0.45rem;
    color: white;
    vertical-align: 1px;
    font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
}
.rel-isa     { background: #2c5282; }
.rel-def     { background: #2f855a; }
.rel-prop    { background: #b7791f; }
.rel-count   { background: #6b46c1; }
.rel-method  { background: #c05621; }
.rel-platform{ background: #319795; }
.rel-rel     { background: #718096; }
.rel-default { background: #4a5568; }
.triplet-toggle-row {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.3rem;
    font-family: var(--sans);
}
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
            <p class="title">Гибридный поиск по научным работам</p>
            <p class="subtitle">{subtitle}</p>
            <p class="meta">Выпускная квалификационная работа</p>
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


def _make_preview(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    """Cut text near a word boundary so we don't end mid-token."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars // 2:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


_FACTS_HEADER = "Here are some facts extracted from the provided text:"

# subj — anything non-greedy; rel — UPPERCASE_WITH_UNDERSCORES; obj — anything until
# the next subj-rel-obj triple or end of line.
_TRIPLET_RE = re.compile(
    r"(\S.*?)\s*->\s*([A-Z][A-Z_0-9]+)\s*->\s*(.+?)"
    r"(?=\s+\S.*?\s*->\s*[A-Z][A-Z_0-9]+\s*->|\s*$)"
)

_RELATION_CSS = {
    "IS_A": "rel-isa",
    "SPECIAL_CASE_OF": "rel-isa",
    "GENERALIZES": "rel-isa",
    "EQUIVALENT_TO": "rel-isa",
    "DEFINED_AS": "rel-def",
    "HAS_PROPERTY": "rel-prop",
    "HAS_CONSTRAINT": "rel-prop",
    "CONSTRAINED_BY": "rel-prop",
    "COUNTED_BY": "rel-count",
    "MEASURED_BY": "rel-count",
    "HAS_ENUMERATION_SEQUENCE": "rel-count",
    "HAS_TRANSVERSAL_COUNT": "rel-count",
    "HAS_TRANSVERSAL": "rel-count",
    "USES_METHOD": "rel-method",
    "SOLVED_BY": "rel-method",
    "REDUCED_TO": "rel-method",
    "USED_IN": "rel-method",
    "USED_FOR": "rel-method",
    "OPTIMIZED_BY": "rel-method",
    "COMPUTED_BY": "rel-method",
    "REPLACES": "rel-method",
    "COMPUTED_ON": "rel-platform",
    "APPLIED_ON_LEVELS": "rel-platform",
    "ORTHOGONAL_TO": "rel-rel",
    "PROVES_CONDITION_FOR": "rel-rel",
    "REQUIRES": "rel-rel",
    "IMPROVES_PERFORMANCE_OF": "rel-rel",
    "ENCODES": "rel-rel",
    "ENCODED_BY": "rel-rel",
    "HAS_COMPLEXITY": "rel-rel",
}


def _parse_facts_block(content: str) -> tuple[list[tuple[str, str, str]], str]:
    """Split content into (triplets, body_markdown).

    LlamaIndex graph retriever prepends a "Here are some facts extracted from
    the provided text:" header followed by triplet lines, then the source
    chunk text. We separate them so the UI can render triplets structurally
    and the body as markdown (formulas, tables) without the triplet noise.
    """
    if _FACTS_HEADER not in content:
        return [], content

    _, _, rest = content.partition(_FACTS_HEADER)
    rest = rest.lstrip("\n ").rstrip()

    parts = re.split(r"\n\s*\n", rest, maxsplit=1)
    triplet_text = parts[0]
    body = parts[1].strip() if len(parts) > 1 else ""

    seen: "OrderedDict[tuple[str, str, str], None]" = OrderedDict()
    for line in triplet_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for m in _TRIPLET_RE.finditer(line):
            subj = m.group(1).strip()
            rel = m.group(2).strip()
            obj = m.group(3).strip().rstrip(",;.")
            key = (subj, rel, obj)
            if key not in seen:
                seen[key] = None

    return list(seen.keys()), body


_RELATION_EDGE_COLOR = {
    "rel-isa":      "#2c5282",
    "rel-def":      "#2f855a",
    "rel-prop":     "#b7791f",
    "rel-count":    "#6b46c1",
    "rel-method":   "#c05621",
    "rel-platform": "#319795",
    "rel-rel":      "#718096",
    "rel-default":  "#4a5568",
}


def _wrap_label(text: str, width: int = 18, max_lines: int = 3) -> str:
    """Soft-wrap long entity labels for graphviz nodes."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
            if len(lines) == max_lines - 1:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    out = "\n".join(lines)
    consumed = sum(len(line.split()) for line in lines)
    if consumed < len(words):
        out += " …"
    return out


def _dot_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _render_triplet_dot(triplets: list[tuple[str, str, str]]) -> str:
    """Build graphviz DOT for triplet subgraph rendering via st.graphviz_chart."""
    nodes: "OrderedDict[str, str]" = OrderedDict()  # id -> wrapped label
    edges: list[tuple[str, str, str, str]] = []  # (src_id, tgt_id, rel, color)

    for subj, rel, obj in triplets:
        nodes.setdefault(subj, _wrap_label(subj))
        nodes.setdefault(obj, _wrap_label(obj))
        css = _RELATION_CSS.get(rel, "rel-default")
        color = _RELATION_EDGE_COLOR[css]
        edges.append((subj, obj, rel, color))

    # subjects (nodes that appear on the left side of any triplet) get accent fill
    subject_ids = {s for s, _, _ in triplets}

    lines = [
        'digraph G {',
        '  rankdir=LR;',
        '  bgcolor="transparent";',
        '  pad="0.15,0.1"; nodesep=0.18; ranksep=0.45;',
        '  margin=0;',
        '  node [shape=box, style="rounded,filled", fontname="Inter, Helvetica", '
        'fontsize=9, color="#cbd2d9", fillcolor="#ffffff", margin="0.08,0.04", '
        'height=0.3];',
        '  edge [fontname="Inter, Helvetica", fontsize=8, arrowsize=0.6, '
        'penwidth=0.9];',
    ]
    for nid, label in nodes.items():
        if nid in subject_ids:
            fill = "#e6edf5"
            border = "#1a365d"
            fontcolor = "#1a365d"
            attrs = (
                f'label="{_dot_escape(label)}", '
                f'fillcolor="{fill}", color="{border}", fontcolor="{fontcolor}", '
                f'penwidth=1.4'
            )
        else:
            attrs = f'label="{_dot_escape(label)}"'
        lines.append(f'  "{_dot_escape(nid)}" [{attrs}];')

    for src, tgt, rel, color in edges:
        lines.append(
            f'  "{_dot_escape(src)}" -> "{_dot_escape(tgt)}" '
            f'[label="{_dot_escape(rel)}", color="{color}", fontcolor="{color}"];'
        )

    lines.append('}')
    return "\n".join(lines)


def render_source_card(idx: int, src: dict, key_prefix: str):
    meta = src.get("metadata") or {}
    topic = meta.get("topic") or ""
    display_title = meta.get("display_title") or ""
    authors = meta.get("authors") or meta.get("author") or ""
    citation = meta.get("citation") or ""
    file_name = (
        meta.get("doc_title")
        or meta.get("file_name")
        or meta.get("title")
        or ""
    )
    chunk_index = meta.get("chunk_index")
    score = src.get("score", 0.0)
    method_used = src.get("source", "")
    content = (src.get("content") or "").strip()

    method_ru = {"vector": "вектор", "graph": "граф"}.get(method_used, method_used)
    header_bits = [
        f"Источник №{idx}",
        f"релевантность {score:.3f}",
        f"найден через: {method_ru}",
    ]
    if chunk_index is not None:
        header_bits.append(f"чанк #{chunk_index}")
    header_meta = " &middot; ".join(header_bits)

    with st.container(border=True):
        st.markdown(
            f'<div class="source-meta">{header_meta}</div>',
            unsafe_allow_html=True,
        )
        if topic:
            st.markdown(
                f'<div class="source-topic">{_esc(topic)}</div>',
                unsafe_allow_html=True,
            )
        if display_title and display_title != topic:
            st.markdown(
                f'<div class="source-title">{_esc(display_title)}</div>',
                unsafe_allow_html=True,
            )
        if authors:
            st.markdown(
                f'<div class="source-authors">Авторы: {_esc(authors)}</div>',
                unsafe_allow_html=True,
            )
        if file_name:
            st.markdown(
                f'<div class="source-file"><code>{_esc(file_name)}</code></div>',
                unsafe_allow_html=True,
            )

        if citation and citation != display_title:
            st.markdown(
                f'<div class="source-citation">{_esc(citation)}</div>',
                unsafe_allow_html=True,
            )

        triplets, body = _parse_facts_block(content)
        body_text = body if body else (content if not triplets else "")

        expander_bits = []
        if triplets:
            expander_bits.append(
                f"подграф знаний ({len(triplets)} "
                f"{'факт' if len(triplets) == 1 else 'фактов'})"
            )
        if body_text:
            expander_bits.append("текст фрагмента")
        if expander_bits:
            label = "Раскрыть: " + " + ".join(expander_bits)
            with st.expander(label, expanded=False):
                if triplets:
                    st.graphviz_chart(
                        _render_triplet_dot(triplets),
                        use_container_width=True,
                    )
                if body_text:
                    st.markdown(body_text)


def render_sources(sources: list[dict], key_prefix: str):
    if not sources:
        return
    show = st.toggle(
        f"Источники ({len(sources)})",
        value=False,
        key=f"{key_prefix}_show_sources",
    )
    if show:
        for i, src in enumerate(sources, 1):
            render_source_card(i, src, f"{key_prefix}_src{i}")


def render_about_block():
    st.markdown(
        """
        <div class="about-box">
        <b>Корпус.</b> Научные работы по диагональным латинским квадратам и
        смежным комбинаторным задачам.<br/><br/>
        <b>Поиск.</b> Совмещает векторный поиск по ChromaDB и обход графа знаний,
        собранного из триплетов (LlamaIndex PropertyGraphIndex).<br/><br/>
        <b>Ответ.</b> Формирует языковая модель по найденному контексту,
        со ссылками на работы по теме и авторам.
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


def _node_color(node: dict) -> str:
    return ENTITY_TYPE_COLORS.get(
        (node.get("properties") or {}).get("type"),
        DEFAULT_ENTITY_COLOR,
    )


def _node_tooltip(node: dict) -> str:
    props = node.get("properties") or {}
    parts = [f"<b>{node.get('label', node['id'])}</b>"]
    type_ = props.get("type")
    if type_:
        parts.append(f"<i>{type_}</i>")
    title = props.get("title")
    if title:
        parts.append(f"источник: {title}")
    return "<br>".join(parts)


def _edge_tooltip(edge: dict) -> str:
    props = edge.get("properties") or {}
    parts = [f"<b>{edge.get('label', '')}</b>"]
    parts.append(f"{edge['source']} → {edge['target']}")
    title = props.get("title")
    if title:
        parts.append(f"источник: {title}")
    return "<br>".join(parts)


PYVIS_OPTIONS = {
    "nodes": {
        "shape": "dot",
        "borderWidth": 2,
        "font": {
            "size": 14,
            "face": "Inter, Helvetica, Arial, sans-serif",
            "color": "#1f2933",
            "strokeWidth": 3,
            "strokeColor": "#ffffff",
        },
        "shadow": {"enabled": False},
    },
    "edges": {
        "color": {"color": "#a0aec0", "highlight": "#1a365d", "hover": "#2c5282"},
        "smooth": {"type": "continuous", "roundness": 0.2},
        "font": {
            "size": 11,
            "face": "Inter, Helvetica, Arial, sans-serif",
            "color": "#52606d",
            "strokeWidth": 0,
            "align": "middle",
        },
        "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
        "width": 1,
    },
    "interaction": {
        "hover": True,
        "tooltipDelay": 100,
        "navigationButtons": True,
        "keyboard": True,
        "hideEdgesOnDrag": True,
    },
    "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -55,
            "centralGravity": 0.01,
            "springLength": 120,
            "springConstant": 0.06,
            "damping": 0.6,
            "avoidOverlap": 0.5,
        },
        "stabilization": {"enabled": True, "iterations": 250, "fit": True},
        "minVelocity": 0.5,
    },
}


def build_pyvis_html(
    nodes: list[dict],
    edges: list[dict],
    height_px: int = 720,
) -> str:
    """Build standalone HTML for a Pyvis network."""
    net = Network(
        height=f"{height_px}px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#1f2933",
        notebook=False,
        cdn_resources="in_line",
    )

    for n in nodes:
        net.add_node(
            n["id"],
            label=(n.get("label") or n["id"])[:60],
            title=_node_tooltip(n),
            color=_node_color(n),
            size=18,
        )

    node_ids = {n["id"] for n in nodes}
    for e in edges:
        if e["source"] not in node_ids or e["target"] not in node_ids:
            continue
        net.add_edge(
            e["source"],
            e["target"],
            label=e.get("label", ""),
            title=_edge_tooltip(e),
        )

    net.set_options(json.dumps(PYVIS_OPTIONS))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        net.write_html(f.name, notebook=False, open_browser=False)
        path = f.name

    with open(path, encoding="utf-8") as f:
        html = f.read()
    try:
        os.remove(path)
    except OSError:
        pass
    return html


def filter_graph(
    data: dict,
    entity_types: list[str],
    relation_types: list[str],
    search: str,
) -> dict:
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []

    if entity_types:
        keep = set(entity_types)
        nodes = [
            n for n in nodes
            if (n.get("properties") or {}).get("type") in keep
        ]

    if search:
        q = search.lower()
        nodes = [
            n for n in nodes
            if q in (n.get("label") or "").lower()
            or q in (n.get("id") or "").lower()
        ]

    node_ids = {n["id"] for n in nodes}

    if relation_types:
        keep_rel = set(relation_types)
        edges = [e for e in edges if e.get("label") in keep_rel]

    edges = [
        e for e in edges
        if e["source"] in node_ids and e["target"] in node_ids
    ]

    if search or entity_types or relation_types:
        connected = {e["source"] for e in edges} | {e["target"] for e in edges}
        if connected:
            nodes = [n for n in nodes if n["id"] in connected]

    return {"nodes": nodes, "edges": edges}


def render_graph_legend(present_types: list[str]):
    if not present_types:
        return
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


@st.cache_data(show_spinner=False, ttl=60)
def cached_graph_data(limit: int) -> Optional[dict]:
    return get_graph_data(limit=limit)


def render_graph_page():
    render_app_header("Визуализация графа знаний")

    with st.sidebar:
        st.markdown("### Параметры графа")
        limit = st.slider("Сколько триплетов загрузить", 50, 800, 300, 50)
        height = st.slider("Высота холста, px", 500, 1100, 760, 20)

    data = cached_graph_data(limit)
    if not data or not data.get("nodes"):
        st.warning("Граф пуст или не загрузился. Проверьте, что API запущен.")
        return

    all_entity_types = sorted({
        (n.get("properties") or {}).get("type") or ""
        for n in data["nodes"]
    } - {""})
    all_relation_types = sorted({
        e.get("label") or "" for e in data["edges"]
    } - {""})

    col_a, col_b, col_c = st.columns([2, 2, 2])
    with col_a:
        entity_filter = st.multiselect(
            "Типы сущностей",
            options=all_entity_types,
            default=[],
            placeholder="все типы",
        )
    with col_b:
        relation_filter = st.multiselect(
            "Типы связей",
            options=all_relation_types,
            default=[],
            placeholder="все связи",
        )
    with col_c:
        search = st.text_input(
            "Поиск по имени сущности",
            value="",
            placeholder="например, DLS, transversal",
        )

    filtered = filter_graph(data, entity_filter, relation_filter, search)
    nodes, edges = filtered["nodes"], filtered["edges"]

    if not nodes:
        st.info("Под фильтр ничего не попало. Сбросьте параметры.")
        return

    st.markdown(
        f"<p style='font-family:var(--sans);color:var(--muted);"
        f"font-size:0.88rem;margin-top:0.2rem;'>"
        f"Вершин: <b style='color:var(--ink);'>{len(nodes)}</b> &middot; "
        f"рёбер: <b style='color:var(--ink);'>{len(edges)}</b> &middot; "
        f"всего в индексе: {len(data['nodes'])} вершин / {len(data['edges'])} триплетов</p>",
        unsafe_allow_html=True,
    )

    present_types = sorted({
        (n.get("properties") or {}).get("type") or ""
        for n in nodes
    } - {""})
    render_graph_legend(present_types)

    html = build_pyvis_html(nodes, edges, height_px=height)
    components.html(html, height=height + 30, scrolling=False)

    with st.expander("Экспорт", expanded=False):
        col_x, col_y = st.columns(2)
        with col_x:
            st.download_button(
                "Скачать HTML графа",
                data=html.encode("utf-8"),
                file_name="hybrid_rag_graph.html",
                mime="text/html",
                use_container_width=True,
            )
        with col_y:
            st.download_button(
                "Скачать GEXF (Gephi)",
                data=build_gexf(nodes, edges).encode("utf-8"),
                file_name="hybrid_rag_graph.gexf",
                mime="application/xml",
                use_container_width=True,
            )


def build_gexf(nodes: list[dict], edges: list[dict]) -> str:
    """Render a minimal GEXF document so the graph can be opened in Gephi."""
    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    id_index = {n["id"]: str(i) for i, n in enumerate(nodes)}
    node_xml = "\n".join(
        f'      <node id="{id_index[n["id"]]}" label="{esc(n.get("label") or n["id"])}">'
        f'<attvalues><attvalue for="0" value="{esc((n.get("properties") or {}).get("type") or "")}"/></attvalues>'
        f"</node>"
        for n in nodes
    )
    edge_xml = "\n".join(
        f'      <edge id="{i}" source="{id_index[e["source"]]}" '
        f'target="{id_index[e["target"]]}" label="{esc(e.get("label") or "")}"/>'
        for i, e in enumerate(edges)
        if e["source"] in id_index and e["target"] in id_index
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n'
        '  <graph mode="static" defaultedgetype="directed">\n'
        '    <attributes class="node">\n'
        '      <attribute id="0" title="entity_type" type="string"/>\n'
        "    </attributes>\n"
        f"    <nodes>\n{node_xml}\n    </nodes>\n"
        f"    <edges>\n{edge_xml}\n    </edges>\n"
        "  </graph>\n"
        "</gexf>\n"
    )


def render_chat_page():
    render_app_header("Вопросы по корпусу научных работ")

    method = st.session_state.method
    st.markdown(
        f'<div class="method-card"><b>{METHOD_LABELS[method]}.</b> '
        f"{METHOD_DESCRIPTIONS[method]}</div>",
        unsafe_allow_html=True,
    )

    for m_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"], f"msg{m_idx}")

    if prompt := st.chat_input("Введите вопрос по корпусу..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ..."):
                result = query_rag(prompt, st.session_state.method)

            if result:
                st.markdown(result["answer"])
                fresh_idx = len(st.session_state.messages)
                render_sources(result.get("sources") or [], f"msg{fresh_idx}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources") or [],
                })
            else:
                st.error("Не удалось получить ответ от сервиса.")


def main():
    st.set_page_config(
        page_title="Гибридный поиск по научным работам",
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

        st.markdown("### Метод поиска")
        method = st.selectbox(
            "Метод",
            options=list(METHOD_LABELS.keys()),
            format_func=lambda k: METHOD_LABELS[k],
            index=list(METHOD_LABELS.keys()).index(st.session_state.method),
            label_visibility="collapsed",
        )
        st.session_state.method = method

        st.markdown("### Индексы")
        render_status_block(get_status())

        st.markdown("### О работе")
        render_about_block()

        if st.button("Очистить диалог", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if page == "Граф знаний":
        render_graph_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()
