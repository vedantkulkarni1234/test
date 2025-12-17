"""Streamlit "Glass Box" UI for visualizing the A* retrieval process.

Run:
    streamlit run glass_box_app.py

This UI animates how the A* retriever explores candidate document chunks.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from astar_retriever import AStarRetriever
from config import config
from embedding_manager import EmbeddingManager


def _build_figure(
    graph: Dict[str, Any],
    pos: Dict[str, Tuple[float, float]],
    visited: Set[str],
    current: Optional[str],
    selected: Set[str],
    highlight_edge: Optional[Tuple[str, str]] = None,
    highlight_edge_label: Optional[str] = None,
) -> go.Figure:
    node_by_id = {n["id"]: n for n in graph.get("nodes", [])}

    edge_x: List[float] = []
    edge_y: List[float] = []

    for edge in graph.get("edges", []):
        src = edge["source"]
        dst = edge["target"]
        if src not in pos or dst not in pos:
            continue

        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="rgba(120,120,120,0.35)"),
            hoverinfo="none",
            mode="lines",
            name="semantic links",
        )
    )

    if highlight_edge and highlight_edge[0] in pos and highlight_edge[1] in pos:
        x0, y0 = pos[highlight_edge[0]]
        x1, y1 = pos[highlight_edge[1]]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                line=dict(width=4, color="rgba(255,0,0,0.7)"),
                mode="lines",
                hoverinfo="text",
                text=[highlight_edge_label or "", highlight_edge_label or ""],
                name="latest expansion",
            )
        )

    node_x: List[float] = []
    node_y: List[float] = []
    node_text: List[str] = []
    node_color: List[str] = []
    node_size: List[int] = []

    for node_id, (x, y) in pos.items():
        node = node_by_id.get(node_id, {"id": node_id, "label": node_id})

        label = node.get("label") or node_id
        sim = node.get("similarity", 0.0)
        rel = node.get("relevance_score", 0.0)
        preview = node.get("text_preview", "")

        node_x.append(x)
        node_y.append(y)
        node_text.append(
            f"<b>{label}</b><br>chunk_id: {node_id}<br>similarity: {sim:.3f}<br>relevance: {rel:.3f}<br><br>{preview}"
        )

        if node_id == current:
            node_color.append("rgba(255,0,0,0.9)")
            node_size.append(18)
        elif node_id in selected:
            node_color.append("rgba(0,160,0,0.85)")
            node_size.append(14)
        elif node_id in visited:
            node_color.append("rgba(255,165,0,0.75)")
            node_size.append(11)
        else:
            node_color.append("rgba(160,160,160,0.55)")
            node_size.append(9)

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color="rgba(50,50,50,0.4)"),
            ),
            name="documents",
        )
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=650,
    )

    return fig


@st.cache_resource
def _load_embedding_manager(cache_dir: str) -> EmbeddingManager:
    return EmbeddingManager(Path(cache_dir))


def main() -> None:
    st.set_page_config(page_title="Glass Box A* Visualizer", layout="wide")
    st.title("Glass Box: Visualize the A* Document Search")

    with st.sidebar:
        st.header("Search Settings")
        cache_dir = st.text_input("Embeddings cache directory", value=str(config.EMBEDDING_CACHE_PATH))
        top_k = st.slider("Target documents (top_k)", min_value=1, max_value=10, value=int(config.TOP_K_DOCUMENTS))
        relevance_threshold = st.slider("Candidate similarity threshold", 0.0, 0.99, float(config.RELEVANCE_THRESHOLD))
        max_depth = st.slider("Max search depth", 1, 20, int(config.MAX_SEARCH_DEPTH))
        max_steps = st.slider("Max animation steps", 25, 500, 150)
        delay_ms = st.slider("Animation delay (ms)", 0, 500, 80)

        st.caption(
            "Tip: this search explores combinations of chunks; highlighted nodes represent the most recently expanded chunk set."
        )

    query = st.text_input("Query", value="")

    run = st.button("Run & Animate", type="primary", disabled=not query.strip())

    if not run:
        st.info("Enter a query and click **Run & Animate**.")
        return

    try:
        embedding_manager = _load_embedding_manager(cache_dir)
    except Exception as e:
        st.error(f"Failed to load embeddings index from {cache_dir}: {e}")
        return

    retriever = AStarRetriever(
        embedding_manager.vector_index,
        {
            "heuristic_weight": float(config.ASTAR_HEURISTIC_WEIGHT),
            "max_depth": int(max_depth),
            "relevance_threshold": float(relevance_threshold),
            "top_k": int(top_k),
        },
    )

    query_embedding = embedding_manager.generator.generate_embedding(query)

    graph_placeholder = st.empty()
    details_col, results_col = st.columns([0.35, 0.65], gap="large")

    with details_col:
        st.subheader("Live decision log")
        decision_placeholder = st.empty()
        stats_placeholder = st.empty()

    with results_col:
        st.subheader("Final selected documents")
        results_placeholder = st.empty()

    graph_data: Optional[Dict[str, Any]] = None
    pos: Dict[str, Tuple[float, float]] = {}

    visited: Set[str] = set()
    current: Optional[str] = None
    selected: Set[str] = set()

    highlight_edge: Optional[Tuple[str, str]] = None
    highlight_label: Optional[str] = None

    final_results: List[Dict[str, Any]] = []

    for event in retriever.search_stream(query, query_embedding, top_k=top_k, max_steps=max_steps):
        if event["type"] == "graph":
            graph_data = event["graph"]

            g_nx = nx.Graph()
            for node in graph_data.get("nodes", []):
                g_nx.add_node(node["id"])
            for edge in graph_data.get("edges", []):
                g_nx.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 0.0))

            pos = nx.spring_layout(g_nx, seed=42)

            fig = _build_figure(graph_data, pos, visited, current, selected)
            graph_placeholder.plotly_chart(fig, use_container_width=True)
            continue

        if event["type"] == "expand":
            current = event.get("expanded_chunk_id")
            if current:
                visited.add(current)

        if event["type"] == "enqueue":
            src = event.get("expanded_chunk_id")
            dst = event.get("added_chunk_id")
            if src and dst:
                highlight_edge = (src, dst)
                conn = event.get("connected_to") or []
                cohesion = event.get("cohesion_bonus")
                highlight_label = (
                    f"added {dst}\ncohesion_bonus={cohesion:.3f}" if isinstance(cohesion, (int, float)) else f"added {dst}"
                )

                decision_placeholder.json(
                    {
                        "step": event.get("step"),
                        "expanded": src,
                        "added": dst,
                        "connected_to": conn,
                        "additional_relevance": event.get("additional_relevance"),
                        "cohesion_bonus": cohesion,
                        "state_size": len(event.get("state_documents", [])),
                    }
                )

        if event["type"] == "result":
            final_results = event.get("results", [])
            selected = {r["chunk_id"] for r in final_results}

            stats_placeholder.json(
                {
                    "visited_states": event.get("visited_states"),
                    "final_path": event.get("final_path"),
                    "final_document_set": event.get("final_document_set"),
                }
            )

            results_placeholder.write(final_results)

        if graph_data and pos:
            fig = _build_figure(
                graph_data,
                pos,
                visited,
                current,
                selected,
                highlight_edge=highlight_edge,
                highlight_edge_label=highlight_label,
            )
            graph_placeholder.plotly_chart(fig, use_container_width=True)

        if delay_ms:
            time.sleep(delay_ms / 1000.0)

    if final_results:
        st.divider()
        st.subheader("Why were these chosen?")
        st.markdown(
            "The graph shows semantic links between candidate chunks (edges). "
            "Nodes light up as the algorithm expands states. The final selected set is highlighted in green."
        )


if __name__ == "__main__":
    main()
