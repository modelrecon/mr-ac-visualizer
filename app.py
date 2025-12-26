import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from cube_utils import load_activity_cube, build_heatmap_df

st.set_page_config(page_title="ActivityCube Visualizer", layout="wide")

st.title("ActivityCube Visualizer")

# --- Load ActivityCube JSON ---

path = st.text_input(
    "ActivityCube JSON path", "activity_cube.json"
)

if st.button("Load"):
    try:
        cube = load_activity_cube(path)
        st.success("Loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load JSON: {e}")
        cube = None
else:
    cube = None

if cube:
    st.subheader("Meta")
    st.json(cube["meta"])

    # --- Metric selector ---

    metrics = list(cube["layers"][0]["core_metrics"].keys())
    metric = st.selectbox("Select metric", metrics)

    # --- Heatmap ---

    st.subheader("Token x Layer Heatmap")
    df_heatmap = build_heatmap_df(cube, metric=metric)

    fig = px.imshow(
        df_heatmap.values,
        x=df_heatmap.columns,
        y=df_heatmap.index,
        labels={"x":"Token", "y":"Layer"},
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Token trace chart ---

    st.subheader("Token Metric Traces by Layer")

    fig2 = go.Figure()
    for token in cube["tokens"]:
        values = df_heatmap[token].values
        fig2.add_trace(go.Scatter(
            x=df_heatmap.index,
            y=values,
            mode="lines+markers",
            name=str(token)
        ))
    fig2.update_layout(
        xaxis_title="Layer",
        yaxis_title=f"{metric}"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Attention explorer (if exists) ---

    if any("attention_scores" in layer for layer in cube["layers"]):
        st.subheader("Attention Maps")

        layer_options = [
            layer["layer_index"]
            for layer in cube["layers"]
            if "attention_scores" in layer
        ]
        attn_layer = st.selectbox("Select Attention Layer", layer_options)

        attn_data = next(
            layer["attention_scores"]
            for layer in cube["layers"]
            if layer["layer_index"] == attn_layer
        )

        fig3 = px.imshow(
            attn_data,
            x=cube["tokens"], y=cube["tokens"],
            labels={"x":"Token (Key)", "y":"Token (Query)"},
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig3, use_container_width=True)
