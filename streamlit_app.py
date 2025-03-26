
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="STRIDE: 3D Readiness Surface Viewer", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("Expanded_Readiness_Spreadsheet.csv")

df = load_data()

st.title("📊 STRIDE: 3D Readiness Surface Viewer (Artificial Data)")
st.markdown("Explore linear vs curved surface relationships between readiness and contributing factors.")

x_col = st.selectbox("Select X-axis variable", df.columns[1:-1], index=0)
y_col = st.selectbox("Select Y-axis variable", df.columns[1:-1], index=1)
z_col = "Readiness"

# New toggle for surface type
surface_type = st.radio("Select surface type:", ["Linear 3D Surface", "Curved 3D Surface"])

# Prepare data
x = df[x_col].values
y = df[y_col].values
z = df[z_col].values
X = np.column_stack((x, y))

# Model selection
if surface_type == "Linear 3D Surface":
    model = LinearRegression()
    model.fit(X, z)
    z_pred = model.predict(X)
    rsq = model.score(X, z)
else:
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, z)
    z_pred = model.predict(X)
    rsq = model.score(X, z)

# Surface grid
grid_size = 30
x_range = np.linspace(x.min(), x.max(), grid_size)
y_range = np.linspace(y.min(), y.max(), grid_size)
xx, yy = np.meshgrid(x_range, y_range)
grid = np.column_stack((xx.ravel(), yy.ravel()))
zz = model.predict(grid).reshape(xx.shape)

# 3D plot
fig = go.Figure()

fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    colorscale="Viridis", opacity=0.8, showscale=True,
    contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
))

color_scale = np.interp(z, (min(z), max(z)), (0, 1))
colors = [f"rgb({int((1 - val) * 255)}, {int(val * 255)}, 150)" for val in color_scale]

fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers+lines",
    marker=dict(size=5, color=colors),
    line=dict(color="gray", width=1),
    hovertext=[f"{x_col}: {a:.1f}<br>{y_col}: {b:.1f}<br>{z_col}: {c:.1f}" for a, b, c in zip(x, y, z)],
    hoverinfo="text"
))

fig.update_layout(
    scene=dict(
        xaxis_title=x_col,
        yaxis_title=y_col,
        zaxis_title=z_col,
        xaxis=dict(showgrid=True, gridcolor="white"),
        yaxis=dict(showgrid=True, gridcolor="white"),
        zaxis=dict(showgrid=True, gridcolor="white")
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    height=700,
    title=f"3D Readiness Surface ({surface_type}) — R²: {rsq:.2f}"
)

st.plotly_chart(fig, use_container_width=True)
