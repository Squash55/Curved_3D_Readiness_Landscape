
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import combinations

st.set_page_config(page_title="3D Linear Readiness Surface", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Expanded_Readiness_Spreadsheet.csv")
    features = [
        "Mission Complexity", "Maintenance Burden", "Personnel Gaps", "Logistics Readiness",
        "Equipment Availability", "Cyber Resilience", "Fuel Supply Score", "Flight Ops Readiness",
        "Medical Support Score", "Training Level"
    ]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=features)
    return df, features

df, features = load_data()
y = np.random.normal(75, 10, len(df))

# Calculate R² for all feature pairs
pair_scores = []
for x_feat, y_feat in combinations(features, 2):
    model = LinearRegression()
    model.fit(df[[x_feat, y_feat]], y)
    r2 = r2_score(y, model.predict(df[[x_feat, y_feat]]))
    pair_scores.append((x_feat, y_feat, r2))

pair_scores = sorted(pair_scores, key=lambda x: x[2], reverse=True)
pair_dict = {f"{x} vs {y} (R²={r2:.2f})": (x, y) for x, y, r2 in pair_scores}

# Sidebar controls
st.sidebar.header("🧭 Feature Pair Selection")
selected_label = st.sidebar.selectbox("Select Feature Pair", list(pair_dict.keys()))
x_feature, y_feature = pair_dict[selected_label]
smoothness = st.sidebar.slider("Surface Smoothness", 10, 60, 30)

# Train model
X_pair = df[[x_feature, y_feature]]
model = LinearRegression()
model.fit(X_pair, y)
Z_pred = model.predict(X_pair)
r2_val = r2_score(y, Z_pred)

# Build surface grid
x_range = np.linspace(X_pair[x_feature].min(), X_pair[x_feature].max(), smoothness)
y_range = np.linspace(X_pair[y_feature].min(), X_pair[y_feature].max(), smoothness)
xx, yy = np.meshgrid(x_range, y_range)
grid_df = pd.DataFrame({x_feature: xx.ravel(), y_feature: yy.ravel()})
zz = model.predict(grid_df).reshape(xx.shape)

# Pin colors
colors = ['red' if val < 70 else 'blue' if val < 85 else 'green' for val in Z_pred]

# Plotly 3D plot
fig = go.Figure()

# Add linear surface
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.85, showscale=False))

# Scatter points
fig.add_trace(go.Scatter3d(
    x=X_pair[x_feature], y=X_pair[y_feature], z=Z_pred,
    mode='markers', marker=dict(size=5, color=colors),
    name="Bases", text=[f"{x_feature}: {a}, {y_feature}: {b}, Readiness: {c:.1f}"
                        for a, b, c in zip(X_pair[x_feature], X_pair[y_feature], Z_pred)],
    hoverinfo="text"
))

# Pin lines
for xi, yi, zi, color in zip(X_pair[x_feature], X_pair[y_feature], Z_pred, colors):
    fig.add_trace(go.Scatter3d(
        x=[xi, xi], y=[yi, yi], z=[0, zi],
        mode='lines', line=dict(color=color, width=2),
        showlegend=False
    ))

# Layout
fig.update_layout(
    title=f"3D Linear Readiness Surface: {x_feature} vs {y_feature} | R²={r2_val:.2f}",
    height=800,
    scene=dict(
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        zaxis_title="Predicted Readiness",
        xaxis=dict(showgrid=True, gridcolor='white'),
        yaxis=dict(showgrid=True, gridcolor='white'),
        zaxis=dict(showgrid=True, gridcolor='white'),
    ),
    margin=dict(l=0, r=0, b=0, t=60)
)

# Render
st.title("📊 3D Linear Readiness Surface")
st.plotly_chart(fig, use_container_width=True)
