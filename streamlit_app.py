
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from itertools import combinations

st.set_page_config(page_title="3D Readiness Landscape", layout="wide")

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

# Simulated readiness scores
y = np.random.normal(75, 10, len(df))

# Rank feature pairs by R²
pair_scores = []
for x_feat, y_feat in combinations(features, 2):
    X = df[[x_feat, y_feat]]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    pair_scores.append((x_feat, y_feat, r2))

top_pairs = sorted(pair_scores, key=lambda x: x[2], reverse=True)
x_feature, y_feature, top_r2 = top_pairs[0]

# UI
st.title("🌐 3D Readiness Landscape (Artificial Data)")
st.markdown(f"Best feature combo: **{x_feature} vs {y_feature}** with R² = {top_r2:.2f}")

# Train on best pair
X_pair = df[[x_feature, y_feature]]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_pair, y)
Z_pred = model.predict(X_pair)

# Surface
grid_size = 30
x_range = np.linspace(X_pair[x_feature].min(), X_pair[x_feature].max(), grid_size)
y_range = np.linspace(X_pair[y_feature].min(), X_pair[y_feature].max(), grid_size)
xx, yy = np.meshgrid(x_range, y_range)
grid_df = pd.DataFrame({x_feature: xx.ravel(), y_feature: yy.ravel()})
zz = model.predict(grid_df).reshape(xx.shape)

# Color-coded pins
colors = ['red' if val < 70 else 'blue' if val < 85 else 'green' for val in Z_pred]

# Create figure
fig = go.Figure()

# Add surface
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.8, showscale=False))

# Add scatter points
fig.add_trace(go.Scatter3d(
    x=X_pair[x_feature],
    y=X_pair[y_feature],
    z=Z_pred,
    mode='markers',
    marker=dict(size=5, color=colors),
    name="Readiness Pins",
    text=[f"{x_feature}: {a}, {y_feature}: {b}, Readiness: {c:.1f}" 
          for a, b, c in zip(X_pair[x_feature], X_pair[y_feature], Z_pred)],
    hoverinfo="text"
))

# Add tree-like lines to base
for xi, yi, zi, color in zip(X_pair[x_feature], X_pair[y_feature], Z_pred, colors):
    fig.add_trace(go.Scatter3d(
        x=[xi, xi],
        y=[yi, yi],
        z=[0, zi],
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False
    ))

# Layout
fig.update_layout(
    title=f"3D Readiness Surface: {x_feature} vs {y_feature}",
    scene=dict(
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        zaxis_title="Predicted Readiness",
        xaxis=dict(showgrid=True, gridcolor='white'),
        yaxis=dict(showgrid=True, gridcolor='white'),
        zaxis=dict(showgrid=True, gridcolor='white'),
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig, use_container_width=True)
