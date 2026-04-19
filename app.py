import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.title("🏙 Predictive Urban Growth Model (Real Data)")

# Load dataset
df = pd.read_csv("train.csv")

# -------------------------------
# Data Cleaning
# -------------------------------

# Keep only relevant columns
df = df[['SalePrice', 'GrLivArea', 'OverallQual', 'YearBuilt']]

# Drop missing
df = df.dropna()

# Rename for clarity
df.rename(columns={'SalePrice': 'price'}, inplace=True)

# -------------------------------
# Feature Engineering
# -------------------------------

# Simulating realistic features from real data
df['infrastructure_score'] = df['OverallQual']
df['population_growth'] = (2024 - df['YearBuilt']) / 100
df['rent_estimate'] = df['price'] * 0.006

df['price_growth'] = df['price'].pct_change().fillna(0)
df['rental_yield'] = df['rent_estimate'] / df['price']

# Growth Score
df['growth_score'] = (
    df['price_growth'] * 0.4 +
    df['rental_yield'] * 0.3 +
    df['infrastructure_score'] * 0.2 +
    df['population_growth'] * 0.1
)

# -------------------------------
# Train Model
# -------------------------------

X = df[['price', 'infrastructure_score', 'population_growth']]
y = df['growth_score']

model = RandomForestRegressor()
model.fit(X, y)

# -------------------------------
# User Input
# -------------------------------

st.subheader("📥 Enter Property Details")

price = st.number_input("Property Price", value=200000)
infra = st.slider("Infrastructure Score (1-10)", 1, 10, 5)
pop = st.slider("Population Growth", 0.01, 0.5, 0.1)

if st.button("Predict Growth"):
    pred = model.predict([[price, infra, pop]])
    st.success(f"📈 Predicted Growth Score: {pred[0]:.4f}")

# -------------------------------
# Visualization (Scatter)
# -------------------------------

st.subheader("📊 Growth Distribution")

fig = px.scatter(
    df,
    x="price",
    y="growth_score",
    color="growth_score"
)

st.plotly_chart(fig)

# -------------------------------
# Top Properties
# -------------------------------

st.subheader("🏆 Top Growth Properties")

top = df.sort_values(by="growth_score", ascending=False).head(10)
st.write(top)