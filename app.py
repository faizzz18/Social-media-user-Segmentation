
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Social Media User Segmentation",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Data and Model
# -----------------------------
user_df = pd.read_csv("user_segments.csv")
kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("📊 Social Media User Segmentation")
st.write(
    "This dashboard groups social media users according to "
    "their interests and engagement behavior using K-Means clustering."
)

st.divider()

# -----------------------------
# Key Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Users", user_df["user_id"].nunique())

with col2:
    st.metric("Total Clusters", user_df["cluster"].nunique())

with col3:
    st.metric(
        "Average Likes",
        f"{user_df['likes'].mean():,.0f}"
    )

with col4:
    st.metric(
        "Average Engagement Rate",
        f"{user_df['engagement_rate'].mean():.4f}"
    )

# -----------------------------
# Cluster Distribution
# -----------------------------
st.subheader("User Distribution by Cluster")

cluster_counts = (
    user_df["cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)

# -----------------------------
# Cluster Profile
# -----------------------------
st.subheader("Cluster Profile")

cluster_profile = user_df.groupby("cluster")[
    [
        "likes",
        "comments",
        "shares",
        "engagement_rate",
        "followers_count",
        "following_count",
        "post_count"
    ]
].mean().round(2)

st.dataframe(
    cluster_profile,
    use_container_width=True
)

# -----------------------------
# Engagement Comparison
# -----------------------------
st.subheader("Average Engagement by Cluster")

engagement_comparison = user_df.groupby("cluster")[
    ["likes", "comments", "shares"]
].mean().round(2)

st.bar_chart(engagement_comparison)

# -----------------------------
# Cluster Selection
# -----------------------------
st.subheader("Explore a User Segment")

selected_cluster = st.selectbox(
    "Select Cluster",
    sorted(user_df["cluster"].unique())
)

selected_users = user_df[
    user_df["cluster"] == selected_cluster
]

st.write(
    f"Number of users in Cluster {selected_cluster}: "
    f"{len(selected_users)}"
)

st.dataframe(
    selected_users,
    use_container_width=True
)

# -----------------------------
# Final Information
# -----------------------------
st.info(
    "The final K-Means model uses K = 4 clusters. "
    "The Silhouette Score obtained during model evaluation was 0.1661."
)
