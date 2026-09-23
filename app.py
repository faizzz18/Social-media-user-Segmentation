import streamlit as st
import pandas as pd

# Load segmented user data
user_df = pd.read_csv("user_segments.csv")

st.set_page_config(
    page_title="Social Media User Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Social Media User Segmentation")
st.write(
    "This dashboard segments social media users based on their "
    "engagement behavior and content interests using K-Means clustering."
)

# Dataset overview
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Users", len(user_df))

with col2:
    st.metric("Number of Clusters", user_df["cluster"].nunique())

with col3:
    st.metric("Average Engagement Rate",
              round(user_df["engagement_rate"].mean(), 4))

# Cluster distribution
st.subheader("User Distribution by Cluster")

cluster_counts = (
    user_df["cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)

# Cluster profile
st.subheader("Cluster Profile")

cluster_profile = user_df.groupby("cluster")[[
    "likes",
    "comments",
    "shares",
    "engagement_rate",
    "followers_count",
    "following_count",
    "post_count"
]].mean().round(2)

st.dataframe(
    cluster_profile,
    use_container_width=True
)

# User segments
st.subheader("User Segments")

selected_cluster = st.selectbox(
    "Select a Cluster",
    sorted(user_df["cluster"].unique())
)

filtered_users = user_df[
    user_df["cluster"] == selected_cluster
]

st.write(
    f"Users belonging to Cluster {selected_cluster}: "
    f"{len(filtered_users)}"
)

st.dataframe(
    filtered_users,
    use_container_width=True
)
