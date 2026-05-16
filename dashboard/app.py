import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Toxicity Dashboard",
    layout="wide"
)

st.title("Live Comment Moderation Dashboard")

# analyze section
comment = st.text_area("Enter Comment")

if st.button("Analyze"):

    response = requests.post(
        "http://127.0.0.1:8000/clean-comment",
        json={"text": comment}
    )

    data = response.json()

    st.subheader("Result")
    st.code(data["cleaned"])
    st.metric("Score", data["score"])

    if data["toxic"]:
        st.error("Toxic")
    else:
        st.success("Clean")


# stats section
stats = requests.get(
    "http://127.0.0.1:8000/stats"
).json()

st.subheader("Live Stats")

col1, col2, col3 = st.columns(3)

col1.metric("Total", stats["total"])
col2.metric("Toxic", stats["toxic"])
col3.metric("Clean", stats["clean"])


# chart
chart_df = pd.DataFrame({
    "Type": ["Toxic", "Clean"],
    "Count": [stats["toxic"], stats["clean"]]
})

st.bar_chart(chart_df.set_index("Type"))


# history
history = requests.get(
    "http://127.0.0.1:8000/history"
).json()

st.subheader("Recent Comments")

df = pd.DataFrame(history)

st.dataframe(df, use_container_width=True)