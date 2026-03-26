"""
AI Data Analyst Agent (Final Version)
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_ai_insights(df, query):
    """Generate AI insights from dataset."""
    data_sample = df.head(10).to_string()

    prompt = f"""
    You are a professional data analyst.

    Dataset sample:
    {data_sample}

    Question:
    {query}

    Provide clear, structured insights.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


def plot_chart(df):
    """Generate simple chart."""
    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) >= 2:
        x = numeric_cols[0]
        y = numeric_cols[1]

        fig, ax = plt.subplots()
        df.plot(x=x, y=y, kind="line", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for visualization.")


def main():
    """Run Streamlit app."""

    st.set_page_config(page_title="AI Data Analyst", layout="wide")

    st.title("📊 AI Data Analyst Agent")
    st.write("Upload dataset and get AI-powered insights.")

    st.divider()

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    query = st.text_input("Ask a question about the dataset")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📊 Basic Statistics")
        st.write(df.describe())

        st.subheader("📈 Data Visualization")
        plot_chart(df)

        if query:
            if st.button("🔍 Analyze Data"):
                with st.spinner("Analyzing data..."):
                    insights = get_ai_insights(df, query)

                st.subheader("🤖 AI Insights")
                st.write(insights)


if __name__ == "__main__":
    main()