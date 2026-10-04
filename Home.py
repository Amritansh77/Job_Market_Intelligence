import streamlit as st

from utils import load_data, apply_global_filters


st.set_page_config(page_title="Job Market Intelligence",page_icon="💼",  layout="wide")

df = load_data()

st.title("💼 Job Market Intelligence Platform")
filtered_df = apply_global_filters(df)
st.markdown("---")
st.subheader("Current Selection")
st.write( f"Showing **{len(filtered_df):,}** jobs "
    f"out of **{len(df):,}** total jobs.")