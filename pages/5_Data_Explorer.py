import pandas as pd
import streamlit as st

from utils import load_data, apply_global_filters



# PAGE CONFIG
st.set_page_config(
    page_title="Data Explorer | Job Market Intelligence",
    page_icon="🔎",
    layout="wide",
)


# LOAD DATA
df = load_data()



# GLOBAL FILTERS
filtered_df = apply_global_filters(df)



# PAGE HEADER
st.title("🔎 Data Explorer")

st.caption("Explore and inspect job-posting records from the filtered dataset.")


# DATASET SUMMARY
st.markdown("---")
st.subheader("📊 Dataset Summary")

total_records = len(filtered_df)

total_companies = (filtered_df["company_name"].nunique()
    if "company_name" in filtered_df.columns
    else 0)

total_locations = (filtered_df["location_clean"].nunique()
    if "location_clean" in filtered_df.columns
    else 0)

salary_records = (filtered_df["has_salary"].sum()
    if "has_salary" in filtered_df.columns
    else 0)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records",
        f"{total_records:,}")

with col2:
    st.metric("Companies",
        f"{total_companies:,}")

with col3:
    st.metric("Locations",
        f"{total_locations:,}")

with col4:
    st.metric("Salary Records",
        f"{int(salary_records):,}")


# EXPLORER CONTROLS

st.markdown("---")
st.subheader("🎛️ Explorer Controls")

search_title = st.text_input(
    "Search Job Title",
    placeholder="e.g. Data Scientist",
    key="explorer_search_title"
)


default_columns = [ "title","company_name","category", "standard_role","location_clean","city",
    "country","salary_currency","salary_min","salary_max","contract_type","working_model_clean",
    "source","created_date",]

available_columns = [column for column in default_columns if column in filtered_df.columns]

selected_columns = st.multiselect(
    "Columns to Display",
    options=filtered_df.columns.tolist(),
    default=available_columns,
    key="explorer_columns"
)

rows_to_show = st.selectbox(
    "Rows to Display",
    [10, 25, 50, 100],
    index=1,
    key="explorer_rows"
)


# APPLY SEARCH
explorer_df = filtered_df.copy()
if search_title.strip():
    explorer_df = explorer_df[ explorer_df["title"]
        .astype(str)
        .str.contains(search_title.strip(),case=False,na=False)
    ]



# DATA TABLE

st.markdown("---")
st.subheader("📋 Job Data")
st.caption( f"Showing {min(len(explorer_df), rows_to_show):,} "
    f"of {len(explorer_df):,} matching records."
)


if explorer_df.empty:
    st.info( "No job postings match the current filters and search." )

else:
    if selected_columns:
        display_df = explorer_df[ selected_columns ].head(rows_to_show)
    else:
        display_df = explorer_df.head( rows_to_show)

    st.dataframe(display_df,
        use_container_width=True,
        hide_index=True )



# RECORD DETAILS
st.markdown("---")
st.subheader("📄 Record Details")

if explorer_df.empty:

    st.info("No records are available to inspect.")

else:
    # Use only the records currently displayed
    detail_df = explorer_df.head(rows_to_show).copy()
    detail_indices = detail_df.index.tolist()
    selected_index = st.selectbox(
        "Select a Job Posting",detail_indices,
        format_func=lambda index: (
            f"{detail_df.loc[index, 'title']} — "
            f"{detail_df.loc[index, 'company_name']}"
        ),
        key="selected_job_record"
    )

    selected_record = detail_df.loc[selected_index]



    # VALUE FORMATTER
    def display_value(value):
        if pd.isna(value):
            return "Not available"

        value = str(value).strip()

        if value == "":
            return "Not available"

        return value



    # DETAILS
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.write(
            f"**Job Title:** "
            f"{display_value(selected_record.get('title'))}"
        )

        st.write(
            f"**Company:** "
            f"{display_value(selected_record.get('company_name'))}"
        )

        st.write(
            f"**Category:** "
            f"{display_value(selected_record.get('category'))}"
        )

        st.write(
            f"**Standard Role:** "
            f"{display_value(selected_record.get('standard_role'))}"
        )

        st.write(
            f"**Location:** "
            f"{display_value(selected_record.get('location_clean'))}"
        )


    with detail_col2:

        st.write(
            f"**Salary Currency:** "
            f"{display_value(selected_record.get('salary_currency'))}"
        )

        st.write(
            f"**Salary Min:** "
            f"{display_value(selected_record.get('salary_min'))}"
        )

        st.write(
            f"**Salary Max:** "
            f"{display_value(selected_record.get('salary_max'))}"
        )

        st.write(
            f"**Contract Type:** "
            f"{display_value(selected_record.get('contract_type'))}"
        )

        st.write(
            f"**Source:** "
            f"{display_value(selected_record.get('source'))}"
        )


# DOWNLOAD FILTERED DATA
st.markdown("---")
st.subheader("⬇️ Download Data")


csv_data = explorer_df.to_csv( index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Data",
    data=csv_data,
    file_name="filtered_job_market_data.csv",
    mime="text/csv",
    use_container_width=True,
    key="download_filtered_data"
)