import pandas as pd
import plotly.express as px
import streamlit as st

from utils import load_data, apply_global_filters


# PAGE CONFIGURATION
st.set_page_config( page_title="Overview | Job Market Intelligence",
    page_icon="🏠",
    layout="wide")



# LOAD DATA
df = load_data()
filtered_df = apply_global_filters(df)

# PAGE HEADER
st.title("🏠 Job Market Overview")
st.caption("Explore hiring volume, locations, salary coverage, "
    "and employment patterns.")

# BASIC KPIs

total_jobs = len(filtered_df)

# Number of unique companies
if "company_name" in filtered_df.columns:
    total_companies = (filtered_df["company_name"].dropna().astype(str).str.strip())
    total_companies = total_companies[total_companies != ""].nunique()
else:
    total_companies = 0


# Number of unique cleaned locations
if "location_clean" in filtered_df.columns:
    total_locations = ( filtered_df["location_clean"].dropna().astype(str) .str.strip())
    total_locations = total_locations[total_locations != "" ].nunique()
else:
    total_locations = 0

# SALARY METRICS
if "salary_mid" in filtered_df.columns:

    salary_data = pd.to_numeric(filtered_df["salary_mid"], errors="coerce")
    salary_data = salary_data.dropna()
else:
    salary_data = pd.Series(dtype="float64")

total_jobs = len(filtered_df)
salary_rows = len(salary_data)

salary_coverage = ((salary_rows / total_jobs) * 100 if total_jobs > 0 else 0)

# Known-currency salary observations
if ( "salary_currency" in filtered_df.columns and "salary_mid" in filtered_df.columns
):
    known_currency_salary = filtered_df[filtered_df["salary_mid"].notna()
        &
        filtered_df["salary_currency"].notna()
        &
        ( filtered_df["salary_currency"].astype(str).str.strip().ne("") )]

    known_currency_salary = known_currency_salary[known_currency_salary["salary_currency"].astype(str)
        .str.strip().str.lower() != "unspecified"]
    known_currency_salary_count = len(known_currency_salary)
else:
    known_currency_salary_count = 0



# CONTRACT TYPE METRICS
if "contract_type" in filtered_df.columns:
    contract_data = (filtered_df["contract_type"].dropna().astype(str)
        .str.strip().str.lower() )

    # Only use the two actual ML classes
    contract_data = contract_data[
        contract_data.isin(["permanent", "contract"])]

else:
    contract_data = pd.Series(dtype="object")


if not contract_data.empty:

    permanent_share = ((contract_data == "permanent").mean() * 100)

    contract_share = ((contract_data == "contract").mean() * 100 )

else:

    permanent_share = 0
    contract_share = 0



# KPI CARDS

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric( "Total Jobs",f"{total_jobs:,}")
with col2:
    st.metric("Companies",f"{total_companies:,}" )
with col3:
    st.metric( "Locations",f"{total_locations:,}")
with col4:
    st.metric("Jobs with Salary",f"{salary_rows:,}")

# KEY MARKET INSIGHTS
st.markdown("---")
st.subheader("📌 Key Market Insights")
insight_col1, insight_col2, insight_col3 = st.columns(3)

# Insight 1: Top Hiring City
with insight_col1:
    if "city" in filtered_df.columns:
        city_data = (filtered_df["city"].dropna().astype(str) .str.strip())
        city_data = city_data[city_data != ""]
        # Remove country-level value confirmed in the dataset
        city_data = city_data[city_data.str.lower() != "india" ]
        city_counts = city_data.value_counts()
    else:
        city_counts = pd.Series(dtype="int64")

    if not city_counts.empty:
        top_city = city_counts.index[0]
        top_city_count = city_counts.iloc[0]
        st.info(f"**Top hiring city**\n\n"
            f"{top_city}\n\n"
            f"{top_city_count:,} postings" )
    else:
        st.info("**Top hiring city**\n\n"
            "No city data available.")


# Insight 2: Salary Coverage
with insight_col2:
    st.info(f"**Salary coverage**\n\n"
        f"{salary_coverage:.1f}%\n\n"
        f"of filtered postings have salary information" )



# Insight 3: Employment Mix
with insight_col3:
    if not contract_data.empty:
        st.info( f"**Employment mix**\n\n"
            f"Permanent: {permanent_share:.1f}%\n\n"
            f"Contract: {contract_share:.1f}%" )
    else:
        st.info( "**Employment mix**\n\n"
            "No permanent/contract labels "
            "available for this selection.")

# CHART SECTION

st.markdown("---")

chart_col1, chart_col2 = st.columns(2)

# CHART 1 — TOP HIRING CITIES

with chart_col1:
    if "city" in filtered_df.columns:
        city_chart = ( filtered_df["city"].dropna() .astype(str).str.strip())
        city_chart = city_chart[city_chart != "" ]
        # Remove country-level value
        city_chart = city_chart[ city_chart.str.lower() != "india"]
        city_chart = (city_chart.value_counts().head(10))
        city_chart = (city_chart.rename_axis("City").reset_index(name="Jobs"))
    else:

        city_chart = pd.DataFrame( columns=["City", "Jobs"])

    if not city_chart.empty:

        city_chart = city_chart.sort_values("Jobs",ascending=True)
        fig_city = px.bar(city_chart, x="Jobs",y="City", orientation="h", title="Top Hiring Cities",
            text="Jobs")
        fig_city.update_traces(textposition="outside",cliponaxis=False)
        fig_city.update_layout( xaxis_title="Jobs",yaxis_title="City",
            margin=dict( l=10,r=50,t=60, b=20),
            height=500)

        st.plotly_chart(fig_city,use_container_width=True )

    else:
        st.info("No city data available for the current selection." )



# CHART 2 — JOBS BY CATEGORY
with chart_col2:
    if "category" in filtered_df.columns:
        category_chart = (filtered_df["category"].dropna().astype(str).str.strip())
        category_chart = category_chart[ category_chart != "" ]
        category_chart = (category_chart.value_counts().head(10))
        category_chart = ( category_chart.rename_axis("Category").reset_index(name="Jobs") )
    else:
        category_chart = pd.DataFrame( columns=["Category", "Jobs"])

    if not category_chart.empty:
        category_chart = category_chart.sort_values( "Jobs",ascending=True)
        fig_category = px.bar( category_chart,x="Jobs",y="Category", orientation="h",
            title="Jobs by Category", text="Jobs")
        fig_category.update_traces( textposition="outside",cliponaxis=False)
        fig_category.update_layout( xaxis_title="Jobs", yaxis_title="Category",
            margin=dict( l=10,r=50, t=60, b=20),height=500)
        st.plotly_chart(fig_category,use_container_width=True )
    else:
        st.info("No category data available for the current selection.")

# DATA FRESHNESS
if "created_date" in df.columns:
    valid_dates = pd.to_datetime(df["created_date"], errors="coerce" ).dropna()
    if not valid_dates.empty:
        data_start = valid_dates.min().date()
        data_end = valid_dates.max().date()
        st.caption( f"Dataset posting-date range: "
            f"{data_start} → {data_end}")
    else:
        st.caption("Dataset posting-date range is unavailable.")
else:
    st.caption( "created_date column is unavailable.")

# CURRENT FILTER SUMMARY

st.caption(f"Showing {len(filtered_df):,} filtered postings "
    f"out of {len(df):,} total postings.")