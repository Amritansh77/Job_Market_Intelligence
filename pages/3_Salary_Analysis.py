import pandas as pd
import plotly.express as px
import streamlit as st
from utils import load_data,apply_global_filters


def normalize_salary_location(value):

    if pd.isna(value):
        return pd.NA

    location = str(value).strip()

    if location == "":
        return pd.NA

    location_lower = location.lower()

    excluded_values = {"india", "canada", "united states", "united kingdom", "australia",
        "germany", "ireland","france","italy","spain", "netherlands","belgium","switzerland",
        "sweden","norway", "denmark","finland", "poland","singapore", "new zealand","usa",
        "us","uk","us/canada","us / canada","remote","us-remote","remote in the us",
        "us remote","nyc-privy","ontario"
    }

    if location_lower in excluded_values:
        return pd.NA

    if "remote" in location_lower:
        return pd.NA

    if "•" in location or ";" in location:
        return pd.NA


    prefixes = ["Hybrid - ",
        "Hybrid – ",
        "Hybrid–",
        "Remote - ",
        "Remote – ",
        "Remote–",
        "On-site - ",
        "On-site – ",
        "Onsite - ",
        "Onsite – "
    ]

    for prefix in prefixes:

        if location.lower().startswith(prefix.lower()):
            location = location[len(prefix):].strip()
            break



    if location.lower().startswith("us-"):
        location = location[3:].strip()

    elif location.lower().startswith("usa-"):
        location = location[4:].strip()

    city_aliases = {
        "nyc": "New York",
        "new york city": "New York",
        "new york": "New York",

        "sf": "San Francisco",
        "san fran": "San Francisco",
        "san francisco": "San Francisco",

        "la": "Los Angeles",
        "los angeles": "Los Angeles",

        "dc": "Washington",
        "washington dc": "Washington",

        "philly": "Philadelphia",
        "philadelphia": "Philadelphia",

        "bangalore": "Bangalore",
        "bengaluru": "Bangalore"
    }

    city = location.split(",")[0].strip()

    if city == "":
        return pd.NA

    city_key = city.lower()

    if city_key in city_aliases:
        return city_aliases[city_key]

    return city

# PAGE CONFIGURATION
st.set_page_config( page_title="Salary Analysis | Job Market Intelligence",
    page_icon="💰",  layout="wide")

# LOAD DATA
df = load_data()
filtered_df = apply_global_filters(df)


# PAGE HEADER
st.title("💰 Salary Analysis")

st.caption( "Explore salary coverage, salary levels, and salary patterns "
    "across roles and locations.")


# EMPTY DATA CHECK
if filtered_df.empty:
    st.warning( "No jobs match the current global filters." )
    st.stop()

# PREPARE SALARY DATA
salary_df = filtered_df.copy()
salary_df["salary_min"] = pd.to_numeric(salary_df["salary_min"],errors="coerce")

salary_df["salary_max"] = pd.to_numeric(salary_df["salary_max"],errors="coerce")

salary_df["salary_mid"] = pd.to_numeric(salary_df["salary_mid"],errors="coerce")


# SALARY CURRENCY CLEANING
if "salary_currency" in salary_df.columns:
    salary_df["salary_currency"] = (salary_df["salary_currency"] .fillna("Unspecified")
        .astype(str).str.strip())

    salary_df.loc[salary_df["salary_currency"].eq(""),"salary_currency"] = "Unspecified"
else:
    salary_df["salary_currency"] = "Unspecified"

# BASIC SALARY COUNTS
total_jobs = len(filtered_df)
salary_rows = salary_df["salary_mid"].notna().sum()
salary_coverage = ((salary_rows / total_jobs) * 100 if total_jobs > 0 else 0)


# KNOWN CURRENCY DATA
known_currency_df = salary_df[salary_df["salary_mid"].notna()].copy()

known_currency_df = known_currency_df[known_currency_df["salary_currency"].notna()]


known_currency_df["salary_currency"] = (known_currency_df["salary_currency"].astype(str).str.strip())
known_currency_df = known_currency_df[known_currency_df["salary_currency"].ne("")]
known_currency_df = known_currency_df[known_currency_df["salary_currency"].str.lower()!= "unspecified"]
known_currency_values = (known_currency_df["salary_currency"])
known_currency_counts = (known_currency_values.value_counts())


# PAGE METRICS
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric( "Salary Coverage",f"{salary_coverage:.1f}%")
with col2:
    st.metric( "Jobs with Salary",f"{salary_rows:,}")
with col3:
    st.metric("Known Currencies",f"{known_currency_values.nunique():,}")
with col4:
    if not known_currency_counts.empty:
        most_common_currency = ( known_currency_counts.index[0] )
        st.metric("Most Common Currency",most_common_currency )
    else:
        st.metric( "Most Common Currency", "N/A")

# SALARY DATA RELIABILITY
st.markdown("---")
st.subheader("⚠️ Salary Data Reliability")

# RECOGNIZED CURRENCIES

recognized_currencies = { "USD","CAD", "EUR","GBP", "AUD"}

currency_series = (salary_df["salary_currency"].dropna().astype(str).str.strip().str.upper())

recognized_currency_series = currency_series[currency_series.isin(recognized_currencies)]

known_currencies = int(recognized_currency_series.nunique())
if known_currencies == 0:
    st.warning(
        "No recognized salary currency is available for the current "
        "filters, so currency-specific salary comparisons cannot be performed."
    )
else:
    st.info(
        f"{known_currencies} recognized salary currencies are available "
        "for the current filters. Salary values from different currencies "
        "are not directly comparable without currency normalization."
    )

# OPTIONAL ESTIMATION / SOURCE NOTE
estimation_columns = ["salary_estimated","is_salary_estimated","salary_source", "salary_type"]
available_estimation_columns = [column for column in estimation_columns if column in salary_df.columns]

if available_estimation_columns:

    column_name = available_estimation_columns[0]

    st.caption( f"Salary source/estimation information is available through " f"the `{column_name}` field." )
else:
    st.caption(
        "The current dashboard dataset does not contain a dedicated "
        "salary-estimation/source flag, so reported and estimated "
        "salary values cannot be separated here.")



# SALARY COVERAGE BY CURRENCY

st.markdown("---")

st.subheader("💱 Salary Coverage by Currency")

currency_chart_df = (
    salary_df.loc[salary_df["salary_mid"].notna(),"salary_currency"].value_counts()
    .rename_axis("Currency").reset_index(name="Jobs"))

if not currency_chart_df.empty:
    currency_chart_df["Percentage"] = ((currency_chart_df["Jobs"]/ salary_rows) * 100 )
    currency_chart_df = currency_chart_df.sort_values("Jobs",ascending=True )
    fig_currency = px.bar(currency_chart_df, x="Jobs",y="Currency",orientation="h",
        text="Jobs",title="Jobs with Salary by Currency" )

    fig_currency.update_traces( textposition="outside", cliponaxis=False )

    fig_currency.update_layout(height=450, xaxis_title="Jobs",yaxis_title="Currency",
        margin=dict(l=10, r=50, t=60, b=20))
    st.plotly_chart(fig_currency, use_container_width=True)

else:
    st.info("No salary data is available for the current selection." )



# CURRENCY SELECTOR
st.markdown("---")
st.subheader("💵 Salary Analysis by Currency")

available_currencies = sorted(known_currency_values.unique().tolist())

if available_currencies:
    selected_currency = st.selectbox( "Select salary currency",available_currencies, key="salary_currency_selector")
    st.caption("Only explicitly specified currencies are available "
        "for salary comparison.")
else:
    selected_currency = None
    st.info("No salary observations with a specified currency "
        "are available for the current filters." )



# SELECTED CURRENCY DATA

if selected_currency is not None:
    currency_salary_df = salary_df[
        (salary_df["salary_currency"] == selected_currency)
        &
        salary_df["salary_mid"].notna()].copy()
else:
    currency_salary_df = pd.DataFrame()


# SELECTED CURRENCY SUMMARY

if not currency_salary_df.empty:
    median_salary = (currency_salary_df["salary_mid"].median() )
    min_salary = (currency_salary_df["salary_mid"].min() )
    max_salary = (  currency_salary_df["salary_mid"].max() )
else:
    median_salary = None
    min_salary = None
    max_salary = None

col1, col2, col3 = st.columns(3)

with col1:
    if median_salary is not None:
        st.metric( f"Median Salary ({selected_currency})",f"{median_salary:,.0f}")
    else:
        st.metric("Median Salary","N/A")
with col2:
    if min_salary is not None:
        st.metric(f"Minimum Salary ({selected_currency})", f"{min_salary:,.0f}")
    else:
        st.metric("Minimum Salary","N/A")
with col3:
    if max_salary is not None:
        st.metric(f"Maximum Salary ({selected_currency})",f"{max_salary:,.0f}")
    else:
        st.metric("Maximum Salary","N/A")


min_group_observations = 2
# OUTLIER HANDLING
if not currency_salary_df.empty:
    q1 = currency_salary_df["salary_mid"].quantile(0.25)
    q3 = currency_salary_df["salary_mid"].quantile( 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_salary_df = currency_salary_df[
        (currency_salary_df["salary_mid"]>= lower_bound)
        &
        ( currency_salary_df["salary_mid"]<= upper_bound)].copy()

    removed_count = (len(currency_salary_df)- len(filtered_salary_df))
else:
    filtered_salary_df = pd.DataFrame()
    removed_count = 0


# SALARY DISTRIBUTION
currency_label = (
    selected_currency
    if pd.notna(selected_currency) and str(selected_currency).strip() != ""
    else "No Recognized Currency"
)
st.markdown("---")
st.subheader( f"📈 Salary Distribution — {currency_label} ")

if not filtered_salary_df.empty:
    fig_distribution = px.histogram( filtered_salary_df,x="salary_mid", nbins=30,
        title="Salary Distribution — IQR Outliers Removed")

    fig_distribution.update_layout(height=450,
        xaxis_title=f"Salary ({selected_currency})",
        yaxis_title="Number of Jobs",
        margin=dict(l=10, r=20,t=60,b=20))

    st.plotly_chart(fig_distribution,use_container_width=True )

    st.caption(f"IQR filtering removed {removed_count:,} "
        f"extreme salary observations from this chart.")

else:
    st.info("Not enough salary data to display the distribution." )

# MEDIAN SALARY BY LOCATION
currency_label = (
    selected_currency
    if pd.notna(selected_currency) and str(selected_currency).strip() != ""
    else "No Recognized Currency"
)

st.markdown("---")
st.subheader(f"📍 Median Salary by City — {currency_label} ")

if not currency_salary_df.empty:
    location_salary = currency_salary_df.copy()
    # Create normalized city from location_clean
    location_salary["salary_city"] = (location_salary["location_clean"].apply(normalize_salary_location) )

    location_salary = location_salary.dropna(
        subset=[ "salary_city","salary_mid"])



    # Group by normalized city
    city_salary = (location_salary.groupby("salary_city")
        .agg(Median_Salary=("salary_mid", "median"),
            Job_Count=("salary_mid", "size"))
        .reset_index())


    # Require at least 2 observations
    city_salary = city_salary[city_salary["Job_Count"] >= 2]

    if not city_salary.empty:
        # Select locations with the most observations
        city_salary = (city_salary.sort_values( "Job_Count",ascending=False ).head(15))
        # Sort by salary for horizontal chart
        city_salary = city_salary.sort_values("Median_Salary", ascending=True)

        fig_city_salary = px.bar( city_salary,x="Median_Salary", y="salary_city", orientation="h",
            text="Median_Salary",
            hover_data={ "Job_Count": True,
                "Median_Salary": ":,.0f"
            },
            title="Median Salary by City"
        )


        fig_city_salary.update_traces(texttemplate="%{text:,.0f}",textposition="outside",cliponaxis=False)

        fig_city_salary.update_layout( height=600,
            xaxis_title=(f"Median Salary ({selected_currency})"
            ),yaxis_title="City",margin=dict(l=10,r=90,t=60,b=20))

        st.plotly_chart(fig_city_salary, use_container_width=True )

        st.caption(
            "City names are normalized from the cleaned location field. "
            "Country-only and obvious multi-location postings are excluded.")
    else:
        st.info(f"No city has at least 2 salary observations "
            f"for {selected_currency} under the current filters." )

else:
    st.info("No salary observations are available "
        "for the selected currency." )

# SALARY RANGE
currency_label = (selected_currency if pd.notna(selected_currency) and str(selected_currency).strip() != ""
    else "No Recognized Currency")

st.markdown("---")
st.subheader( f"📊 Salary Range — {currency_label} ")

if not filtered_salary_df.empty:
    salary_range_df = (filtered_salary_df[[ "salary_min","salary_max"] ].dropna())
    if not salary_range_df.empty:
        range_long = salary_range_df.melt(value_vars=[ "salary_min","salary_max"],
            var_name="Salary Type", value_name="Salary")
        range_long["Salary Type"] = ( range_long["Salary Type"].map(
                {"salary_min": "Minimum Salary","salary_max": "Maximum Salary" }))
        fig_range = px.box( range_long,x="Salary",y="Salary Type",orientation="h",
            title="Salary Range Distribution" )
        fig_range.update_layout(height=350,xaxis_title=f"Salary ({selected_currency})",
            yaxis_title="",margin=dict(l=10,r=20, t=60, b=20))
        st.plotly_chart( fig_range,use_container_width=True)
    else:
        st.info( "Minimum and maximum salary values are not "
            "available together for the selected currency." )
else:
    st.info("No salary observations are available.")

# DATA QUALITY SUMMARY
st.markdown("---")
st.subheader("🔎 Salary Data Quality")

quality_col1, quality_col2, quality_col3, quality_col4 = (st.columns(4))
with quality_col1:
    st.metric( "Total Postings",f"{total_jobs:,}")
with quality_col2:
    st.metric("Postings with Salary", f"{salary_rows:,}")
with quality_col3:
    st.metric( "Missing Salary",f"{total_jobs - salary_rows:,}")
with quality_col4:
    unspecified_salary_rows = (salary_df[ salary_df["salary_mid"].notna()
            &
            salary_df["salary_currency"].str.lower().eq("unspecified")].shape[0])
    st.metric( "Salary with Unspecified Currency",f"{unspecified_salary_rows:,}" )

st.caption( "Salary analysis is based on the current global filter selection.")

st.caption( "City-level salary comparisons require at least 2 salary observations per city.")