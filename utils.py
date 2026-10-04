from pathlib import Path
import re

import pandas as pd
import streamlit as st


# DATA PATH
DATA_PATH = Path(__file__).resolve().parent / "jobs_feature_engineered.csv"



# COUNTRY EXTRACTION

def extract_country(value):
    """
    Extract country from location_clean.

    Rules:
    - Explicit country names are detected.
    - Common US/UK location formats are supported.
    - Indian cities/states are mapped to India.
    - If multiple different countries are detected in the same
      location string, country is left unknown rather than guessing.
    """

    if pd.isna(value):
        return pd.NA

    location = str(value).strip()

    if location == "":
        return pd.NA

    text = location.lower()

    detected_countries = set()

    # Explicit country names

    country_patterns = {
        "united states": "United States",
        "united kingdom": "United Kingdom",
        "new zealand": "New Zealand",
        "australia": "Australia",
        "canada": "Canada",
        "germany": "Germany",
        "ireland": "Ireland",
        "france": "France",
        "italy": "Italy",
        "spain": "Spain",
        "netherlands": "Netherlands",
        "belgium": "Belgium",
        "switzerland": "Switzerland",
        "sweden": "Sweden",
        "norway": "Norway",
        "denmark": "Denmark",
        "finland": "Finland",
        "poland": "Poland",
        "singapore": "Singapore",
        "india": "India",
    }

    for pattern, country in country_patterns.items():
        if re.search(rf"\b{re.escape(pattern)}\b", text):
            detected_countries.add(country)

    # US indicators

    us_patterns = [
        r"\busa\b",
        r"\bu\.s\.a\b",
        r"\bunited states\b",
        r"\bus-",
        r"\bus/",
        r"\bus,",
    ]

    if any(re.search(pattern, text) for pattern in us_patterns):
        detected_countries.add("United States")

    # UK indicators

    uk_patterns = [
        r"\buk-",
        r"\buk/",
        r"\buk,",
        r"\buk\b",
        r"\bunited kingdom\b",
    ]

    if any(re.search(pattern, text) for pattern in uk_patterns):
        detected_countries.add("United Kingdom")

    # Indian cities


    indian_cities = { "bangalore", "bengaluru","hyderabad", "mumbai","pune", "chennai","delhi",
        "new delhi","noida","gurgaon","gurugram", "ahmedabad", "navi mumbai", "kolkata",  "jaipur",
        "indore",  "jabalpur", "chandigarh","kochi",  "coimbatore","surat", "vadodara", "nagpur",
        "lucknow",  "bhopal",
    }

    # Indian states


    indian_states = { "karnataka","maharashtra","telangana","tamil nadu", "uttar pradesh","gujarat",
        "haryana", "west bengal","rajasthan", "madhya pradesh", "kerala", "punjab","odisha",
        "bihar", "jharkhand", "andhra pradesh",
    }

    location_parts = [part.strip().lower()
        for part in re.split(r"[,;•|]", location)
        if part.strip()
    ]

    for part in location_parts:

        if part in indian_cities:
            detected_countries.add("India")

        if part in indian_states:
            detected_countries.add("India")

    # Return only when country is unambiguous

    if len(detected_countries) == 1:
        return next(iter(detected_countries))

    # Multiple countries in one location string
    # Example: US / Canada
    if len(detected_countries) > 1:
        return pd.NA

    return pd.NA



# LOAD DATA


@st.cache_data
def load_data():
    """
    Load the final feature-engineered dataset.
    """

    df = pd.read_csv(DATA_PATH)

    # Convert date

    if "created_date" in df.columns:
        df["created_date"] = pd.to_datetime(
            df["created_date"],
            errors="coerce"
        )

    # Create country column

    if "location_clean" in df.columns:

        df["country"] = df["location_clean"].apply( extract_country)

    else:

        df["country"] = pd.NA

    return df



# CLEAN CITY OPTIONS


def get_clean_city_options(df):
    """
    Return clean city values for the global City filter.

    Removes obvious non-city / remote / multi-location values.
    """

    if "city" not in df.columns:
        return ["All"]

    cities = (df["city"].dropna().astype(str).str.strip() )

    # Remove empty values
    cities = cities[cities != ""]

    # Obvious non-city values
    invalid_values = {
        "india",
        "canada",
        "united states",
        "united kingdom",
        "usa",
        "u.s.a",
        "us",
        "uk",
        "remote",
        "us-remote",
        "us remote",
    }

    cities = cities[ ~cities.str.lower().isin(invalid_values) ]

    # Remove obvious multi-location / remote strings
    cities = cities[ ~cities.str.contains(r"remote|•|;|/|\|", case=False,na=False) ]

    return ["All"] + sorted(cities.unique().tolist())



# GLOBAL FILTERS


def apply_global_filters(df):

    st.markdown("### 🌍 Global Filters")

    # COUNTRY OPTIONS

    country_options = ["All"]

    if "country" in df.columns:

        countries = ( df["country"] .dropna() .astype(str).str.strip() )

        countries = countries[countries != ""]

        country_options += sorted(countries.unique().tolist())


    # COUNTRY SELECTBOX

    # Prevent invalid old session value
    if st.session_state.get("global_country") not in country_options:
        st.session_state["global_country"] = "All"


    # CITY SOURCE

    city_source = df.copy()

    if (
        st.session_state.get("global_country", "All") != "All"
        and "country" in city_source.columns
    ):

        city_source = city_source[ city_source["country"] .astype(str) .str.strip()
            ==
            st.session_state["global_country"]
        ]


    # CITY OPTIONS
    city_options = get_clean_city_options(city_source)

    # Prevent invalid old city after changing country
    if st.session_state.get("global_city") not in city_options:
        st.session_state["global_city"] = "All"


    # CATEGORY OPTIONS

    category_options = ["All"]

    if "category" in df.columns:

        categories = ( df["category"].dropna().astype(str) .str.strip())

        categories = categories[categories != ""]

        category_options += sorted(categories.unique().tolist())

    if st.session_state.get("global_category") not in category_options:
        st.session_state["global_category"] = "All"


    # CONTRACT TYPE OPTIONS

    contract_options = ["All"]

    if "contract_type" in df.columns:

        contract_types = (df["contract_type"].dropna().astype(str).str.strip())

        contract_types = contract_types[ contract_types != "" ]

        contract_options += sorted(
            contract_types.unique().tolist())

    if st.session_state.get("global_contract") not in contract_options:
        st.session_state["global_contract"] = "All"


    # DATE RANGE

    if "created_date" in df.columns:

        valid_dates = df["created_date"].dropna()

        if not valid_dates.empty:

            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            default_dates = (min_date, max_date)

        else:
            min_date = None
            max_date = None
            default_dates = None

    else:
        min_date = None
        max_date = None
        default_dates = None


    # FILTER UI

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    row2_col1, row2_col2 = st.columns(2)

    # COUNTRY
    with row1_col1:

        selected_country = st.selectbox("Country",country_options,
            key="global_country"
        )

    # CITY

    with row1_col2:
        selected_city = st.selectbox( "City", city_options,key="global_city" )

    # CATEGORY
    with row1_col3:

        selected_category = st.selectbox( "Category", category_options,  key="global_category" )

    # DATE RANGE
    with row2_col1:

        if default_dates is not None:

            selected_dates = st.date_input(
                "Date Range",
                value=default_dates,
                min_value=min_date,
                max_value=max_date,
                key="global_date"
            )

        else:

            selected_dates = None

    # CONTRACT TYPE
    with row2_col2:
        selected_contract = st.selectbox("Contract Type",contract_options,
            key="global_contract"
        )


    # APPLY FILTERS

    filtered_df = df.copy()

    # Country
    if (
        selected_country != "All"
        and "country" in filtered_df.columns
    ):

        filtered_df = filtered_df[filtered_df["country"]
            .astype(str).str.strip()
            ==
            selected_country
        ]

    # City


    if (
        selected_city != "All"
        and "city" in filtered_df.columns
    ):

        filtered_df = filtered_df[filtered_df["city"]
            .astype(str).str.strip()
            ==
            selected_city
        ]

    # Category

    if (
        selected_category != "All"
        and "category" in filtered_df.columns
    ):

        filtered_df = filtered_df[filtered_df["category"]
            .astype(str).str.strip()
            ==
            selected_category
        ]

    # Contract Type

    if (
        selected_contract != "All"
        and "contract_type" in filtered_df.columns
    ):

        filtered_df = filtered_df[ filtered_df["contract_type"]
            .astype(str).str.strip()
            ==
            selected_contract
        ]

    # Date

    if (
        selected_dates is not None
        and "created_date" in filtered_df.columns
    ):

        if isinstance(selected_dates, (tuple, list)):

            if len(selected_dates) == 2:

                start_date = selected_dates[0]
                end_date = selected_dates[1]

                filtered_df = filtered_df[
                    (filtered_df["created_date"].dt.date >= start_date )
                    &
                    (filtered_df["created_date"].dt.date<= end_date)
                ]

    return filtered_df