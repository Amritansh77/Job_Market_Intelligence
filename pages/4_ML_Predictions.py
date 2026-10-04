import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from utils import load_data, apply_global_filters



# PAGE CONFIG
st.set_page_config( page_title="ML Predictions | Job Market Intelligence",
    page_icon="🤖",
    layout="wide",)



# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

SALARY_RIDGE_PATH = ( MODEL_DIR / "salary_ridge_evaluation_model.pkl")

SALARY_AVAILABILITY_PATH = (MODEL_DIR / "salary_availability_xgb_model.pkl")

CONTRACT_TYPE_PATH = ( MODEL_DIR / "contract_type_xgb_model_final.pkl")



# LOAD MODELS

salary_ridge_model = joblib.load(SALARY_RIDGE_PATH)

salary_availability_model = joblib.load(SALARY_AVAILABILITY_PATH)

contract_artifact = joblib.load( CONTRACT_TYPE_PATH)

contract_model = contract_artifact["model"]

contract_features = contract_artifact["features"]

contract_threshold = float(contract_artifact["threshold"])

contract_target_mapping = (contract_artifact["target_mapping"])


# ============================================================
# LOAD DATA + GLOBAL FILTERS
# ============================================================

df = load_data()

filtered_df = apply_global_filters(df)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_model_categories(model, feature_name):
    """
    Return categories learned by the model's fitted
    OneHotEncoder for a specific categorical feature.
    """

    preprocessor = model.named_steps["preprocessor"]

    for _, transformer, columns in preprocessor.transformers_:

        if transformer in ("drop", "passthrough"):
            continue

        columns = list(columns)

        if feature_name not in columns:
            continue

        feature_index = columns.index(feature_name )

        # Direct encoder
        if hasattr(transformer, "categories_"):

            return [str(value) for value in transformer.categories_[feature_index]]

        # Encoder inside pipeline
        if hasattr(transformer, "named_steps"):

            for step in transformer.named_steps.values():

                if hasattr(step, "categories_"):

                    return [str(value) for value in step.categories_[feature_index]]

    return []


def title_function(title):
    """
    Rule-based title function used by the salary model.
    """

    title = str(title).lower()

    if any(word in title for word in [
        "counsel",
        "legal",
        "attorney",
        "lawyer",
    ]):
        return "Legal"

    elif any(word in title for word in [
        "account executive",
        "sales",
        "sales development",
        "business development",
        "account manager",
    ]):
        return "Sales"

    elif any(word in title for word in [
        "data scientist",
        "data analyst",
        "data engineer",
        "machine learning",
        "artificial intelligence",
        "ai engineer",
        "ml engineer",
    ]):
        return "Data"

    elif any(word in title for word in [
        "engineer",
        "engineering",
        "developer",
        "architect",
    ]):
        return "Engineering"

    elif any(word in title for word in [
        "product manager",
        "product director",
        "director of product",
        "product",
    ]):
        return "Product"

    elif any(word in title for word in [
        "recruiter",
        "human resources",
        "hr ",
        "people",
    ]):
        return "Human Resources"

    elif any(word in title for word in [
        "customer success",
        "customer support",
        "support",
    ]):
        return "Customer Success"

    elif any(word in title for word in [
        "marketing",
        "brand",
        "communications",
    ]):
        return "Marketing"

    else:
        return "Other"


def title_features(title):
    """
    Create the 11 title-derived features used by
    the classification models.
    """

    title = ( ""
        if title is None
        else str(title).strip())

    lower_title = title.lower()

    result = { "title_word_count": len(  title.split() ),
        "title_length": len(title),
        "title_has_senior": int(
            bool(re.search( r"\bsenior\b|\bsr\.?\b",lower_title) )
        ),

        "title_has_lead": int(
            bool( re.search( r"\blead\b", lower_title ) )
        ),

        "title_has_manager": int(
            bool(re.search( r"\bmanager\b", lower_title ) )
        ),

        "title_has_director": int(
            bool(re.search( r"\bdirector\b", lower_title)  )
        ),

        "title_has_intern": int(
            bool(re.search( r"\bintern\b|\binternship\b", lower_title  ) )
        ),

        "title_has_junior": int(
            bool(re.search(r"\bjunior\b|\bjr\.?\b", lower_title))
        ),

        "title_has_principal": int(
            bool( re.search(r"\bprincipal\b", lower_title) )
        ),

        "title_has_staff": int(
            bool( re.search( r"\bstaff\b", lower_title))
        ),

        "title_has_associate": int(
            bool( re.search(r"\bassociate\b",lower_title))
        ),
    }

    return result



# HEADER
st.title("🤖 ML Predictions")



# 1. SALARY PREDICTION
st.header("💰 Salary Prediction")

# JOB TITLE

job_title = st.text_input( "Job Title",
    placeholder="e.g. Data Scientist",
    key="salary_prediction_job_title")



# COMPANY + LOCATION

col1, col2 = st.columns(2)

with col1:

    salary_company_options = (get_model_categories( salary_ridge_model,"company_name" ) )

    salary_company_default = "Stripe"

    company = st.selectbox("Company", salary_company_options,
        index=(salary_company_options.index( salary_company_default )
            if salary_company_default
            in salary_company_options
            else 0),
        key="salary_prediction_company"
    )


with col2:
    salary_location_options = ( get_model_categories( salary_ridge_model, "location_clean") )

    # Prefer a commonly occurring single location
    # instead of a long multi-location string.
    location_counts = (df["location_clean"].dropna().astype(str).str.strip() .value_counts() )

    valid_location_counts = (location_counts[location_counts.index.isin( salary_location_options )])

    single_location_counts = (valid_location_counts[~valid_location_counts.index.str.contains(
                ";", regex=False, na=False ) ] )

    if not single_location_counts.empty:
        salary_location_default = (single_location_counts.index[0])

    elif not valid_location_counts.empty:
        salary_location_default = (valid_location_counts.index[0] )

    else:
        salary_location_default = (salary_location_options[0]  if salary_location_options  else "" )

    salary_location_index = (salary_location_options.index( salary_location_default)
        if salary_location_default
        in salary_location_options
        else 0
    )

    location = st.selectbox( "Location",salary_location_options,index=salary_location_index,
        key="salary_prediction_location" )



# CURRENCY + TITLE FUNCTION
col3, col4 = st.columns(2)

with col3:
    salary_currency_options = [  "AUD",  "CAD", "EUR", "GBP","USD", ]
    salary_currency = st.selectbox("Salary Currency",salary_currency_options,
        index=salary_currency_options.index( "USD"),
        key="salary_prediction_currency" )


with col4:
    salary_title_function_options = (get_model_categories(salary_ridge_model,"title_function") )

    # Automatically suggest title function from Job Title
    if job_title.strip():
        suggested_title_function = ( title_function(job_title) )
    else:
        suggested_title_function = "Data"

    # Update the suggestion only when the job title changes.
    # This allows the user to manually override the dropdown.
    if ( "salary_prediction_last_job_title"
        not in st.session_state
        or
        st.session_state[
            "salary_prediction_last_job_title"
        ] != job_title
    ):

        st.session_state[ "salary_prediction_title_function" ] = suggested_title_function
        st.session_state["salary_prediction_last_job_title"] = job_title

    selected_title_function = st.selectbox( "Title Function",salary_title_function_options,
        key="salary_prediction_title_function")

    st.caption(f"Suggested from job title: "
        f"{suggested_title_function}" )


# POSTING YEAR + MONTH

salary_date_source = ( filtered_df if not filtered_df.empty else df)

salary_dates = (salary_date_source[ ["posting_year","posting_month"]].dropna().copy())

if salary_dates.empty:
    salary_posting_year_default = 2019
    salary_posting_month_default = 1
else:
    latest_salary_date = (salary_dates.sort_values( ["posting_year", "posting_month"]).iloc[-1])

    salary_posting_year_default = int(latest_salary_date["posting_year"])

    salary_posting_month_default = int(latest_salary_date["posting_month"] )


date_col1, date_col2 = st.columns(2)

with date_col1:
    salary_posting_year = st.number_input("Posting Year",
        min_value=2019, max_value=2030,
        value=salary_posting_year_default,
        step=1, key="salary_prediction_year")

with date_col2:

    salary_posting_month = st.number_input("Posting Month",
        min_value=1,  max_value=12,
        value=salary_posting_month_default,
        step=1,key="salary_prediction_month")



# PREDICT BUTTON

predict_salary_button = st.button( "🔮 Predict Salary",
    type="primary",use_container_width=True,
    key="predict_salary_button")

# SALARY PREDICTION

if predict_salary_button:
    if not job_title.strip():
        st.warning("Please enter a job title.")
    else:
        # Posting date
        posting_year = int(salary_posting_year )

        posting_month = int( salary_posting_month)

        # Title features
        title = job_title.strip()
        lower_title = title.lower()

        title_word_count = len(title.split())
        title_length = len(title)
        title_has_senior = int("senior" in lower_title)
        title_has_lead = int("lead" in lower_title)
        title_has_manager = int("manager" in lower_title)
        title_has_director = int("director" in lower_title)
        title_has_intern = int("intern" in lower_title)
        title_has_principal = int("principal" in lower_title )
        title_has_staff = int( "staff" in lower_title)
        title_has_associate = int("associate" in lower_title )

        # Exact 16 Ridge features
        salary_input = pd.DataFrame(
            [
                {
                    "company_name": company,
                    "salary_currency": salary_currency,
                    "location_clean": location,
                    "title_function": selected_title_function,
                    "posting_year": posting_year,
                    "posting_month": posting_month,
                    "title_word_count": title_word_count,
                    "title_length": title_length,
                    "title_has_senior": title_has_senior,
                    "title_has_lead": title_has_lead,
                    "title_has_manager": title_has_manager,
                    "title_has_director": title_has_director,
                    "title_has_intern": title_has_intern,
                    "title_has_principal": title_has_principal,
                    "title_has_staff": title_has_staff,
                    "title_has_associate": title_has_associate,
                }
            ]
        )

        # Prediction
        predicted_salary_inr = float(salary_ridge_model.predict( salary_input )[0])

        # Result
        st.markdown("---")
        st.subheader( "💰 Predicted Salary")
        result_col1, result_col2 = (st.columns(2) )
        with result_col1:
            st.metric("Estimated Salary (INR)",
                f"₹{predicted_salary_inr:,.0f}"
            )

        with result_col2:
            if predicted_salary_inr >= 10_000_000:
                salary_scale = (f"₹{predicted_salary_inr / 10_000_000:.2f} "
                    "Crore"
                )

            else:
                salary_scale = (f"₹{predicted_salary_inr / 100_000:.2f} "
                    "Lakh"
                )

            st.metric("Salary Scale",salary_scale )

        st.caption(
            "Model estimate of the INR-normalized salary "
            "midpoint. This is an experimental estimate, "
            "not a guaranteed salary."
        )



# 2. SALARY AVAILABILITY
st.markdown("---")
st.header("💵 Salary Availability")



# MODEL CATEGORIES
availability_source_options = ( get_model_categories( salary_availability_model, "source" ))

availability_category_options = ( get_model_categories(salary_availability_model, "category" ))

availability_role_options = ( get_model_categories( salary_availability_model,"standard_role"))

availability_location_options = (get_model_categories(salary_availability_model,"location_clean" ))



# SOURCE + CATEGORY
col1, col2 = st.columns(2)
with col1:
    availability_source_default = "Adzuna"
    availability_source = st.selectbox("Source",
        availability_source_options,
        index=( availability_source_options.index(
                availability_source_default
            )
            if availability_source_default
            in availability_source_options
            else 0
        ),
        key="availability_source"
    )


with col2:

    availability_category_default = "IT Jobs"

    availability_category = st.selectbox(
        "Category",
        availability_category_options,
        index=(
            availability_category_options.index(
                availability_category_default
            )
            if availability_category_default
            in availability_category_options
            else 0
        ),
        key="availability_category"
    )



# STANDARD ROLE + LOCATION

col3, col4 = st.columns(2)

with col3:

    availability_role_default = ("DevOps Engineer" )

    availability_standard_role = st.selectbox(
        "Standard Role",
        availability_role_options,
        index=(availability_role_options.index(
                availability_role_default
            )
            if availability_role_default
            in availability_role_options
            else 0
        ),
        key="availability_standard_role"
    )


with col4:
    availability_location_default = ("Indore, Madhya Pradesh" )
    availability_location = st.selectbox("Location",
        availability_location_options,
        index=(
            availability_location_options.index(
                availability_location_default
            )
            if availability_location_default
            in availability_location_options
            else 0
        ),
        key="availability_location"
    )



# JOB TITLE

availability_job_title = st.text_input( "Job Title", placeholder="e.g. Data Scientist",
    key="availability_job_title")


# POSTING YEAR + MONTH
col5, col6 = st.columns(2)

with col5:
    availability_posting_year = (
        st.number_input( "Posting Year", min_value=2019, max_value=2030,
            value=2019, step=1,  key="availability_posting_year")
    )


with col6:
    availability_posting_month = (
        st.number_input( "Posting Month", min_value=1, max_value=12,
            value=6, step=1, key="availability_posting_month")
    )


# PREDICT BUTTON
availability_predict_button = st.button(
    "🔮 Predict Salary Availability",
    type="primary",
    use_container_width=True,
    key="availability_predict_button"
)



# PREDICTION
if availability_predict_button:
    if not availability_job_title.strip():
        st.warning(  "Please enter a job title.")
    else:
        # Title features
        availability_title_features = (title_features( availability_job_title ))

        # Exact 17 features

        availability_input = pd.DataFrame(
            [
                {
                    "source": availability_source,
                    "category": availability_category,
                    "standard_role": ( availability_standard_role),
                    "location_clean": ( availability_location),
                    "posting_year": int(availability_posting_year),
                    "posting_month": int(availability_posting_month),
                    **availability_title_features,
                }
            ]
        )

        # Prediction

        predicted_class = int(salary_availability_model.predict(availability_input )[0] )


        # Probabilities
        probabilities = (salary_availability_model.predict_proba(availability_input )[0])

        classes = (salary_availability_model.named_steps["model"].classes_)

        probability_map = {int(cls): float(probability)
            for cls, probability
            in zip(classes,probabilities) }


        salary_available_probability = ( probability_map.get(1,0.0))

        salary_unavailable_probability = ( probability_map.get(0, 0.0 ))


        # Result
        st.markdown("---")

        st.subheader("💵 Salary Availability Result")

        result_col1, result_col2 = ( st.columns(2))
        with result_col1:
            if predicted_class == 1:
                st.success(
                    "Salary information is predicted "
                    "to be available."
                )

            else:
                st.info(
                    "Salary information is predicted "
                    "to be unavailable."
                )


        with result_col2:
            if predicted_class == 1:
                st.metric(
                    "Probability of Salary Available",
                    f"{salary_available_probability * 100:.1f}%"
                )

            else:

                st.metric(
                    "Probability of Salary Unavailable",
                    f"{salary_unavailable_probability * 100:.1f}%"
                )

        st.caption(
            "The model estimates whether the job posting "
            "is likely to contain salary information. "
            "This is a statistical prediction, not a "
            "causal explanation."
        )

# 3. CONTRACT TYPE
st.markdown("---")
st.header("📄 Contract Type")

# MODEL CATEGORIES
contract_category_options = ( get_model_categories( contract_model,"category"))
contract_role_options = (get_model_categories( contract_model,"standard_role"))

contract_location_options = ( get_model_categories( contract_model,"location_clean"))

contract_time_options = (get_model_categories(contract_model,"contract_time"))

# CATEGORY + STANDARD ROLE

col1, col2 = st.columns(2)


with col1:
    contract_category_default = "IT Jobs"

    contract_category = st.selectbox("Category",
        contract_category_options,
        index=( contract_category_options.index(
                contract_category_default)
            if contract_category_default
            in contract_category_options
            else 0
        ),
        key="contract_prediction_category"
    )


with col2:
    contract_role_default = ( "DevOps Engineer" )
    contract_standard_role = st.selectbox("Standard Role",
        contract_role_options,
        index=(contract_role_options.index(
                contract_role_default )
            if contract_role_default
            in contract_role_options
            else 0
        ),
        key="contract_prediction_role"
    )

# LOCATION + CONTRACT TIME

col3, col4 = st.columns(2)
with col3:
    contract_location_default = ("Indore, Madhya Pradesh" )

    contract_location = st.selectbox("Location",
        contract_location_options,
        index=(contract_location_options.index(
                contract_location_default
            )
            if contract_location_default
            in contract_location_options
            else 0
        ),
        key="contract_prediction_location"
    )


with col4:
    # Prefer full-time if available.
    contract_time_default = "full_time"

    contract_time = st.selectbox( "Contract Time",
        contract_time_options,
        index=(contract_time_options.index(
                contract_time_default
            )
            if contract_time_default
            in contract_time_options
            else 0
        ),
        key="contract_prediction_time"
    )


# JOB TITLE

contract_job_title = st.text_input(
    "Job Title",
    placeholder="e.g. DevOps Engineer",
    key="contract_prediction_job_title"
)

# POSTING YEAR + MONTH

col5, col6 = st.columns(2)

with col5:
    contract_posting_year = (
        st.number_input( "Posting Year",
            min_value=2019, max_value=2030,
            value=2026, step=1,
            key="contract_prediction_year"
        )
    )

with col6:
    contract_posting_month = (
        st.number_input(
            "Posting Month",
            min_value=1,
            max_value=12,
            value=1,
            step=1,
            key="contract_prediction_month"
        )
    )


# PREDICT BUTTON
contract_predict_button = st.button(
    "🔮 Predict Contract Type",
    type="primary",
    use_container_width=True,
    key="contract_predict_button"
)


# CONTRACT TYPE PREDICTION

if contract_predict_button:
    if not contract_job_title.strip():
        st.warning("Please enter a job title.")

    else:
        # Title features
        contract_title_features = (title_features( contract_job_title))

        # Build feature dictionary
        contract_values = {

            "location_clean": contract_location,
            "category":contract_category,
            "standard_role":contract_standard_role,
            "contract_time":contract_time,
            "posting_year":int(contract_posting_year),
            "posting_month":int(contract_posting_month),
            **contract_title_features,
        }


        # Verify feature schema
        missing_features = [feature
            for feature in contract_features
            if feature not in contract_values ]


        if missing_features:
            st.error(
                "Missing Contract Type model features: "
                f"{missing_features}"
            )

        else:
            # Exact saved feature order
            contract_input = pd.DataFrame(
                [
                    { feature: contract_values[ feature]
                        for feature
                        in contract_features
                    }
                ]
            )



            # Prediction
            predicted_class = int(contract_model.predict(contract_input )[0] )

            # Probabilities
            probabilities = (contract_model.predict_proba(contract_input )[0] )

            classes = (contract_model.named_steps["model"].classes_)

            probability_map = {int(cls): float(probability)
                for cls, probability
                in zip( classes,probabilities )
            }

            predicted_probability = (probability_map[predicted_class ])

            # Target mapping
            reverse_mapping = { int(value): str(key)
                for key, value
                in contract_target_mapping.items()
            }

            predicted_contract = ( reverse_mapping.get( predicted_class, str(predicted_class)))

            # Result
            st.markdown("---")
            st.subheader("📄 Contract Type Result" )
            result_col1, result_col2 = (st.columns(2) )

            with result_col1:
                st.metric(
                    "Predicted Contract Type",
                    predicted_contract.title()
                )


            with result_col2:
                st.metric(
                    "Prediction Confidence",
                    f"{predicted_probability * 100:.1f}%"
                )

            st.caption(
                f"Classification threshold: "
                f"{contract_threshold:.2f}"
            )



