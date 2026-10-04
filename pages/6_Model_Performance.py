import pandas as pd
import streamlit as st




# PAGE CONFIG

st.set_page_config(page_title="Model Performance | Job Market Intelligence",
    page_icon="📊",
    layout="wide",
)


# PAGE HEADER
st.title("📊 Model Performance")

st.caption(
    "Evaluation results for the trained machine-learning models "
    "used in the Job Market Intelligence platform."
)


# MODEL OVERVIEW
st.markdown("---")
st.subheader("📌 Model Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 💰 Salary Prediction")

    st.write("**Model:** Ridge Regression")
    st.write("**Task:** Salary midpoint prediction")
    st.write("**Target:** INR-normalized salary midpoint")
    st.write("**Validation:** GroupKFold")


with col2:
    st.markdown("### 💵 Salary Availability")

    st.write("**Model:** Tuned XGBoost")
    st.write("**Task:** Binary classification")
    st.write("**Target:** Salary information availability")
    st.write("**Validation:** Stratified Cross-Validation")


with col3:

    st.markdown("### 📄 Contract Type")

    st.write("**Model:** Tuned XGBoost")
    st.write("**Task:** Binary classification")
    st.write("**Target:** Permanent / Contract")
    st.write("**Validation:** StratifiedGroupKFold")



# SALARY PREDICTION PERFORMANCE
st.markdown("---")
st.subheader("💰 Salary Prediction Performance")


st.markdown("### Development Cross-Validation")

cv_col1, cv_col2, cv_col3 = st.columns(3)

with cv_col1:
    st.metric("MAE","₹43.89 Lakh")

with cv_col2:
    st.metric("RMSE","₹59.26 Lakh")

with cv_col3:
    st.metric("R²","0.105")

st.markdown("### Final Holdout")

holdout_col1, holdout_col2, holdout_col3 = st.columns(3)

with holdout_col1:
    st.metric("MAE","₹36.27 Lakh")

with holdout_col2:
    st.metric( "RMSE","₹50.72 Lakh"  )

with holdout_col3:
    st.metric("R²","0.328")


st.info(
    "The final holdout represents unseen job records from the same "
    "overall company pool. It should not be interpreted as an "
    "unseen-employer generalization score."
)




# SALARY AVAILABILITY PERFORMANCE
st.markdown("---")
st.subheader("💵 Salary Availability Performance")

st.markdown("### Final Holdout")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy","82.05%")

with col2:
    st.metric( "Precision","76.89%")

with col3:
    st.metric("Recall","71.76%" )

col4, col5, col6 = st.columns(3)
with col4:
    st.metric( "F1-score","74.23%")

with col5:
    st.metric("ROC-AUC","89.08%" )

with col6:
    st.metric("Average Precision","0.83")


st.info(
    "The final holdout contained 1,454 job postings and was kept "
    "locked during model development and tuning. The reported "
    "classification results use the predefined 0.50 threshold."
)

# SALARY AVAILABILITY CONFUSION MATRIX
st.markdown("### Confusion Matrix")

confusion_matrix = pd.DataFrame(
    {
        "Predicted: No Salary": [817, 148],
        "Predicted: Salary Available": [113, 376],
    },
    index=["Actual: No Salary",
        "Actual: Salary Available",
    ],
)

st.dataframe(confusion_matrix, use_container_width=True)

st.caption(
    "True Negatives = 817 | False Positives = 113 | "
    "False Negatives = 148 | True Positives = 376"
)


# CONTRACT TYPE PERFORMANCE

st.markdown("---")
st.subheader("📄 Contract Type Performance")


# DEVELOPMENT CROSS-VALIDATION

st.markdown("### Development Inner Cross-Validation")

cv_col1, cv_col2, cv_col3, cv_col4, cv_col5 = st.columns(5)

with cv_col1:
    st.metric("Accuracy","73.19%" )
with cv_col2:
    st.metric("Precision","71.28%")
with cv_col3:
    st.metric("Recall","60.98%" )
with cv_col4:
    st.metric( "F1-score", "65.60%")
with cv_col5:
    st.metric("ROC-AUC","0.7648")


# FINAL HOLDOUT
st.markdown("### Final Holdout")

holdout_col1, holdout_col2, holdout_col3 = st.columns(3)

with holdout_col1:
    st.metric("Accuracy", "66.67%"  )

with holdout_col2:
    st.metric( "Precision","66.67%")

with holdout_col3:
    st.metric("Recall","45.45%")

holdout_col4, holdout_col5, holdout_col6 = st.columns(3)

with holdout_col4:
    st.metric("F1-score","54.05%" )

with holdout_col5:
    st.metric( "ROC-AUC","0.7069" )

with holdout_col6:
    st.metric("Average Precision","0.6704" )


st.info(
    "The final holdout contained 51 postings from 36 previously "
    "unseen companies, with zero company overlap. The evaluation "
    "used the predefined 0.50 classification threshold."
)




# CONTRACT TYPE CONFUSION MATRIX

st.markdown("### Confusion Matrix")

contract_confusion_matrix = pd.DataFrame(
    {
        "Predicted: Permanent": [24, 12],
        "Predicted: Contract": [5, 10],
    },
    index=[
        "Actual: Permanent",
        "Actual: Contract",
    ],
)

st.dataframe(contract_confusion_matrix,use_container_width=True)

st.caption(
    "True Negatives = 24 | False Positives = 5 | "
    "False Negatives = 12 | True Positives = 10"
)




# CONTRACT TYPE UNCERTAINTY
st.markdown("### 📐 Uncertainty Analysis")

st.caption(
    "95% company-cluster bootstrap confidence intervals "
    "based on 5,000 resamples."
)


bootstrap_ci = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
        ],
        "Point Estimate": [
            "0.6667",
            "0.6667",
            "0.4545",
            "0.5405",
            "0.7069",
        ],
        "95% Confidence Interval": [
            "0.4773 – 0.8222",
            "0.3846 – 0.9000",
            "0.2273 – 0.6875",
            "0.2963 – 0.7308",
            "0.5298 – 0.8549",
        ],
    }
)


st.dataframe(bootstrap_ci, use_container_width=True, hide_index=True)


st.info(
    "The relatively wide confidence intervals reflect the small "
    "number of holdout postings and companies."
)


# OVERALL INTERPRETATION & LIMITATIONS
st.markdown("---")
st.subheader("📝 Model Interpretation & Limitations")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 💰 Salary Prediction")
    st.write(
        "The salary model estimates the INR-normalized midpoint "
        "of an advertised salary range."
    )

    st.write(
        "The salary dataset was relatively small and concentrated "
        "among a limited number of companies."
    )

    st.write(
        "The model should therefore be treated as an experimental "
        "salary-estimation component."
    )


with col2:
    st.markdown("### 💵 Salary Availability")

    st.write(
        "The XGBoost classifier estimates whether a job posting "
        "is likely to contain salary information."
    )

    st.write(
        "Salary-disclosure patterns can differ across sources, "
        "locations, categories, and other posting characteristics."
    )

    st.write(
        "Predictions represent statistical model outputs rather "
        "than causal explanations of salary disclosure."
    )


with col3:
    st.markdown("### 📄 Contract Type")

    st.write(
        "The XGBoost classifier distinguishes between permanent "
        "and contract postings."
    )

    st.write(
        "Only 256 of 7,269 postings had contract-type labels, "
        "and all labeled observations came from Adzuna."
    )

    st.write(
        "The final holdout contained only 51 postings from "
        "36 previously unseen companies, producing substantial "
        "uncertainty."
    )


st.info(
    "Model performance should be interpreted in the context of "
    "the evaluation design, dataset coverage, and sample size. "
    "Reported metrics are fixed evaluation results and are not "
    "recomputed by the Global Filters on this page."
)