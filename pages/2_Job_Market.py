import pandas as pd
import plotly.express as px
import streamlit as st
from utils import load_data, apply_global_filters


# Page Configuration
st.set_page_config(page_title="Job Market | Job Market Intelligence",
    page_icon="📊",layout="wide")



# Load Data
df = load_data()
filtered_df = apply_global_filters(df)


# Header
st.title("📊 Job Market Analysis")

st.caption( "Explore hiring patterns across categories, roles, "
    "companies, time, and job-description keywords.")


# CURRENT DATA CHECK
if filtered_df.empty:
    st.warning( "No jobs match the current global filters." )
    st.stop()


# CATEGORY / ROLE VIEW

st.markdown("---")
st.subheader("📌 Category & Role Analysis")
view_type = st.radio("Choose view", ["Category Overview", "Role Drill-down"],
    horizontal=True)


# CATEGORY OVERVIEW
if view_type == "Category Overview":
    if "category" in filtered_df.columns:
        category_data = ( filtered_df["category"].dropna().astype(str).str.strip() )
        category_data = category_data[ category_data != "" ]
        category_counts = (  category_data.value_counts() .head(15) .rename_axis("Category")
                             .reset_index(name="Jobs"))
        if not category_counts.empty:
            category_counts = category_counts.sort_values( "Jobs", ascending=True )
            fig = px.bar(category_counts,x="Jobs", y="Category", orientation="h",text="Jobs",
                         title="Top Job Categories" )
            fig.update_traces( textposition="outside",cliponaxis=False )
            fig.update_layout(height=550,xaxis_title="Number of Jobs", yaxis_title="Category",
                margin=dict( l=10,r=60, t=60,b=20 ))
            st.plotly_chart(fig, use_container_width=True )
        else:
            st.info("No category data available.")
    else:
        st.info("Category column is not available." )

# ROLE DRILL-DOWN
else:
    if ( "category" in filtered_df.columns and "standard_role" in filtered_df.columns
    ):
        category_options = (filtered_df["category"] .dropna().astype(str) .str.strip() )
        category_options = sorted( category_options[ category_options != ""].unique().tolist())
        if category_options:
            selected_category = st.selectbox("Select Category",["All"] + category_options,
                key="role_category" )
            role_df = filtered_df.copy()
            if selected_category != "All":
                role_df = role_df[role_df["category"].astype(str).str.strip() == selected_category]

            role_data = (role_df["standard_role"].dropna().astype(str).str.strip())
            role_data = role_data[ role_data != "" ]
            role_counts = (role_data.value_counts().head(15)
                .rename_axis("Role").reset_index(name="Jobs") )
            if not role_counts.empty:
                role_counts = role_counts.sort_values("Jobs",ascending=True  )
                fig = px.bar( role_counts,x="Jobs", y="Role",orientation="h",
                    text="Jobs",title="Top Roles")
                fig.update_traces(textposition="outside",cliponaxis=False )
                fig.update_layout(height=550,xaxis_title="Number of Jobs",yaxis_title="Role",
                    margin=dict( l=10, r=60, t=60, b=20 ) )
                st.plotly_chart(fig, use_container_width=True )
            else:
                st.info( "No roles available for this category." )
        else:
            st.info("No categories available.")
    else:
        st.info(  "Required category/role columns are unavailable.")


# TOP COMPANIES
st.markdown("---")
st.subheader("🏢 Top Hiring Companies")
if "company_name" in filtered_df.columns:
    company_data = (filtered_df["company_name"].dropna().astype(str).str.strip() )
    company_data = company_data[company_data != ""]
    company_counts = (company_data.value_counts().head(15).rename_axis("Company").reset_index(name="Jobs") )
    if not company_counts.empty:
        company_counts = company_counts.sort_values( "Jobs",ascending=True )
        fig_company = px.bar(company_counts,x="Jobs",y="Company", orientation="h",
            text="Jobs", title="Top Hiring Companies")
        fig_company.update_traces(textposition="outside",cliponaxis=False )
        fig_company.update_layout(height=600,xaxis_title="Number of Jobs", yaxis_title="Company",
            margin=dict( l=10,r=60,t=60,b=20 ) )
        st.plotly_chart( fig_company,use_container_width=True  )
    else:
        st.info( "No company data available.")
else:
    st.info( "company_name column is not available." )

# HIRING TREND
st.markdown("---")
st.subheader("📈 Hiring Trend")
if "created_date" in filtered_df.columns:
    trend_df = filtered_df.copy()
    trend_df["created_date"] = pd.to_datetime(trend_df["created_date"],errors="coerce" )
    trend_df = trend_df.dropna(subset=["created_date"] )
    if not trend_df.empty:
        monthly_trend = ( trend_df
            .assign( Month=trend_df["created_date"].dt.to_period("M").astype(str) )
            .groupby("Month")
            .size() .reset_index(name="Jobs"))

        fig_trend = px.line(monthly_trend,x="Month", y="Jobs", markers=True, title="Job Postings Over Time")

        fig_trend.update_layout( xaxis_title="Month",yaxis_title="Job Postings", height=450,
            margin=dict(l=10,r=20,t=60,b=20 ))

        st.plotly_chart(fig_trend, use_container_width=True )
    else:
        st.info("No valid date data available." )

else:
    st.info( "created_date column is not available." )


# SKILLS / KEYWORDS
st.markdown("---")
st.subheader("🧠 Skills / Keywords in Job Descriptions")
st.caption( "This section measures how often selected keywords appear "
    "in job descriptions. It is a keyword-presence analysis, "
    "not a full NLP-based skill extraction system.")

skill_keywords = ["Python", "SQL","Excel","Power BI","Tableau", "AWS","Azure", "GCP","Java","JavaScript",
    "React","Node.js", "Docker","Kubernetes","Spark","TensorFlow", "PyTorch","scikit-learn",
    "Machine Learning", "Deep Learning"]

if "content_clean" in filtered_df.columns:
    description_data = ( filtered_df["content_clean"].fillna("").astype(str) )
    keyword_results = []
    total_description_rows = len(description_data)
    if total_description_rows > 0:
        for keyword in skill_keywords:
            count = description_data.str.contains(keyword,case=False,na=False, regex=False).sum()
            percentage = ((count / total_description_rows) * 100)
            keyword_results.append({"Keyword": keyword,"Jobs": count,"Percentage": percentage  })

    skills_df = pd.DataFrame( keyword_results )
    skills_df = skills_df[ skills_df["Jobs"] > 0]
    skills_df = skills_df.sort_values( "Jobs",ascending=True ).tail(15)

    if not skills_df.empty:
        fig_skills = px.bar( skills_df,x="Jobs", y="Keyword",orientation="h",text="Jobs",
            title="Most Frequently Mentioned Skills / Keywords")

        fig_skills.update_traces(textposition="outside",cliponaxis=False)

        fig_skills.update_layout( height=550,xaxis_title="Jobs Mentioning Keyword", yaxis_title="Keyword",
            margin=dict( l=10, r=60, t=60,b=20))

        st.plotly_chart( fig_skills,use_container_width=True )

        st.caption(
            "Counts represent the number of filtered job descriptions "
            "containing each keyword."
        )
    else:
        st.info( "No selected keywords were found in the current descriptions.")
else:
    st.info( "content_clean column is not available." )


# FILTER SUMMARY


st.markdown("---")

st.caption( f"Showing analysis for {len(filtered_df):,} filtered postings "
    f"out of {len(df):,} total postings.")

