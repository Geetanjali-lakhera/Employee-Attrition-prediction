import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(layout="wide")

# Load model
with open('attrition_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict"])

# Load data
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Dashboard
if page == "Dashboard":
    st.title("Employee Attrition Analysis")

    # Sidebar Filters (UPDATED)
    st.sidebar.header("Filters")
    monthly_income = st.sidebar.slider("Monthly Income", int(df['MonthlyIncome'].min()), int(df['MonthlyIncome'].max()), (3000, 10000))
    monthly_rate = st.sidebar.slider("Monthly Rate", int(df['MonthlyRate'].min()), int(df['MonthlyRate'].max()), (5000, 20000))
    age = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), (25, 50))
    years_at_company = st.sidebar.slider("Years at Company", int(df['YearsAtCompany'].min()), int(df['YearsAtCompany'].max()), (1, 10))
    hourly_rate = st.sidebar.slider("Hourly Rate", int(df['HourlyRate'].min()), int(df['HourlyRate'].max()), (30, 100))

    # Apply filters
    filtered_df = df[
        (df['MonthlyIncome'].between(*monthly_income)) &
        (df['MonthlyRate'].between(*monthly_rate)) &
        (df['Age'].between(*age)) &
        (df['YearsAtCompany'].between(*years_at_company)) &
        (df['HourlyRate'].between(*hourly_rate))
    ]

    attrition_rate = round(filtered_df['Attrition'].mean(), 2)
    first_attr = "Yes" if attrition_rate > 0 else "No"

    col1, col2, col3 = st.columns(3)
    col1.metric("Sum of Attrition Rate", attrition_rate)
    col2.metric("First Attrition", first_attr)
    col3.metric("Total Employees", len(filtered_df))

    st.subheader("Count of Attrition by Years At Company")
    line_df = filtered_df.groupby('YearsAtCompany')['Attrition'].sum().reset_index()
    st.line_chart(line_df.set_index('YearsAtCompany'))

    st.subheader("Count of Attrition Rate by Job Role")
    bar_df = filtered_df.groupby('JobRole')['Attrition'].sum().sort_values(ascending=False)
    st.bar_chart(bar_df)

    st.subheader("Sum of Attrition Rate by Gender")
    pie_df = filtered_df.groupby('Gender')['Attrition'].sum().reset_index()
    fig = px.pie(pie_df, names='Gender', values='Attrition', title='Attrition by Gender')
    st.plotly_chart(fig)

    st.subheader("Attrition Summary Table by JobRole")
    table_df = filtered_df.groupby(['JobRole', 'Attrition']).size().unstack(fill_value=0)
    table_df['Total'] = table_df.sum(axis=1)
    st.dataframe(table_df)

    st.subheader("Top Features Affecting Attrition (Model-Based)")
    importances = model.feature_importances_
    features = model.feature_names_in_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    feat_imp_df['Importance (%)'] = feat_imp_df['Importance'] * 100

    sns.barplot(x='Importance (%)', y='Feature', data=feat_imp_df.head(10), palette="viridis")
    st.pyplot(plt.gcf())
    plt.clf()

# Prediction Page
elif page == "Predict":
    st.title("Employee Attrition Analysis")

    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    OverTime = st.selectbox("OverTime", ["No", "Yes"])
    Age = st.slider("Age", 18, 60, 30)
    DailyRate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
    TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)
    MonthlyRate = st.number_input("Monthly Rate", min_value=1000, max_value=30000, value=10000)
    HourlyRate = st.number_input("Hourly Rate", min_value=30, max_value=150, value=60)
    DistanceFromHome = st.slider("Distance From Home", 1, 30, 5)
    YearsAtCompany = st.slider("Years At Company", 0, 40, 5)
    NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 10, 1)
    PercentSalaryHike = st.slider("Percent Salary Hike", 10, 25, 15)
    JobRole = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
        "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"
    ])

    over_time_enc = 1 if OverTime == "Yes" else 0
    jobrole_mapping = {
        "Sales Executive": 7, "Research Scientist": 6, "Laboratory Technician": 3,
        "Manufacturing Director": 1, "Healthcare Representative": 2, "Manager": 4,
        "Sales Representative": 8, "Research Director": 5, "Human Resources": 0
    }

    input_data = pd.DataFrame([{
        'MonthlyIncome': MonthlyIncome,
        'OverTime': over_time_enc,
        'Age': Age,
        'DailyRate': DailyRate,
        'TotalWorkingYears': TotalWorkingYears,
        'MonthlyRate': MonthlyRate,
        'HourlyRate': HourlyRate,
        'DistanceFromHome': DistanceFromHome,
        'YearsAtCompany': YearsAtCompany,
        'NumCompaniesWorked': NumCompaniesWorked,
        'PercentSalaryHike': PercentSalaryHike,
        'JobRole': jobrole_mapping.get(JobRole, 0)
    }])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        result = "⚠️ Can Leave" if prediction[0] == 1 else "✅ Can Stay"
        st.subheader(f"Prediction: {result}")
