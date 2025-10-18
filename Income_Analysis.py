
import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(
    page_title="Adult Income Data Explorer", layout="wide")

st.title("💼 Adult Income Data Explorer")
st.markdown("""Welcome!""")

# -------------------------------
# Load and clean data
# -------------------------------
df = pd.read_csv("data.csv")
# Strip any hidden spaces in column names
df.columns = df.columns.str.strip()

st.header("Dataset Overview")
st.write(df.head())
st.write(f"Dataset shape: {df.shape}")

#Univariate Analysis

st.header("Univariate Analysis")
st.subheader("Age Distribution")
fig1 = px.histogram(df, x='age', nbins=30, title="Age Distribution")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Educational Levels")
fig2 = px.bar(df['education'].value_counts().reset_index(),
              x='education', y='count', title="Education Count")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Region of Origin Distribution")
fig3 = px.bar(df['region'].value_counts().reset_index(),
              x='region', y='count', title="Region Count")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Weekly Hours Worked")
fig4 = px.histogram(df, x='hours_per_week', nbins=30, title="Weekly Hours Distribution")
st.plotly_chart(fig4, use_container_width=True)

st.subheader("Gender Distribution")
fig11 = px.pie(df, names='gender', title="Gender Distribution", color='gender',
             color_discrete_map={'Male':"#313DEC", '    Female  ':"#3BDAEF"})
st.plotly_chart(fig11, use_container_width=True)


#Bivariate Analysis — Income Disparities

st.header("Bivariate Analysis — Income Disparities")
st.subheader("Income by Education")
education_order = ['Preschool', '1st to 4th Grade', '5th to 6th Grade', '7th to 8th Grade', '9th Grade', 
                   '10th Grade', '11th Grade', '12th Grade',
                   'High School Graduate', 'College',  "Associate's Degree (Academic)",
                   "Associate's Degree (Vocational)", "Bachelor's Degree",  "Master's Degree",
                   'Professional School', 'Doctorate/PhD']

fig7 = px.bar(df, x='education', color='income', barmode='group', title="Income by Education", 
              category_orders={'education': education_order})
st.plotly_chart(fig7, use_container_width=True)

st.subheader("Capital Activity by Income")
cap_income = df.groupby(['income', 'has_capital_activity']).size().reset_index(name='count')
fig5 = px.pie(cap_income, names='has_capital_activity', values='count', 
                     title="Capital Activity by Income",
                     color='has_capital_activity',
                     color_discrete_map={0:'#636EFA', 1:'#EF553B'},
                     facet_col='income')
st.plotly_chart(fig5, use_container_width=True)

st.subheader("Income by Gender")
fig8 = px.bar(df, x='gender', color='income', barmode='group', title="Income by Gender")
st.plotly_chart(fig8, use_container_width=True)


#Multivariate Analysis — Gender × Education × Income

st.header("Intersectional Trends — Gender × Education × Income")
st.subheader("Income by Education and Gender")
fig10 = px.bar(df, x='education', color='income', facet_col='gender',
               title="Income by Education and Gender", 
               category_orders={'education': education_order})
st.plotly_chart(fig10, use_container_width=True)

# End of the app
