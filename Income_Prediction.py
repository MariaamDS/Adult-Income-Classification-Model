import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(layout='wide', page_title='Income Classification App', page_icon='💰')

st.markdown("""
    <h1 style='text-align:center; color:#2C3E50;'>💰 Income Classification App</h1>
    <p style='text-align:center; color:gray;'>Predict whether a person earns above or below a certain income threshold</p>
""", unsafe_allow_html=True)

# --- Load Data ---
data_path = Path("data.csv")
model_path = Path("LGBM.pkl")

if not data_path.exists():
    st.error("❌ 'data.csv' not found. Please make sure it’s in the same folder as this script.")
    st.stop()

if not model_path.exists():
    st.error("❌ 'LGBM.pkl' model file not found. Please make sure it’s in the same folder as this script.")
    st.stop()

df = pd.read_csv(data_path)

with st.expander("📊 Preview Dataset"):
    st.dataframe(df.head())

#Initialize session state
for key in ['net_manual', 'emp_manual']:
    if key not in st.session_state:
        st.session_state[key] = None

#Sidebar Inputs
st.sidebar.header("⚙️ Numeric Inputs")

age = st.sidebar.slider("Age", 17, 90, 38)
capital_gain = st.sidebar.number_input("Capital Gain in USD", min_value=0.0, max_value=99999.0, value=0.0, step=100.0)
capital_loss = st.sidebar.number_input("Capital Loss in USD", min_value=0.0, max_value=4356.0, value=0.0, step=100.0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

#Net Capital
if capital_gain == 0 and capital_loss == 0:
    net_capital = st.sidebar.number_input("Net Capital in USD", min_value=-4356.0, max_value=99999.0, value=0.0, step = 100.0)
else:
    net_capital = capital_gain - capital_loss

st.session_state['net_manual'] = net_capital

st.info(f"💡 Net Capital in USD= {net_capital} (Calculated from gain and loss)")

#Has Capital Activity
if capital_gain != 0 or capital_loss != 0 or net_capital != 0:
    has_capital_activity = "Yes"
else:
    has_capital_activity = "No"
st.caption(f"Capital Activity: **{has_capital_activity}**")

#Employment Type
if hours_per_week < 35:
    default_emp = "Part_time"
elif hours_per_week <= 45:
    default_emp = "Full_time"
else:
    default_emp = "Overtime"

st.session_state['emp_manual'] = default_emp if st.session_state['emp_manual'] is None else st.session_state['emp_manual']

employment_type = st.sidebar.selectbox(
    "Employment Type",
    ['Part_time', 'Full_time', 'Overtime'],
    index=['Part_time', 'Full_time', 'Overtime'].index(st.session_state['emp_manual']),
    key='emp_manual'
)

#Main Page Layout
st.markdown("### 🧠 Categorical Inputs")
col1, col2, col3 = st.columns(3)

with col1:
    workclass = st.selectbox("Workclass", df['workclass'].dropna().unique())
    education = st.selectbox("Education", df['education'].dropna().unique())
    education_order = {'Preschool':1,
                       '1st to 4th Grade' :2,
                       '5th to 6th Grade': 3,
                       '7th to 8th Grade':4, 
                       '9th Grade': 5,
                       '10th Grade': 6,
                       '11th Grade': 7, '12th Grade': 8,
                       'High School Graduate': 9, 
                       'College':10, 
                       "Associate's Degree (Academic)":11,
                       "Associate's Degree (Vocational)":12,
                       "Bachelor's Degree":13,  
                       "Master's Degree":14,
                       'Professional School':15, 'Doctorate/PhD':16}

    # Convert education string → ordinal numeric value
    education_encoded = education_order.get(education, None)

    if education_encoded is None:
        st.error("Unknown education category selected — please verify the mapping.")
        st.stop()

    marital_status = st.selectbox("Marital Status", df['marital_status'].dropna().unique())


with col2:
    occupation = st.selectbox("Occupation", df['occupation'].dropna().unique())
    relationship = st.selectbox("Relationship", df['relationship'].dropna().unique())
    race = st.selectbox("Race", df['race'].dropna().unique())

with col3:
    gender = st.selectbox("Gender", df['gender'].dropna().unique())
    region = st.selectbox("Region", df['region'].dropna().unique())

#Load Model
model = joblib.load('LGBM.pkl')


input_cols = df.drop(['income', 'fnlwgt'], axis=1).columns

new_data = pd.DataFrame(columns=input_cols, data =[[age, workclass, education, marital_status,
                                                    occupation, relationship, race, gender, capital_gain,
                                                    capital_loss, hours_per_week, region, net_capital,
                                                    1 if has_capital_activity == 'Yes' else 0,                                                  
                                                    employment_type]])

#Prediction
st.markdown("## 🔍 Predict Income Level")

if st.button('Predict Income'):
    try:
        result = model.predict(new_data)[0]
        if result == 0:
            st.success("💼 Prediction: **Low Income (≤ $50k)**")
        else:
            st.warning("💰 Prediction: **High Income (> $50k)**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
