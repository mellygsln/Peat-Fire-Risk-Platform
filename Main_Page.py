import streamlit as st

st.set_page_config(page_title="Beranda", layout="wide")

st.title("Welcome to Peat Fire Risk Platform")
st.write("Please use the **sidebar** menu to access the other page.")

st.write("**Platform Objective**")
st.write("""
This platform aims to support early detection and risk mitigation of peatland fires by providing data-driven predictions using machine learning models. It is designed for researchers, environmental agencies, and decision-makers working on peatland sustainability and fire prevention.
""")

st.markdown("---")
st.write("### 🗂️ Information Page")
st.write("##### 1. Main Page")
st.write("""
Contain general information about the platform.
""")

st.write("##### 2. Prediction Page")
st.write("""
Prediction of time series data and fire risk:
- **Predict Time Series**: Using **Long Short-Term Memory (LSTM)** for 4 data types: temperature, water table, soil moisture, and rainfall.
- **Predict Fire Risk**: There are 3 methods that can be used to predict the risk of peatland fires: **XGBoost, SVM, and Random Forest**.
""")


st.markdown("---")
st.markdown("### ⚠️ Important Information")
st.write("""
The prediction results are generated using machine learning models and should be interpreted with caution.  
Always consult experts or relevant authorities before making any decisions or taking action.
""")

st.markdown("##### 📊 Data Sources")
st.write("""
This platform uses environmental data including temperature, water table depth, soil moisture, and rainfall.

- **www.bmkg.go.id**: Source of temperature data.  
- **www.prims.brgm.go.id**: Source of water table, soil moisture, and rainfall data.

Please pay attention to the data units used:

- **Temperature**: Celcius (°C)  
- **Water Table**: meters (m)  
- **Soil Moisture**: percentage (%)  
- **Rainfall**: millimeters (mm)
""")

st.markdown("---")
st.write("##### 👨‍💻 About the Developer")
st.write("""
This platform was developed by Melly.  
For questions or technical issues, feel free to contact via email: guslainimelly@gmail.com
""")
