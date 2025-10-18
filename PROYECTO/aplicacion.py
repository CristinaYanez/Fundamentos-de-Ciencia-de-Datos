import streamlit as st
import pandas as pd
import joblib 

import os
kmeans_pipeline=joblib.load(os.path.join(os.path.dirname(__file__), "kmeans_pipeline.pkl"))
rf_pipeline=joblib.load(os.path.join(os.path.dirname(__file__), "random_forest_pipeline.pkl"))

products = pd.read_csv('archive/products.csv')
team = pd.read_csv('archive/sales_teams.csv') 
accounts=pd.read_csv('archive/accounts.csv') 
sector_options = accounts['sector'].dropna().unique()
location_options=accounts['office_location'].fillna('').unique()
st.title("Simulador de Ventas")

# kmeans_pipeline = joblib.load("kmeans_pipeline.pkl")
# rf_pipeline = joblib.load("random_forest_pipeline.pkl")
    
st.header("Información del cliente")
with st.form("cluster_input_form"):
    sector = st.selectbox("Sector", options=sorted(sector_options))
    year_established = st.number_input("Año de Fundación", min_value=1800, max_value=2025, value=2000)
    revenue = st.number_input("Ingresos (USD)", min_value=0.0, value=100000.0)
    employees = st.number_input("Número de Empleados", min_value=1, value=50)
    office_location = st.selectbox("Ubicación de Oficina", options=sorted(location_options))
    product = st.selectbox("Producto", products['product'].unique())
    submitted = st.form_submit_button("Siguiente") 
    if submitted:
        input_df = pd.DataFrame([{
            'sector': sector,
            'office_location': office_location,
            'year_established': year_established,
            'revenue': revenue,
            'employees': employees, 
        }]) 
        cluster = kmeans_pipeline.predict(input_df)[0]
        st.success(f"El clúster asignado es: {cluster}")
        model_df=team
        model_df['Clúster']=cluster
        model_df['product']=product
        model_df=model_df.merge(products, on='product', how='left')
        predictions = rf_pipeline.predict(model_df)
        model_df['predicción']=predictions
        model_df['predicción']=model_df['predicción'].apply(lambda x: round(x,2))
        max_row = model_df.loc[model_df['predicción'].idxmax()]
        st.write(max_row)



