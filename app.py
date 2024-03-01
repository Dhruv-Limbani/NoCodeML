import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def display_data_summary(data):
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Number of rows:   {data.shape[0]}")
    with col2:
        st.write(f"Number of columns:   {data.shape[1]}")
    col3, col4 = st.columns(2)
    with col3:
        st.write("Data Types:")
        dtypes_df = data.dtypes.reset_index()
        dtypes_df.columns = ['Column Name', 'Data Type']
        st.write(dtypes_df)
    with col4:
        st.write("Basic Statistics:")
        st.write(data.describe())

def display_missing_values(data):
    st.subheader("Missing Values")
    missing_values_count = data.isnull().sum()
    missing_values_count = missing_values_count[missing_values_count > 0]  # Filter columns with missing values
    if missing_values_count.empty:
        st.write("No missing values found!")
    else:
        st.write("Number of missing values for each column:")
        mvalues_df = missing_values_count.reset_index()
        mvalues_df.columns = ['Column Name', 'Missing Value Count']
        st.write(mvalues_df)

def main():
    st.title("No-Code ML Model Building App")
    
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
         # Display uploaded data
        st.write("Uploaded Data:")
        st.write(data)
        
        # Display data summary
        display_data_summary(data)    

        # Display missing values and options for handling them
        display_missing_values(data)   
        
    
if __name__ == "__main__":
    main()
