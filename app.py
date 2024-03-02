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
        return mvalues_df

def convert_data_types(data, column_mapping):
    try:
        for data_type, columns in column_mapping.items():
            for column_name in columns:
                if data_type == "int":
                    data[column_name] = data[column_name].astype(int)
                elif data_type == "float":
                    data[column_name] = data[column_name].astype(float)
                elif data_type == "object":
                    data[column_name] = data[column_name].astype(object)
                elif data_type == "datetime":
                    data[column_name] = pd.to_datetime(data[column_name])
        outputs = ""
        for data_type, columns in column_mapping.items():
            outputs += f"Columns: {(', ').join(columns)} were successfully converted to {data_type} data type and "
        st.session_state.outputs.append(outputs[:-4]+"!")
        return 1
    except ValueError as e:
        st.error(f"Error converting selected columns to their specified data types: {e}")
        return 0


def impute_missing_values(data, column_mapping):
    try:
        for imputation_method, columns in column_mapping.items():
            for column_name in columns:
                if imputation_method == "Mean":
                    data[column_name].fillna(data[column_name].mean(), inplace=True)
                elif imputation_method == "Median":
                    data[column_name].fillna(data[column_name].median(), inplace=True)
                elif imputation_method == "Mode":
                    data[column_name].fillna(data[column_name].mode()[0], inplace=True)
        outputs = ""
        for imputation_method, columns in column_mapping.items():
            outputs += f"Imputed missing values in {(', ').join(columns)} by their corresponding {imputation_method} value and "
        st.session_state.outputs.append(outputs[:-4]+" successfully!")
        return 1
    except ValueError as e:
        st.error(f"Error imputing missing values for selected columns: {e}")
        return 0

def main():
    if 'column_mapping_for_imputation' not in st.session_state:
        st.session_state['column_mapping_for_imputation'] = {}

    if 'column_mapping_for_conversion' not in st.session_state:
        st.session_state['column_mapping_for_conversion'] = {}

    if "outputs" not in st.session_state:
        st.session_state.outputs = []
    st.title("No-Code ML Model Building App")
    
    st.sidebar.header("Upload Data")
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None
    st.session_state['uploaded_file'] = st.sidebar.file_uploader("Upload your CSV file")
    

    if st.session_state['uploaded_file'] is not None:
        # Display uploaded data
        if 'data' not in st.session_state:
            st.session_state['data'] = pd.read_csv(st.session_state['uploaded_file'])
        df = st.session_state['data'].copy()
        st.write("Uploaded Data:")
        st.write(st.session_state['data'])

        # Display data summary
        display_data_summary(st.session_state['data'])    
        
        # Display missing values and options for handling them
        mvalues_df = display_missing_values(st.session_state['data'])   

        # Missing value imputation
        st.sidebar.subheader("Impute Missing Values")
        imputation_in_progress = st.sidebar.checkbox("Enable Imputation")
        if imputation_in_progress:
            if mvalues_df is None:
                st.error("No Missing Values Found")
            else:
                options = list(mvalues_df['Column Name'].values)
                column_group = st.multiselect("Select columns for this imputation:", options=options)
                imputation_method = st.selectbox("Select imputation method for selected columns:", options=["Mean", "Median", "Mode"])
                if st.button("Add Imputation") and column_group:
                    if imputation_method in st.session_state['column_mapping_for_imputation'].keys():
                        st.session_state['column_mapping_for_imputation'][imputation_method] += column_group
                    else:
                        st.session_state['column_mapping_for_imputation'][imputation_method] = column_group
                

                if st.button("Reset Imputation Selection"):
                    st.session_state['column_mapping_for_imputation'] = {}
                st.write(st.session_state['column_mapping_for_imputation'])
                if st.button("Impute") and len(st.session_state['column_mapping_for_imputation'])!=0:
                    impute_status = impute_missing_values(st.session_state['data'], st.session_state['column_mapping_for_imputation'])
                    st.session_state['column_mapping_for_imputation'] = {}

                    # Display updated data after imputation
                    if impute_status:
                        st.subheader("Updated Data")
                        st.write(st.session_state['data'])
                        mvalues_df = display_missing_values(st.session_state['data'])



        #Data type conversion
        st.sidebar.subheader("Convert Data Types")
        conversion_in_progress = st.sidebar.checkbox("Enable Conversion")
        if conversion_in_progress:
            options_for_conv = list(st.session_state['data'].columns)
            column_group_for_conv = st.multiselect("Select columns for this conversion:", options=options_for_conv)
            new_data_type = st.selectbox("Select new data type for selected columns:", options=["int", "float", "object", "datetime"])
            if st.button("Add Conversion") and column_group_for_conv:
                if new_data_type in st.session_state['column_mapping_for_conversion'].keys():
                    st.session_state['column_mapping_for_conversion'][new_data_type] += column_group_for_conv
                else:
                    st.session_state['column_mapping_for_conversion'][new_data_type] = column_group_for_conv

            if st.button("Reset Conversion Selection"):
                st.session_state['column_mapping_for_conversion'] = {}
            st.write(st.session_state['column_mapping_for_conversion'])
            if st.button("Convert") and len(st.session_state['column_mapping_for_conversion'])!=0:
                conv_status = convert_data_types(st.session_state['data'], st.session_state['column_mapping_for_conversion'])
                st.session_state['column_mapping_for_conversion'] = {}

                # Display updated data after imputation
                if conv_status:
                    st.subheader("Updated Data")
                    st.write(st.session_state['data'])        

        for output in st.session_state['outputs']:
            st.write(output)

            # Button to restart the app
        if st.sidebar.button("Restart App"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.experimental_rerun()

    
if __name__ == "__main__":
    main()
