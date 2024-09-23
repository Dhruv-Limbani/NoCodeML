import streamlit as st
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# def display_data_summary(data):
#     st.subheader("Data Summary")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.write(f"Number of rows:   {data.shape[0]}")
#     with col2:
#         st.write(f"Number of columns:   {data.shape[1]}")
#     col3, col4 = st.columns(2)
    
#     with col3:
#         st.write("Data Types:")
        
#         dtypes_df = data.dtypes.reset_index()
        
#         dtypes_df.columns = ['Column Name', 'Data Type']
#         st.write(dtypes_df)
        
#     with col4:
#         st.write("Basic Statistics:")
#         st.write(data.describe())


def display_missing_values(data):
    missing_values_count = data.isnull().sum()
    missing_values_count = missing_values_count[missing_values_count > 0]  # Filter columns with missing values
    if missing_values_count.empty:
        st.write("No missing values found!")
    else:
        st.write("Number of missing values for each column:")
        mvalues_df = missing_values_count.reset_index()
        mvalues_df.columns = ['Column Name', 'Missing Value Count']
        return mvalues_df

# def convert_data_types(data, column_mapping):
#     try:
#         for data_type, columns in column_mapping.items():
#             for column_name in columns:
#                 if data_type == "int":
#                     data[column_name] = data[column_name].astype(int)
#                 elif data_type == "float":
#                     data[column_name] = data[column_name].astype(float)
#                 elif data_type == "object":
#                     data[column_name] = data[column_name].astype(object)
#                 elif data_type == "datetime":
#                     data[column_name] = pd.to_datetime(data[column_name])
#         outputs = ""
#         for data_type, columns in column_mapping.items():
#             outputs += f"Columns: {(', ').join(columns)} were successfully converted to {data_type} data type and "
#         st.session_state.outputs.append(outputs[:-4]+"!")
#         return 1
#     except ValueError as e:
#         st.error(f"Error converting selected columns to their specified data types: {e}")
#         return 0

# def go_to_data_version(dfv_dict):
#     ver = st.selectbox("Choose the version of data", options = dfv_dict.keys())

#     show_summary(dfv_dict[ver])

#     chs_this = st.button("Choose this as your current data version")

#     if chs_this:
#         st.session_state['df'] = dfv_dict[ver]

# def save_df_version(df,label):
#     st.session_state.df_version[label] = df

def handle_missing_vals(df, method, mvalues_df):
    if method == 1:
        df.dropna(inplace=True)
    else:   
        if mvalues_df is None:
            st.error("No Missing Values Found")
        else:
            disp_col_name = []
            disp_method_name = []
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
            
            
            for mthd, c_group in st.session_state['column_mapping_for_imputation'].items():
                for c_name in c_group:
                    if c_name not in disp_col_name:
                        disp_col_name.append(c_name)
                        disp_method_name.append(mthd)
            
            st.dataframe(pd.DataFrame({"Column":disp_col_name,
                                    "Method":disp_method_name}))

            if st.button("Impute") and len(st.session_state['column_mapping_for_imputation'])!=0:
                try:
                    for imputation_method, columns in st.session_state['column_mapping_for_imputation'].items():
                        for column_name in columns:
                            if imputation_method == "Mean":
                                df[column_name].fillna(df[column_name].mean(), inplace=True)
                            elif imputation_method == "Median":
                                df[column_name].fillna(df[column_name].median(), inplace=True)
                            elif imputation_method == "Mode":
                                df[column_name].fillna(df[column_name].mode()[0], inplace=True)
                    outputs = ""
                    for imputation_method, columns in st.session_state['column_mapping_for_imputation'].items():
                        outputs += f"Imputed missing values in {(', ').join(columns)} by their corresponding {imputation_method} value and "
                    st.session_state.outputs.append(outputs[:-4]+" successfully!")
                except ValueError as e:
                    st.error(f"Error imputing missing values for selected columns: {e}")
                st.session_state['column_mapping_for_imputation'] = {}
                return df


def show_summary(df):
    # Extract summary information manually
    summary = {
        "Column": df.columns,
        "Non-Null Count": [df[col].notnull().sum() for col in df.columns],
        "Dtype": [df[col].dtype for col in df.columns]
    }
    # Create a DataFrame for the summary
    summary_df = pd.DataFrame(summary)

    # Count of columns by dtype
    dtype_counts = df.dtypes.value_counts().reset_index()
    dtype_counts.columns = ["Dtype", "Column Count"]

    size_df = {
            "Axis" : ["Samples","Features"],
            "Count": [df.shape[0], df.shape[1]]
        }

    return summary_df, dtype_counts, size_df

def show_unique_values(df,columns):
    # Create a list to store the summary data
    uniq_val_data = []
    
    for col in columns:
        dtype = df[col].dtype
        unique_values = df[col].unique()
        strg = ""
        for uv in unique_values[:-1]:
            strg = strg + f"{uv}, "
        strg = strg + f"{unique_values[-1]}"
        
        # Add the column data to the summary list
        uniq_val_data.append({
            "Column": col,
            "Data Type": dtype,
            "Unique Values (sorted)": strg
        })
    
    # Create a DataFrame from the summary data
    uniq_val_df = pd.DataFrame(uniq_val_data)
    
    # Display the summary in a table format
    st.dataframe(uniq_val_df)

uf = None

def main():
    global uf

    if 'column_mapping_for_imputation' not in st.session_state:
        st.session_state['column_mapping_for_imputation'] = {}

    if 'column_mapping_for_conversion' not in st.session_state:
        st.session_state['column_mapping_for_conversion'] = {}

    if "outputs" not in st.session_state:
        st.session_state.outputs = []

    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

    if 'df' not in st.session_state:
        st.session_state['df'] = None


    st.title("No-Code ML Model Building App")
    
    st.session_state['uploaded_file'] = st.sidebar.file_uploader("Upload your data in CSV file format")
    if st.session_state['uploaded_file'] is not None:
        if st.session_state['df'] is None:
            st.session_state['df'] = pd.read_csv(st.session_state['uploaded_file'])
        st.write("Uploaded Data:")
        df_placeholder = st.empty()
        df_placeholder.dataframe(st.session_state['df'], use_container_width=True)

        numerical_columns = st.session_state['df'].select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = [x for x in st.session_state['df'].columns if x not in numerical_columns]

        a,b,c = show_summary(st.session_state['df'])
        col1, col2 = st.columns([3,2])
        with col1:
            sum_placeholder1 = st.empty()
            sum_placeholder1.dataframe(a)

        # Display the count of columns by dtype
        with col2:
            st.write("Count of Columns by Data type:")
            sum_placeholder2 = st.empty()
            sum_placeholder2.dataframe(b)

            st.write("Dataset Size: ")
            sum_placeholder3 = st.empty()
            sum_placeholder3.dataframe(c)
        
        task = st.sidebar.selectbox("Choose Task:", ['Clean Data', 'Data Analysis and Visualization','Model Building',"Change Data Version"])

        if task == 'Clean Data':

            miss_val_handling = st.sidebar.checkbox("Handle Missing Values")
            if miss_val_handling:
                st.subheader("Missing Values Handling")
                mval_df_placeholder = st.empty()
                mvalues_df = display_missing_values(st.session_state['df'])
                mval_df_placeholder.dataframe(mvalues_df)
                miss_val_handling_method = st.selectbox("Choose method for handling missing values",
                options = ["Select","Drop all the rows with missing value in any of it's colums","Imputation"])

                if miss_val_handling_method == "Drop all the rows with missing value in any of it's colums":
                    handle_missing_vals(st.session_state['df'],1,mvalues_df)
                    df_placeholder.dataframe(st.session_state['df'])
                    a,b,c = show_summary(st.session_state['df'])
                    sum_placeholder1.dataframe(a)
                    sum_placeholder2.dataframe(b)
                    sum_placeholder3.dataframe(c)
                    mvalues_df = display_missing_values(st.session_state['df'])
                    mval_df_placeholder.dataframe(mvalues_df)
                elif miss_val_handling_method == "Imputation":
                    handle_missing_vals(st.session_state['df'],2,mvalues_df)
                    df_placeholder.dataframe(st.session_state['df'])
                    a,b,c = show_summary(st.session_state['df'])
                    sum_placeholder1.dataframe(a)
                    sum_placeholder2.dataframe(b)
                    sum_placeholder3.dataframe(c)
                    mvalues_df = display_missing_values(st.session_state['df'])
                    mval_df_placeholder.dataframe(mvalues_df)

        
        # # Display data summary
        # display_data_summary(st.session_state['data'])    
        
        # # Display missing values and options for handling them
        # mvalues_df = display_missing_values(st.session_state['data'])   


        # st.sidebar.subheader("Get Columns Info")
        # columns_info_in_progress = st.sidebar.checkbox("Enable Column Analysis")
        # if columns_info_in_progress:
        #     st.write(type(st.session_state['data']['bath']))

        # # Missing value imputation
        # st.sidebar.subheader("Impute Missing Values")
        # imputation_in_progress = st.sidebar.checkbox("Enable Imputation")
        # if imputation_in_progress:
        #     if mvalues_df is None:
        #         st.error("No Missing Values Found")
        #     else:
        #         options = list(mvalues_df['Column Name'].values)
        #         column_group = st.multiselect("Select columns for this imputation:", options=options)
        #         imputation_method = st.selectbox("Select imputation method for selected columns:", options=["Mean", "Median", "Mode"])
        #         if st.button("Add Imputation") and column_group:
        #             if imputation_method in st.session_state['column_mapping_for_imputation'].keys():
        #                 st.session_state['column_mapping_for_imputation'][imputation_method] += column_group
        #             else:
        #                 st.session_state['column_mapping_for_imputation'][imputation_method] = column_group
                

        #         if st.button("Reset Imputation Selection"):
        #             st.session_state['column_mapping_for_imputation'] = {}
        #         st.write(st.session_state['column_mapping_for_imputation'])
        #         if st.button("Impute") and len(st.session_state['column_mapping_for_imputation'])!=0:
        #             impute_status = impute_missing_values(st.session_state['data'], st.session_state['column_mapping_for_imputation'])
        #             st.session_state['column_mapping_for_imputation'] = {}

        #             # Display updated data after imputation
        #             if impute_status:
        #                 st.subheader("Updated Data")
        #                 st.write(st.session_state['data'])
        #                 mvalues_df = display_missing_values(st.session_state['data'])



        # #Data type conversion
        # st.sidebar.subheader("Convert Data Types")
        # conversion_in_progress = st.sidebar.checkbox("Enable Conversion")
        # if conversion_in_progress:
        #     options_for_conv = list(st.session_state['data'].columns)
        #     column_group_for_conv = st.multiselect("Select columns for this conversion:", options=options_for_conv)
        #     new_data_type = st.selectbox("Select new data type for selected columns:", options=["int", "float", "object", "datetime"])
        #     if st.button("Add Conversion") and column_group_for_conv:
        #         if new_data_type in st.session_state['column_mapping_for_conversion'].keys():
        #             st.session_state['column_mapping_for_conversion'][new_data_type] += column_group_for_conv
        #         else:
        #             st.session_state['column_mapping_for_conversion'][new_data_type] = column_group_for_conv

        #     if st.button("Reset Conversion Selection"):
        #         st.session_state['column_mapping_for_conversion'] = {}
        #     st.write(st.session_state['column_mapping_for_conversion'])
        #     if st.button("Convert") and len(st.session_state['column_mapping_for_conversion'])!=0:
        #         conv_status = convert_data_types(st.session_state['data'], st.session_state['column_mapping_for_conversion'])
        #         st.session_state['column_mapping_for_conversion'] = {}

        #         # Display updated data after imputation
        #         if conv_status:
        #             st.subheader("Updated Data")
        #             st.write(st.session_state['data'])        

        # for output in st.session_state['outputs']:
        #     st.write(output)

        #     # Button to restart the app
        # if st.sidebar.button("Restart App"):
        #     st.cache_data.clear()
        #     st.cache_resource.clear()
        #     st.experimental_rerun()

    
if __name__ == "__main__":
    main()
