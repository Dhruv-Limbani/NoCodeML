import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from scipy.stats import pointbiserialr
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
import pickle
import io

def update_view(df,p1,p2,p3,p4):
    p1.dataframe(df)
    a,b,c = show_summary(df)
    p2.dataframe(a)
    p3.dataframe(b)
    p4.dataframe(c)
    
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
            "Count": [0 if df.shape[1]==0 else df.shape[0], df.shape[1]]
        }

    return summary_df, dtype_counts, size_df

def get_missing_values_df(data):
    missing_values_count = data.isnull().sum()
    missing_values_count = missing_values_count[missing_values_count > 0]  # Filter columns with missing values
    if missing_values_count.empty:
        return pd.DataFrame({})
    else:
        mvalues_df = missing_values_count.reset_index()
        mvalues_df.columns = ['Column Name', 'Missing Value Count']
        return mvalues_df

def handle_missing_vals(df, method, mvalues_df):
    if method == 1:
        if st.button("Confirm"):
            df.dropna(inplace=True)
    else:   
        if mvalues_df is None:
            return df
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

def get_unique_values_df(df,columns):
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
    
    return uniq_val_df

def replace(df, col, old, new):
    df[col].replace(old, new, inplace = True)

def change_data_type(df):
    disp_col_name = []
    disp_method_name = []
    options_for_conv = list(df.columns)
    column_group_for_conv = st.multiselect("Select columns for this conversion:", options=options_for_conv)
    new_data_type = st.selectbox("Select new data type for selected columns:", options=["int", "float", "object", "datetime"])
    if st.button("Add Conversion") and column_group_for_conv:
        if new_data_type in st.session_state['column_mapping_for_conversion'].keys():
            st.session_state['column_mapping_for_conversion'][new_data_type] += column_group_for_conv
        else:
            st.session_state['column_mapping_for_conversion'][new_data_type] = column_group_for_conv

    if st.button("Reset Conversion Selection"):
        st.session_state['column_mapping_for_conversion'] = {}

    for mthd, c_group in st.session_state['column_mapping_for_conversion'].items():
        for c_name in c_group:
            if c_name not in disp_col_name:
                disp_col_name.append(c_name)
                disp_method_name.append(mthd)
    
    st.dataframe(pd.DataFrame({"Column":disp_col_name,
                            "Method":disp_method_name}))
    if st.button("Convert") and len(st.session_state['column_mapping_for_conversion'])!=0:
        try:
            for data_type, columns in st.session_state['column_mapping_for_conversion'].items():
                for column_name in columns:
                    if data_type == "int":
                        df[column_name] = df[column_name].astype(int)
                    elif data_type == "float":
                        df[column_name] = df[column_name].astype(float)
                    elif data_type == "object":
                        df[column_name] = df[column_name].astype(object)
                    elif data_type == "datetime":
                        df[column_name] = pd.to_datetime(df[column_name])
            outputs = ""
            for data_type, columns in st.session_state['column_mapping_for_conversion'].items():
                outputs += f"Columns: {(', ').join(columns)} were successfully converted to {data_type} data type and "
            st.session_state.outputs.append(outputs[:-4]+"!")

        except ValueError as e:
            st.error(f"Error converting selected columns to their specified data types: {e}")

        st.session_state['column_mapping_for_conversion'] = {}

def rem_outliers(df, col_options):
    disp_col_name = []
    disp_method_name = []
    threshold = 3 # for z-score
    column_group = st.multiselect("Select columns for outlier removal:", options=col_options)
    or_method = st.selectbox("Select method for selected columns", options=["IQR", "Z-Score"])
    if st.button("Add") and column_group:
        if or_method in st.session_state['column_mapping_for_or'].keys():
            st.session_state['column_mapping_for_or'][or_method] += column_group
        else:
            st.session_state['column_mapping_for_or'][or_method] = column_group

    if st.button("Reset Selection"):
        st.session_state['column_mapping_for_or'] = {}

    for mthd, c_group in st.session_state['column_mapping_for_or'].items():
        for c_name in c_group:
            if c_name not in disp_col_name:
                disp_col_name.append(c_name)
                disp_method_name.append(mthd)
    
    st.dataframe(pd.DataFrame({"Column":disp_col_name,
                            "Method":disp_method_name}))
    if st.button("Remove Outliers") and len(st.session_state['column_mapping_for_or'])!=0:
        try:
            for met, columns in st.session_state['column_mapping_for_or'].items():
                for col in columns:
                    if met == "IQR":
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
                    elif met == "Z-Score":
                        z_scores = np.abs(stats.zscore(df[col]))
                        df =  df[z_scores < threshold]
            outputs = ""
            for met, columns in st.session_state['column_mapping_for_or'].items():
                outputs += f"Outliers were removed from columns: {(', ').join(columns)} using {met} technique and "
            st.session_state.outputs.append(outputs[:-4]+"!")

        except ValueError as e:
            st.error(f"Error removing outliers: {e}")

        st.session_state['column_mapping_for_or'] = {}

def show_outlier_detection(df, numerical_columns, method, target_cls=""):
    
    if method==1:
        for column in numerical_columns:

            fig, axes = plt.subplots(1,2,figsize=(15,4))

            # Box Plot
            sns.boxplot(y=df[column],ax=axes[0])
            axes[0].set_title(f'Box plot of {column}')

            # Histogram
            sns.histplot(df[column], kde=True, bins=30, ax=axes[1])
            axes[1].set_title(f'Histogram of {column}')

            st.pyplot(fig)
    
    if method==0:
        stats = df.groupby(target_cls).describe().T
        st.dataframe(stats)

def bivariate_categorical(df, col1, col2):
    fig, axes = plt.subplots(1,2,figsize=(15,4))
    ct = pd.crosstab(df[col2],df[col1], normalize = 'index')
    # Bar Plot
    st.write("Contingency Table")
    st.dataframe(ct)
    ct.plot(kind='bar', ax=axes[0], color=sns.color_palette('Set2'))
    axes[0].set_title(f'Proportion of {col1.title()} by {col2.title()}')
    axes[0].set_xlabel(col2)
    axes[0].set_ylabel('Proportion')
    axes[0].legend(title=f'{col1.title()}')
    # Stacked Bar Plot
    ct.plot(kind='bar', stacked=True, ax=axes[1], color=sns.color_palette('Set2'))
    axes[1].set_title(f'Stacked Bar Chart of {col1.title()} by {col2.title()}')
    axes[1].set_xlabel(col2)
    axes[1].set_ylabel('Proportion')
    axes[1].legend(title=f'{col1.title()}')
    st.pyplot(fig)

def show_EDA(df, columns, method):
    if method==0:
        column = st.selectbox("Choose variable for Univariate Analysis:", options = categorical_columns)
        if column:
            fig, axes = plt.subplots(1,2,figsize=(15,4))
            # Box Plot
            sns.countplot(x=column, data=df, ax=axes[0])
            axes[0].set_title(f'Box plot of {column}')

            # Histogram
            df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1])
            axes[1].set_title(f'Histogram of {column}')

            st.pyplot(fig)

            st.dataframe(df[column].value_counts())

    if method==1:
        column1 = st.selectbox("Choose variable 1 for Bivariate Analysis:", options = columns)
        column2 = st.selectbox("Choose variable 2 for Bivariate Analysis:", options = columns)
        if column1 and column2:
            if column1 in categorical_columns and column2 in categorical_columns:
                bivariate_categorical(df, column2, column1)
                flip_axis = st.checkbox("Flip axis")
                if flip_axis:
                    bivariate_categorical(df,column1,column2)
                
            else:
                cat = column1
                num = column2
                if (column2 in categorical_columns) and (column1 in numerical_columns):
                    cat = column2
                    num = column1
                
                if (cat in categorical_columns) and (num in numerical_columns):
                    fig, axes = plt.subplots(1,2,figsize=(15,4))
                    ct = pd.crosstab(df[cat],df[num], normalize = 'index')
                    # Histogram
                    sns.histplot(data=df, x=num, hue=cat, multiple='stack', kde=True, bins=20, ax=axes[0])
                    axes[0].set_title(f'Histogram of {num} by {cat}')

                    # Boxplot
                    sns.boxplot(data=df, x=cat, y=num, ax=axes[1])
                    axes[1].set_title(f'Boxplot of {num} by {cat}')

                    st.pyplot(fig)
                else:
                    st.write("Please select atleast one categorical column")
            
    if method==2:
        correlation = df[numerical_columns].corr()
        st.dataframe(correlation)
        fig = plt.figure(figsize=(8,4))
        a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
        a.set_xticklabels(a.get_xticklabels(), rotation=90)
        a = a.set_title('Correlation within Attributes')
        st.pyplot(fig)

def show_feat_impt(df, columns, target, method):
    if method==0:
        m = st.selectbox("Choose the ensemble model:",options=['Random Forest Classifier', 'Gradient Boosting Classifier'])
        if m == 'Random Forest Classifier':
            model = RandomForestClassifier(random_state=42)

            model.fit(df[columns], df[target])

            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': columns, 'Importance': importances})

            feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        else:
            model = GradientBoostingClassifier(random_state=42)

            model.fit(df[columns], df[target])

            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': columns, 'Importance': importances})

            feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        fig, axes = plt.subplots(1,2,figsize=(10, 6))
        axes[1].pie(feature_importance_df['Importance'], labels=feature_importance_df['Feature'], autopct='%1.1f%%')
        axes[1].set_title("Feature Importance")
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=axes[0])
        axes[0].set_title("Feature Importance")
        st.pyplot(fig)
    
    if method==1:
        X_encoded = df[columns].drop(target,axis=1).apply(LabelEncoder().fit_transform)
        y_encoded = LabelEncoder().fit_transform(df[target])
        m = st.selectbox("Choose the method:",options=['Chi-Square Test', 'ANOVA F-value', 'Point Biserial Correlation', 'Mutual Information', 'RandomForestClassifier'])
        if m == 'Chi-Square Test':
            chi2_values, p_values = chi2(X_encoded, y_encoded)
            importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Chi2': chi2_values, 'p-value': p_values})
            importance_df = importance_df.sort_values(by='Chi2', ascending=False)

            fig = plt.figure(figsize=(10, 6))
            a = sns.barplot(x='Chi2', y='Feature', data=importance_df)
            a = a.set_title("Feature Importance")
            st.pyplot(fig)    
        elif m == 'ANOVA F-value':
            f_values, p_values = f_classif(X_encoded, y_encoded)
            importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'F-Value': f_values, 'p-value': p_values})
            importance_df = importance_df.sort_values(by='F-Value', ascending=False)
            
            fig = plt.figure(figsize=(10, 6))
            a = sns.barplot(x='F-Value', y='Feature', data=importance_df)
            a = a.set_title("Feature Importance")
            st.pyplot(fig)  
        elif m == 'Point Biserial Correlation':
            correlations = []
            for column in X_encoded.columns:
                corr, _ = pointbiserialr(y_encoded, X_encoded[column])
                correlations.append((column, corr))
            correlation_df = pd.DataFrame(correlations, columns=['Feature', 'Point Biserial Correlation'])
            correlation_df = correlation_df.sort_values(by='Point Biserial Correlation', ascending=False)
            
            fig = plt.figure(figsize=(10, 6))
            a = sns.barplot(x='Point Biserial Correlation', y='Feature', data=correlation_df)
            a = a.set_title("Feature Importance")
            st.pyplot(fig)
            
        elif m == 'Mutual Information':
            mi_scores = mutual_info_classif(X_encoded, y_encoded)
            mi_df = pd.DataFrame({'Feature': X_encoded.columns, 'Mutual Information': mi_scores})
            mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

            fig = plt.figure(figsize=(10, 6))
            a = sns.barplot(x='Mutual Information', y='Feature', data=mi_df)
            a = a.set_title("Feature Importance")
            st.pyplot(fig)

        else:
            model = RandomForestClassifier()
            model.fit(X_encoded, y_encoded)
            importances = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            fig = plt.figure(figsize=(10, 6))
            a = sns.barplot(x='Importance', y='Feature', data=importance_df)
            a = a.set_title("Feature Importance")
            st.pyplot(fig)

def drop_cols(df):
    cols_to_drop = st.multiselect("Select columns to drop", options=df.columns)
    if cols_to_drop:
        if st.button("Drop"):
            df.drop(cols_to_drop, axis=1, inplace=True)
            st.success("Dropped!")

def generate_train_test_data(df):
    test_size = st.slider("Select Test Size (in %):", min_value=5.0, max_value=95.0, step=5.0)
    if st.button("split into train and test data"):
        st.session_state['train_data'], st.session_state['test_data'] = train_test_split(df, test_size=test_size/100, random_state=42)
        st.session_state['train_data'] = st.session_state['train_data'].reset_index(drop=True)
        st.session_state['test_data'] = st.session_state['test_data'].reset_index(drop=True)

        st.success("Train and Test Data Ready")
    
    if st.session_state['train_data'] is not None:  
        colt, colts = st.columns(2, gap='small')
        with colt:
            trb = st.button("Show Training Data Summary")
        with colts:
            tsb = st.button('Show Testing Data summary')
        
        if trb:
            a,b,c = show_summary(st.session_state['train_data'])
            st.write("Overview of Training Data")
            st.dataframe(st.session_state['train_data'].head())
            st.write("Summary of Training Data")
            col1, col2 = st.columns([3,2])
            with col1:
                sum_placeholder1 = st.empty()
                sum_placeholder1.dataframe(a)

            with col2:
                st.write("Count of Columns by Data type:")
                sum_placeholder2 = st.empty()
                sum_placeholder2.dataframe(b)

                st.write("Dataset Size: ")
                sum_placeholder3 = st.empty()
                sum_placeholder3.dataframe(c)
        if tsb:
            a,b,c = show_summary(st.session_state['test_data'])
            st.write("Overview of Testing Data")
            st.dataframe(st.session_state['test_data'].head())
            st.write("Summary of Testing Data")
            col1, col2 = st.columns([3,2])
            with col1:
                sum_placeholder1 = st.empty()
                sum_placeholder1.dataframe(a)

            with col2:
                st.write("Count of Columns by Data type:")
                sum_placeholder2 = st.empty()
                sum_placeholder2.dataframe(b)

                st.write("Dataset Size: ")
                sum_placeholder3 = st.empty()
                sum_placeholder3.dataframe(c)
        
        coldt, coldts = st.columns(2, gap="small")
        with coldt:
            st.download_button(
                label="Download Train Data as CSV",
                data=st.session_state['train_data'].to_csv(index=False).encode('utf-8'),
                file_name='train_data.csv',
                mime='text/csv',
            )
        with coldts:
            st.download_button(
                label="Download Test Data as CSV",
                data=st.session_state['test_data'].to_csv(index=False).encode('utf-8'),
                file_name='test_data.csv',
                mime='text/csv',
            )

def norm_encode_feats(df):
    disp_col_name = []
    disp_method_name = []
    column_group = st.multiselect("Select columns for transformation", options=df.columns)
    method = st.selectbox("Select transformation method", options=["Label Encoding", "One-Hot Encoding", "Ordinal Encoding", "Standard Scaling", "Min-Max Scaling"])

    if st.button("Add") and column_group:
        if method in st.session_state['column_mapping_for_tr'].keys():
            st.session_state['column_mapping_for_tr'][method] += column_group
        else:
            st.session_state['column_mapping_for_tr'][method] = column_group

    if st.button("Reset Selection"):
        st.session_state['column_mapping_for_tr'] = {}

    transformers = []
    for mthd, c_group in st.session_state['column_mapping_for_tr'].items():
        if len(c_group):
            if mthd == 'Standard Scaling':
                transformers.append(('standard_scaler', StandardScaler(), c_group))
            elif mthd == 'Min-Max Scaling':
                transformers.append(('minmax_scaler', MinMaxScaler(), c_group))
            if mthd == 'One-Hot Encoding':
                transformers.append(('onehot_encoder', OneHotEncoder(sparse_output=False), c_group))
            elif mthd == 'Ordinal Encoding':
                transformers.append(('ordinal_encoder', OrdinalEncoder(), c_group))

        for c_name in c_group:
            if c_name not in disp_col_name:
                disp_col_name.append(c_name)
                disp_method_name.append(mthd)
    
    st.dataframe(pd.DataFrame({"Column":disp_col_name,
                            "Transformation Method":disp_method_name}))
    if st.button("Transform Columns") and len(st.session_state['column_mapping_for_tr'])!=0:
        try:
            original_dtypes = df.dtypes.to_dict()

            st.session_state['ct'] = ColumnTransformer(
                transformers = transformers,
                remainder='passthrough' 
            )
            transformed_data = st.session_state['ct'].fit_transform(df)
            
            transformed_col_names = list(st.session_state['ct'].get_feature_names_out())
            passthrough_columns = [col for col in df.columns if col not in (st.session_state['column_mapping_for_tr']['Standard Scaling'] + st.session_state['column_mapping_for_tr']['Min-Max Scaling'] + st.session_state['column_mapping_for_tr']['One-Hot Encoding'] + st.session_state['column_mapping_for_tr']['Ordinal Encoding'])]
            
            cleaned_column_names = []
            for name in transformed_col_names:
                if 'remainder__' in name:
                    rem = 'remainder__'
                elif 'minmax_scaler__' in name:
                    rem = 'minmax_scaler__'
                elif 'standard_scaler__' in name:
                    rem = 'standard_scaler__'
                else:
                    rem = ''
                cleaned_column_names.append(name.replace(rem, ''))

            st.session_state['transformed_data'] = pd.DataFrame(transformed_data, columns=cleaned_column_names)

            for col in st.session_state['transformed_data'].columns:
                if col in passthrough_columns:
                    st.session_state['transformed_data'][col] = st.session_state['transformed_data'][col].astype(original_dtypes[col])
                elif col in (st.session_state['column_mapping_for_tr']['Standard Scaling'] + st.session_state['column_mapping_for_tr']['Min-Max Scaling']):
                    st.session_state['transformed_data'][col] = st.session_state['transformed_data'][col].astype(float)
            st.session_state['le'] = LabelEncoder()
            for col in st.session_state['column_mapping_for_tr']['Label Encoding']:
                st.session_state['transformed_data'][col] = st.session_state['le'].fit_transform(st.session_state['transformed_data'][col])
            
            outputs = ""
            for met, columns in st.session_state['column_mapping_for_tr'].items():
                outputs += f"Columns: {(', ').join(columns)} using {met} technique and "
            st.session_state.outputs.append(outputs[:-4]+"!")

            st.session_state['transformed_data']
        except ValueError as e:
            st.error(f"Error transforming columns: {e}")

        st.session_state['column_mapping_for_tr'] = {
            "Label Encoding": [],
            "One-Hot Encoding": [],
            "Ordinal Encoding": [],
            "Standard Scaling": [], 
            "Min-Max Scaling": []
        }

    if st.session_state['transformed_data'] is not None:          
        if st.button("Show Transformed Data Summary"):
            a,b,c = show_summary(st.session_state['transformed_data'])
            st.write("Overview of Transformed Data")
            st.dataframe(st.session_state['transformed_data'].head())
            st.write("Summary of Transformed Data")
            col1, col2 = st.columns([3,2])
            with col1:
                sum_placeholder1 = st.empty()
                sum_placeholder1.dataframe(a)

            with col2:
                st.write("Count of Columns by Data type:")
                sum_placeholder2 = st.empty()
                sum_placeholder2.dataframe(b)

                st.write("Dataset Size: ")
                sum_placeholder3 = st.empty()
                sum_placeholder3.dataframe(c)

        coldata, coldtrs = st.columns(2, gap="small")
        with coldata:
            st.download_button(
                label="Download Transformed Data as CSV",
                data=st.session_state['transformed_data'].to_csv(index=False).encode('utf-8'),
                file_name='transformed_data.csv',
                mime='text/csv',
            )
        with coldtrs:
            with io.BytesIO() as buffer:
                pickle.dump((st.session_state['ct'], st.session_state['le']), buffer)
                buffer.seek(0)
                st.download_button(
                    label="Download Column Transformer and Label Encoder as Pickle file",
                    data=buffer,
                    file_name="columntransformer_labelencoder.pkl",
                    mime="application/octet-stream"
                )

def handle_class_imbalane(df):
    target_column = st.selectbox("Select the target column", df.columns)
    sampling_technique = st.radio(
            "Choose a sampling technique to handle class imbalance",
            ("Oversampling (SMOTE)", "Undersampling")
        )


if 'column_mapping_for_imputation' not in st.session_state:
    st.session_state['column_mapping_for_imputation'] = {}

if 'column_mapping_for_conversion' not in st.session_state:
    st.session_state['column_mapping_for_conversion'] = {}

if 'column_mapping_for_or' not in st.session_state:
    st.session_state['column_mapping_for_or'] = {}

if 'column_mapping_for_tr' not in st.session_state:
    st.session_state['column_mapping_for_tr'] = {
        "Label Encoding": [],
        "One-Hot Encoding": [],
        "Ordinal Encoding": [],
        "Standard Scaling": [], 
        "Min-Max Scaling": []
    }

if "outputs" not in st.session_state:
    st.session_state.outputs = []

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'transformed_data' not in st.session_state:
    st.session_state['transformed_data'] = None

if 'train_data' not in st.session_state:
    st.session_state['train_data'] = None

if 'test_data' not in st.session_state:
    st.session_state['test_data'] = None

if 'ct' not in st.session_state:
    st.session_state['ct'] = None

if 'le' not in st.session_state:
    st.session_state['le'] = None

# changed = False

st.title("No-Code ML Model Building App")

st.session_state['uploaded_file'] = st.sidebar.file_uploader("Upload your data in CSV file format")

st.sidebar.write("Please rerun if you change your data")
if st.sidebar.button("Rerun"):
    st.session_state.clear()
    st.rerun()
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

    with col2:
        st.write("Count of Columns by Data type:")
        sum_placeholder2 = st.empty()
        sum_placeholder2.dataframe(b)

        st.write("Dataset Size: ")
        sum_placeholder3 = st.empty()
        sum_placeholder3.dataframe(c)
    
    task = st.sidebar.selectbox("Choose Task:", ['Select', 'Clean Data', 'Data Analysis and Visualization','Prepare Data for Model','Model Building'])

    if task == 'Clean Data':
        st.subheader("Data Cleaning",divider=True)
        miss_val_handling = st.sidebar.checkbox("Handle Missing Values")
        if miss_val_handling:
            st.subheader("Missing Values Handling")
            mvalues_df = get_missing_values_df(st.session_state['df'])
            if mvalues_df.empty:
                st.error("No missing values found!")
            else:
                st.write("Missing values for each column")
                mval_df_placeholder = st.empty()
                mval_df_placeholder.dataframe(mvalues_df)
                miss_val_handling_method = st.selectbox("Choose method for handling missing values",
                options = ["Select","Drop all the rows with missing value in any of it's colums","Imputation"])

                if miss_val_handling_method == "Drop all the rows with missing value in any of it's colums":
                    handle_missing_vals(st.session_state['df'],1,mvalues_df)
                    update_view(st.session_state['df'],df_placeholder, sum_placeholder1, sum_placeholder2, sum_placeholder3)
                    mvalues_df = get_missing_values_df(st.session_state['df'])
                    mval_df_placeholder.dataframe(mvalues_df)
                elif miss_val_handling_method == "Imputation":
                    handle_missing_vals(st.session_state['df'],2,mvalues_df)
                    update_view(st.session_state['df'],df_placeholder, sum_placeholder1, sum_placeholder2, sum_placeholder3)
                    mvalues_df = get_missing_values_df(st.session_state['df'])
                    mval_df_placeholder.dataframe(mvalues_df)

        replace_val = st.sidebar.checkbox("Replace Values")
        change_dt = st.sidebar.checkbox("Change Column Data Types")

        if replace_val or change_dt:
            uni_val_df_placeholder = st.empty()
            uni_val_df_placeholder.dataframe(get_unique_values_df(st.session_state['df'],st.session_state['df'].columns))

            if replace_val:
                st.subheader("Replace values in columns",divider=True)
                rep_col = st.selectbox("Choose the column for transformation", options = st.session_state['df'].columns)

                if rep_col:
                    rep_val = st.multiselect("Choose the value for replacing", options = st.session_state['df'][rep_col].unique())
                    if rep_val:
                        new_val = st.text_input("Enter the new value: ")
                        if st.button("Replace"):
                            if rep_col in numerical_columns and st.session_state['df'][rep_col].dtype == 'int64':
                                replace(st.session_state['df'], rep_col, rep_val, int(new_val))
                            elif rep_col in numerical_columns and st.session_state['df'][rep_col].dtype == 'float64':
                                replace(st.session_state['df'], rep_col, rep_val, float(new_val))
                            else:
                                replace(st.session_state['df'], rep_col, rep_val, new_val)
                            update_view(st.session_state['df'],df_placeholder, sum_placeholder1, sum_placeholder2, sum_placeholder3)
                            uni_val_df_placeholder.dataframe(get_unique_values_df(st.session_state['df'],st.session_state['df'].columns))

            if change_dt:
                st.subheader("Changing Data types of columns",divider=True)
                change_data_type(st.session_state['df'])
                update_view(st.session_state['df'],df_placeholder, sum_placeholder1, sum_placeholder2, sum_placeholder3)
                uni_val_df_placeholder.dataframe(get_unique_values_df(st.session_state['df'],st.session_state['df'].columns))

        outlier_d_r = st.sidebar.checkbox("Outlier Detection and Removal")
        if outlier_d_r:
            st.subheader("Outlier Detection:",divider=True)
            target_cls = st.selectbox("Select Target Class against which you want to perform outlier detection", options=st.session_state['df'].columns)
            numerical_columns_stats = st.checkbox("Show Statistics of Numerical Attributes by target class for Outlier Detection")
            if numerical_columns_stats:
                show_outlier_detection(st.session_state['df'],[],0,target_cls)
            
            numerical_columns_viz = st.checkbox("Visualize Numerical Attributes for Outlier Detection")
            if numerical_columns_viz:
                col_list_for_otl_detection = st.multiselect("Select Attributes for visualization", options=numerical_columns)
                if col_list_for_otl_detection:
                    show_outlier_detection(st.session_state['df'],col_list_for_otl_detection,1)
                else:
                    st.write("Please select at least one column to visualize.")

            outlier_r = st.checkbox("Remove Outliers")
            if outlier_r:
                rem_outliers(st.session_state['df'], numerical_columns)
                update_view(st.session_state['df'],df_placeholder, sum_placeholder1, sum_placeholder2, sum_placeholder3)

        download_cleaned_data = st.sidebar.checkbox("Download Cleaned Data")
        
        if download_cleaned_data:
            st.download_button(
            label="Download cleaned CSV",
            data=st.session_state['df'].to_csv(index=False).encode('utf-8'),
            file_name='cleaned_data.csv',
            mime='text/csv',
        )
                    
    if task == "Data Analysis and Visualization":
        st.subheader("Data Analysis and Visualization",divider=True)
        st.sidebar.header("Choose Tasks")
        unique_values = st.sidebar.checkbox("Unique Values")
        EDA = st.sidebar.checkbox("EDA")
        feat_impt = st.sidebar.checkbox("Measure Feature Importance")

        if unique_values:
            st.header("Unique Values:",divider=True)
            col_list_for_unique_vals = st.multiselect("Select Columns for Displaying Unique Values", options=st.session_state['df'].columns)
            uni_val_df_placeholder = st.empty()
            if col_list_for_unique_vals:
                uni_val_df_placeholder.dataframe(get_unique_values_df(st.session_state['df'],col_list_for_unique_vals))
            else:
                st.write("Please select at least one column to display its details.")
 
        if EDA:
            st.header("Exploratory Data Analysis:", divider=True)
            univar_analysis = st.checkbox("Show Univariate Analysis")
            if univar_analysis:
                show_EDA(st.session_state['df'],categorical_columns,0)
            
            bivar_analysis = st.checkbox("Show Bivariate Analysis")
            if bivar_analysis:
                show_EDA(st.session_state['df'],st.session_state['df'].columns,1)
                

            mulvar_analysis = st.checkbox("Show Multivariate Analysis")
            if mulvar_analysis:
                show_EDA(st.session_state['df'],numerical_columns,2)

        if feat_impt:
            st.header("Feature Importance", divider=True)
            target_cls_feat_impt = st.selectbox("Select Target Class to identify important features for.", options=categorical_columns)
            num_feat_impt = st.checkbox("Measure Feature Importance of Numerical Attributes")
            if num_feat_impt:
                show_feat_impt(st.session_state['df'],numerical_columns,target_cls_feat_impt,0)
            cat_feat_impt = st.checkbox("Measure Feature Importance of Categorical Attributes")
            if cat_feat_impt:
                show_feat_impt(st.session_state['df'],categorical_columns,target_cls_feat_impt,1)
    
    if task == "Prepare Data for Model":
        drop_ir_c = st.sidebar.checkbox("Drop Irrelevant Columns")
        if drop_ir_c:
            st.subheader("Drop Irrelevant Features",divider=True)
            drop_cols(st.session_state['df'])
            update_view(st.session_state['df'],df_placeholder, sum_placeholder1, sum_placeholder2, sum_placeholder3)
        tr_ts_split = st.sidebar.checkbox("Get train and test data")
        if tr_ts_split:
            st.subheader("Generate Train and Test Split", divider=True)
            generate_train_test_data(st.session_state['df'])

        norm_encode_cols = st.sidebar.checkbox("Normalize Numerical Attributes and Encode Categorical Features")
        if norm_encode_cols:
            st.subheader("Normalize and Encode Features", divider=True)
            norm_encode_feats(st.session_state['df'])

        handle_cls_imbalance = st.sidebar.checkbox("Handle Class Imbalance")
        if handle_cls_imbalance:
            st.subheader("Handle Class Imbalance (In development)", divider=True)
            st.warning("Sorry! This feature is in development phase")
            handle_class_imbalane(st.session_state['df'])

        dim_red = st.sidebar.checkbox("Dimensionality Reduction")
        if dim_red:
            st.subheader("Dimensionality Reduction", divider=True)
            st.warning("Sorry! This feature is in development phase")


    st.subheader("Your notes", divider=True)
    user_notes = st.text_area("Take your notes here:", key="user_notes", height=200)
    
    if user_notes:
        st.download_button(
            label="Download your notes",
            data=user_notes,
            file_name="data_cleaning_notes.txt",
            mime="text/plain"
        )

