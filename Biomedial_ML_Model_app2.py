import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# Set page configuration
st.set_page_config(page_title="Biomedical ML Model Trainer and Predictor", layout="wide")

# Custom CSS for colors
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox>div>div {
        color: black;
    }
    .sidebar .sidebar-content {
        background-color: #e8e8e8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load and preprocess the dataset, handling NaN values
def load_and_preprocess_dataset(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.dropna(inplace=True)  # Drop rows with NaN values
        return df
    return None

# Function to preprocess the dataset
def preprocess_dataset(df, selected_columns):
    categorical_features = df[selected_columns].select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df[selected_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Imputers for handling NaN values
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numerical_features + categorical_features

# Function to train the model with GridSearchCV
# Function to train the model with GridSearchCV
def train_model(df, target, model_name, params, selected_columns):
    X = df[selected_columns]
    y = df[target]

    # Encode categorical target variable if necessary
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    preprocessor, feature_names = preprocess_dataset(df, selected_columns)
    preprocessor.fit(X_train)

    if model_name == 'KNN':
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', KNeighborsClassifier())])
        param_grid = {'classifier__n_neighbors': params['n_neighbors']}
    elif model_name == 'Logistic Regression':
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression(max_iter=params['max_iter']))])
        param_grid = {'classifier__C': params['C']}

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy', error_score='raise')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Check if the model supports predict_proba
    if hasattr(best_model, 'predict_proba'):
        try:
            y_score = best_model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
        except ValueError as e:
            st.error(f"Error calculating ROC AUC: {e}")
            roc_auc = "N/A (model does not support probability estimates)"
    else:
        roc_auc = "N/A (model does not support probability estimates)"

    class_report = classification_report(y_test, y_pred)

    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    with open('selected_columns.pkl', 'wb') as f:
        pickle.dump(selected_columns, f)

    return accuracy, precision, recall, f1, conf_matrix, roc_auc, class_report, feature_names
# Function to validate the model using cross-validation
def validate_model(df, target, model_name, params, selected_columns):
    X = df[selected_columns]
    y = df[target].values.ravel()  # Ensure y is a 1-dimensional array

    preprocessor, feature_names = preprocess_dataset(df, selected_columns)

    if model_name == 'KNN':
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', KNeighborsClassifier())])
        param_grid = {'classifier__n_neighbors': params['n_neighbors']}
    elif model_name == 'Logistic Regression':
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression(max_iter=params['max_iter']))])
        param_grid = {'classifier__C': params['C']}

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy')
    cross_val_scores = cross_val_score(grid_search, X, y, cv=5, scoring='accuracy')

    return cross_val_scores.mean(), cross_val_scores.std()

import pandas as pd
import pickle

def make_prediction(input_data):
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # Check input data columns
    expected_columns = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]
    if not all(column in input_data.columns for column in expected_columns):
        missing_columns = [column for column in expected_columns if column not in input_data.columns]
        raise ValueError(f"Missing columns in input data: {missing_columns}")

    # Transform input data
    input_data_transformed = preprocessor.transform(input_data)

    # Ensure transformed data is a DataFrame
    if not isinstance(input_data_transformed, pd.DataFrame):
        input_data_transformed = pd.DataFrame(input_data_transformed, columns=expected_columns)

    # Make predictions
    try:
        predictions = model.predict(input_data_transformed)
    except Exception as e:
        raise ValueError(f"Error predicting with the model: {e}")

    return predictions

# Function to plot the confusion matrix
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

# Function to plot the correlation matrix
def plot_correlation_matrix(df, selected_columns):
    corr_matrix = df[selected_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    st.pyplot(plt)

# Function to plot model performance
def plot_model_performance(accuracy, precision, recall, f1):
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    metrics_df = pd.DataFrame(metrics, index=[0])
    st.bar_chart(metrics_df)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Upload Dataset", "Train Model", "Make Prediction"])

if page == "Home":
    st.title("Welcome to the Biomedical ML Model Trainer and Predictor")
    st.markdown("""
    ## Overview
    This application allows you to upload a biomedical dataset, train machine learning models (KNN and Logistic Regression), and make predictions.
    
    Use the sidebar to navigate between the different functionalities:
    - **Upload Dataset:** Upload your dataset and visualize the correlation matrix.
    - **Train Model:** Train a machine learning model on your dataset.
    - **Make Prediction:** Make predictions using the trained model.
    
    ## Team Members
    - ASMAE ESSALAMI
    - DOUNIA BEN DAOUD
    - NOUHAYLA EL KARICH
    - IMANE EL BOURTIAA
    
    ### © 2024 IABM - Biomedical ML Model team.
    """)

elif page == "Upload Dataset":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_and_preprocess_dataset(uploaded_file)
        st.write("Dataset Loaded:")
        st.write(df.head())
        st.write("Summary Statistics:")
        st.write(df.describe())
        selected_columns = st.multiselect("Select features for correlation matrix", df.columns.tolist(), default=df.columns.tolist())
        if st.button("Show Correlation Matrix"):
            plot_correlation_matrix(df, selected_columns)

elif page == "Train Model":
    st.header("Train Model")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_and_preprocess_dataset(uploaded_file)
        st.write("Dataset Loaded:")
        st.write(df.head())
        st.write("Summary Statistics:")
        st.write(df.describe())

        target = st.selectbox("Select target column", df.columns)
        selected_columns = st.multiselect("Select features", df.columns.tolist(), default=df.columns.tolist())

        model_name = st.selectbox("Select model", ["KNN", "Logistic Regression"])
        if model_name == "KNN":
            n_neighbors = st.slider("Number of neighbors (k)", 1, 20, 5)
            params = {"n_neighbors": [n_neighbors]}
        elif model_name == "Logistic Regression":
            C = st.slider("Inverse of regularization strength (C)", 0.01, 10.0, 1.0)
            max_iter = st.slider("Maximum iterations", 100, 1000, 200)
            params = {"C": [C], "max_iter": max_iter}

        if st.button("Train Model"):
            accuracy, precision, recall, f1, conf_matrix, roc_auc, class_report, feature_names = train_model(df, target, model_name, params, selected_columns)
            cv_mean, cv_std = validate_model(df, target, model_name, params, selected_columns)
            
            st.write("Training completed.")
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision)
            st.write("Recall:", recall)
            st.write("F1 Score:", f1)
            st.write("Confusion Matrix:")
            plot_confusion_matrix(conf_matrix)
            st.write("ROC AUC Score:", roc_auc)
            st.write("Classification Report:")
            st.text(class_report)
            st.write("Cross-validation Mean Accuracy:", cv_mean)
            st.write("Cross-validation Std Dev:", cv_std)
            plot_model_performance(accuracy, precision, recall, f1)

elif page == "Make Prediction":
    st.header("Make Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file for prediction", type="csv")
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("Input Data:")
        st.write(input_data)

        if st.button("Predict"):
            prediction = make_prediction(input_data)
            st.write("Prediction for selected rows:", prediction)
