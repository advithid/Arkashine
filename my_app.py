import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
import shap
import pickle

# Streamlit app title with color
st.markdown("<h1 style='text-align: center; color: blue;'>Soil Data Processing and Model Training</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    soil_data = pd.read_csv(uploaded_file)

    # Convert all columns to numeric
    for column in soil_data.columns:
        soil_data[column] = pd.to_numeric(soil_data[column], errors='coerce')

    # Fill missing values with mean
    soil_data = soil_data.fillna(soil_data.mean())

    # Extract y and x
    if 'index' in soil_data.columns:
        y = soil_data['index'].values
        x_columns = soil_data.columns.drop('index')
        x = soil_data[x_columns].values

        # Apply Savitzky-Golay filter
        window_length = 11
        polyorder = 2
        y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
        x_smooth = np.apply_along_axis(lambda m: savgol_filter(m, window_length=window_length, polyorder=polyorder), axis=0, arr=x)

        # Save smoothed data to a new DataFrame
        smoothed_data = pd.DataFrame(x_smooth, columns=[f'{col}_smooth' for col in x_columns])
        smoothed_data['index'] = y
        smoothed_data.to_csv('smoothed_data.csv', index=False)

        # Display the smoothed data columns with color
        st.markdown("<h2 style='color: green;'>Smoothed Data Columns:</h2>", unsafe_allow_html=True)
        st.write(smoothed_data.columns)

        # Prepare for train-test split
        smoothed_data = smoothed_data.drop(columns=['index'])
        target_columns = ['P_smooth', 'K_smooth', 'PH_smooth', 'EC_smooth', 'OC_smooth',
                          'Ca_smooth', 'Mg_smooth', 'S_smooth', 'Fe_smooth', 'Mn_smooth',
                          'Cu_smooth', 'Zn_smooth', 'B_smooth']

        train_test_splits = {}
        for target in target_columns:
            X = smoothed_data.drop(columns=[target])
            y = smoothed_data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            train_test_splits[target] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            st.markdown(f"<h3 style='color: darkorange;'>Shapes for target variable '{target}':</h3>", unsafe_allow_html=True)
            st.write("Shape of X_train:", X_train.shape)
            st.write("Shape of X_test:", X_test.shape)
            st.write("Shape of y_train:", y_train.shape)
            st.write("Shape of y_test:", y_test.shape)

        # Define and train models
        models = {
            'Lasso': Lasso(alpha=1.0, max_iter=1000),
            'Ridge': Ridge(alpha=1.0),
            'SVR': SVR(kernel='rbf'),
            'BPNN': MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=42),
            'PLSRegression': PLSRegression(n_components=2)
        }

        # Train and evaluate models
        for model_name, model in models.items():
            st.markdown(f"<h3 style='color: darkred;'>Training {model_name} model...</h3>", unsafe_allow_html=True)
            for target, splits in train_test_splits.items():
                X_train = splits['X_train']
                X_test = splits['X_test']
                y_train = splits['y_train']
                y_test = splits['y_test']

                # Data scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train the model
                model.fit(X_train_scaled, y_train)

                # Save the model to a pickle file
                with open(f'{model_name}_{target}.pkl', 'wb') as f:
                    pickle.dump(model, f)

                # Model evaluation
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)

                # Display evaluation metrics with color
                st.markdown(f"<h4 style='color: purple;'>{model_name} for {target} - Train R<sup>2</sup> Score: {train_score:.2f}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='color: purple;'>{model_name} for {target} - Test R<sup>2</sup> Score: {test_score:.2f}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='color: purple;'>{model_name} for {target} - Train MSE: {train_mse:.2f}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='color: purple;'>{model_name} for {target} - Test MSE: {test_mse:.2f}</h4>", unsafe_allow_html=True)

                # SHAP summary plot
                feature_names_list = X_train.columns.tolist()
                explainer = shap.Explainer(model.predict, X_train_scaled)
                shap_values = explainer.shap_values(X_train_scaled)
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_train_scaled, plot_type='bar', feature_names=feature_names_list, show=False)
                st.pyplot(fig)

    else:
        st.markdown("<h3 style='color: red;'>The dataset does not contain the 'index' column.</h3>", unsafe_allow_html=True)

else:
    st.markdown("<h3 style='color: red;'>Please upload a CSV file.</h3>", unsafe_allow_html=True)
