import streamlit as st
import pandas as pd
import numpy as np
import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns



# Load the pre-trained model
with open('top_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict_churn(customer_data):
    """
    Predicts whether the customer is likely to churn or not.

    Parameters:
    - customer_data: A dictionary containing customer information (keys: tenure, monthly_charges, contract, etc.).

    Returns:
    - prediction: A string indicating the predicted churn status ('Churn' or 'Not Churn').
    """
    # Preprocess the user inputs and create a DataFrame
    # Assuming customer_data is a dictionary with keys corresponding to feature names
    # Perform any necessary data preprocessing steps (e.g., encoding categorical variables)
    # Convert the processed data into a DataFrame
    
    # Make predictions using the pre-trained model
    # Note: You may need to adjust the input format based on how your model expects the data
    prediction = model.predict(customer_data)  # Replace 'customer_data' with your processed DataFrame
    
    # Convert the prediction result to human-readable format
    prediction = 'Churn' if prediction == 1 else 'Not Churn'
    
    return prediction

# Function to visualize churn patterns
# def visualize_churn(df):
#     """
#     Visualizes churn patterns based on different features.

#     Parameters:
#     - df: DataFrame containing churn prediction data.
#     """
#     # Visualize distribution of churn across contract type
#     st.subheader('Distribution of Churn by Contract Type')
#     contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
#     st.bar_chart(contract_churn)

#     # Visualize churn rate by tenure
#     st.subheader('Churn Rate by Tenure')
#     churn_rate_by_tenure = df.groupby('tenure')['Churn'].mean()
#     plt.plot(churn_rate_by_tenure.index, churn_rate_by_tenure.values)
#     plt.xlabel('Tenure')
#     plt.ylabel('Churn Rate')
#     st.pyplot()

# Main Streamlit application
def main():
    # Title and introduction
    st.title('Churn Prediction Dashboard')
    st.write('Welcome to the Churn Prediction Dashboard! Input customer information to predict churn and explore churn patterns.')

    # User inputs for prediction
    st.sidebar.header('Enter Customer Information')
    tenure = st.sidebar.slider('Tenure (months)', min_value=0, max_value=72, value=12)
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', min_value=0.0, max_value=150.0, value=50.0)
    contract_type = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])

    # Create a dictionary with user inputs
    customer_data = {'tenure': tenure, 'MonthlyCharges': monthly_charges, 'Contract': contract_type}

    # Make predictions and display result
    if st.sidebar.button('Predict Churn'):
        prediction = predict_churn(customer_data)
        st.write(f'Prediction: {prediction}')

    # Load and visualize churn prediction data
    df_churn = pd.read_csv('./Telco-Customer-Churn-cleaned.csv')  # Assuming you have the cleaned dataset
    st.header('Churn Analysis')
    visualize_churn(df_churn)

if __name__ == '__main__':
    main()
