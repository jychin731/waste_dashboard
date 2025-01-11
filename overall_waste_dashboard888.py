
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the model path relative to the current working directory
model_path = "tuned_svr_model_Overall_Total_Waste_Generated_(tonne).pkl"

# Check if the file exists to avoid further errors
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the model
svr_model = joblib.load(model_path)

# Retained features for Overall Waste
retained_features = [
    '0 - 4 Years', '55 - 59 Years', '55 Year', 'Construction',
    'Ownership Of Dwellings', 'Accommodation & Food Services',
    '15 - 19 Years', '81 Year', '54 Year',
    'Literacy Rate (15 Years & Over) - Females',
    '10th Decile (Highest) (Dollar)', 'Overseas Citizens',
    'Median Monthly Household Income From Work Including Employer CPF Contributions (Dollar)',
    'Average Monthly Household Income From Work Per Household Member (Including Employer CPF Contributions) (Dollar)',
    'Literacy Rate (15 Years & Over) - Males', '56 Year',
    'Services Producing Industries', '80 Year', '4th Decile (Dollar)',
    'Add: Taxes On Products', '40 - 44 Years (Per Thousand Females)',
    'Finance & Insurance', '9 Year',
    'Household Reference Persons Aged 50-64 Years', 'Singapore Residents',
    '5th Decile (Dollar)',
    'Monthly Household Income From Work Per Household Member (Including Employer CPF Contributions) At 20th Percentile (Dollar)',
    '6th Decile (Dollar)', '15 - 19 Years (Per Thousand Females)',
    '7th Decile (Dollar)',
    'Real Estate, Professional Services And Administrative & Support Services',
    '85 - 89 Years', '9th Decile (Dollar)', 'Switzerland',
    'Median Monthly Household Income From Work Per Household Member (Including Employer CPF Contributions) (Dollar)',
    '70 Years & Over', '8th Decile (Dollar)', '2nd Decile (Dollar)',
    '3rd Decile (Dollar)', '80 Years & Over', '3-Person Households (Number)',
    '4-Person Households (Number)', '82 Year', 'Other Services Industries',
    '90 Years & Over', '1st Decile (Lowest) (Dollar)', 'Malays (Per Female)',
    '20 - 24 Years (Per Thousand Females)', '50 - 54 Years', '53 Year',
    'Information & Communications', '75 Years & Over', '20 - 24 Years',
    'United Arab Emirates', '60 Year',
    'Nitrogen Dioxide (Annual Mean) (Microgram Per Cubic Metre)', '8 Year',
    '60 - 64 Years', '85 Years & Over', '64 Year', '65 Years & Over',
    '80 - 84 Years', 'Utilities', 'Bangladesh',
    '35 - 39 Years (Per Thousand Females)'
]

# Mapping for renamed features
feature_mapping = {
    'Overseas Citizens': 'Overseas Citizens (Total)',
    'Singapore Residents': 'Singapore Residents (Total)',
    'Construction': 'GDP (Million Dollars): Construction',
    'Accommodation & Food Services': 'GDP (Million Dollars): Accommodation & Food Services',
    'Services Producing Industries': 'GDP (Million Dollars): Services Producing Industries',
    'Real Estate, Professional Services And Administrative & Support Services': 'GDP (Million Dollars): Real Estate, Professional Services And Administrative & Support Services',
    'Information & Communications': 'GDP (Million Dollars): Information & Communications',
    'Finance & Insurance': 'GDP (Million Dollars): Finance & Insurance',
    'Utilities': 'GDP (Million Dollars): Utilities',
    'Other Services Industries': 'GDP (Million Dollars): Other Services Industries',
    'Average Monthly Household Income From Work Per Household Member (Including Employer CPF Contributions) (Dollar)': 'Average Income (Including Employer CPF Contributions) (Dollar)',
    'Literacy Rate (15 Years & Over) - Females': 'Females Literacy Rate (15 Years & Over)',
    'Literacy Rate (15 Years & Over) - Males': 'Males Literacy Rate (15 Years & Over)',
    'Switzerland': 'Tourism: Switzerland (Number)',
    'United Arab Emirates': 'Tourism: United Arab Emirates (Number)',
    'Bangladesh': 'Tourism: Bangladesh (Number)',
    'Ownership Of Dwellings': 'Ownership of Dwellings (Number)',
    'Nitrogen Dioxide (Annual Mean) (Microgram Per Cubic Metre)': 'Annual Mean Nitrogen Dioxide (ppb)',
}

# Sidebar for user input (custom features from mapping)
st.sidebar.header("User Input Parameters")
user_inputs = {}

for feature in retained_features:
    if feature in feature_mapping:  # If the feature is mapped, create a slider
        user_inputs[feature] = st.sidebar.slider(
            feature_mapping[feature],
            min_value=0,
            max_value=100000,
            value=5000,
            step=1000
        )
    else:  # Set unmapped features to default value of 0
        user_inputs[feature] = 0

# Convert user inputs to a feature set
def generate_full_feature_set_from_sidebar(user_inputs):
    full_features = np.zeros((1, len(retained_features)))
    for i, feature in enumerate(retained_features):
        full_features[0, i] = user_inputs[feature]  # Use slider value directly or default
    return full_features

# Prepare input features using user inputs
input_features = generate_full_feature_set_from_sidebar(user_inputs)

# Prediction function
def predict_overall_waste(features):
    return svr_model.predict(features)[0]

# Generate prediction
overall_waste_prediction = predict_overall_waste(input_features)

# Main content
st.title("Overall Waste Prediction Dashboard")
st.write("### Predicted Overall Waste (tonnes):")
st.metric(label="Prediction", value=f"{overall_waste_prediction:.2f} tonnes")

# Visualization
st.write("### Prediction Visualization")
fig, ax = plt.subplots()
ax.bar(["Overall Waste Prediction"], [overall_waste_prediction], color='skyblue', width=0.5)
ax.set_title("Overall Waste Prediction (Visualization)", fontsize=16)
ax.set_xlabel("Category", fontsize=12)
ax.set_ylabel("Tonnes", fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
# Display the exact value above the bar
ax.text(0, overall_waste_prediction + 100, f"{overall_waste_prediction:.2f} tonnes", ha='center', fontsize=12, color='black')
st.pyplot(fig)
