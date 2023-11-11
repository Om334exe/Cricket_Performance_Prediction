import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load your dataset
# Assuming your dataset is in a CSV file named 'batsman_data.csv'
batters = pd.read_csv('/home/om/Desktop/REPOS/Cricket_Prediction_Model/Batters.csv')
batters['Player'] = batters['Player'].str.split('(').str[0].str.strip()

# Convert 'HS' column to numeric, considering '*' as not out
batters['HS'] = batters['HS'].replace({'\*': ''}, regex=True).astype(int)

# Extract features (X) and target variable (y)
X = batters[['Mat', 'Inns', 'NO', 'HS', 'Ave', 'BF', 'SR', '100', '50', '0', '4s', '6s']]
y = batters['Runs']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Lasso Regression model
model = Lasso()

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model.fit(X_train_scaled, y_train)

# Streamlit app
st.title("Cricket Performance Prediction")

st.markdown(
    """
    <style>
        body {
            background-color: #2A3A4A;
            color: #FFFFFF;
        }
        .stTextInput {
            background-color: #4B6177;
            color: #FFFFFF;
        }
        .stButton {
            background-color: #6A88A0;
            color: #FFFFFF;
        }
        .stMarkdown {
            background-color: #2A3A4A;
            color: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Player name input
player_name = st.text_input("Enter the Batsman name:")

# Check if the player exists in the dataset
player_data = batters[batters['Player'] == player_name]
if player_data.empty:
    st.write(f"Player '{player_name}' not found in the dataset.")
else:
    # Player features
    player_features = player_data[['Mat', 'Inns', 'NO', 'HS', 'Ave', 'BF', 'SR', '100', '50', '0', '4s', '6s']]
    player_features['HS'] = player_features['HS'].replace({'\*': ''}, regex=True).astype(int)

    # Standardize the player's features using the same scaler
    player_features_scaled = scaler.transform(player_features)

    # Make predictions for the player using Lasso Regression
    predicted_runs = model.predict(player_features_scaled)

    previous_total_runs = player_data['Runs'].values[0]

    # Calculate the new total runs, ensuring it's non-negative
    if previous_total_runs > predicted_runs[0]:
        new_total_runs = round(previous_total_runs - predicted_runs[0])
    else:
        new_total_runs = round( predicted_runs[0] - previous_total_runs)



    st.write(f'Next Match Runs for {player_name} after the Next Match: {new_total_runs}')
