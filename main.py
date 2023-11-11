import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

batters = pd.read_csv('/home/om/Desktop/REPOS/Cricket_Prediction_Model/Batters.csv')
bowlers = pd.read_csv('/home/om/Desktop/REPOS/Cricket_Prediction_Model/Bowlers.csv')

bowlers['Player'] = bowlers['Player'].str.replace(r'\(.*\)', '').str.strip()


batters['HS'] = batters['HS'].replace({'\*': ''}, regex=True).astype(int)

batters['Player'] = batters['Player'].str.replace(r'\(.*\)', '').str.strip()

selected_features_batsman = ['Mat', 'Inns', 'NO', 'HS', 'Ave', 'BF', 'SR', '100', '50', '0', '4s', '6s']
df_selected_batsman = batters[selected_features_batsman]

df_selected_batsman = df_selected_batsman.dropna()

X_batsman = df_selected_batsman[['Mat', 'Inns', 'NO', 'HS', 'Ave', 'BF', 'SR', '100', '50', '0', '4s', '6s']]
y_batsman = batters['Runs']

imputer_batsman = SimpleImputer(strategy='mean')
X_imputed_batsman = imputer_batsman.fit_transform(X_batsman)

scaler = StandardScaler()
X_imputed_batsman = scaler.fit_transform(X_imputed_batsman)

model_batsman = Lasso(alpha=0.1)  # You can adjust the alpha parameter
model_batsman.fit(X_imputed_batsman, y_batsman)

selected_features_bowler = ['Wkts', 'Econ']
df_selected_bowler = bowlers[selected_features_bowler]

df_selected_bowler = df_selected_bowler.dropna()

X_bowler = df_selected_bowler[['Wkts', 'Econ']]
y_bowler = bowlers['Wkts']

imputer_bowler = SimpleImputer(strategy='mean')
X_imputed_bowler = imputer_bowler.fit_transform(X_bowler)

model_bowler = LinearRegression()
model_bowler.fit(X_imputed_bowler, y_bowler)

background_image = "CWC23-Fixtures-announcement-16x9-v2.jpeg"
background_css = f"""
    <style>
        body {{
            background-image: url({background_image});
            background-size: cover;
        }}
    </style>
"""

st.markdown(background_css, unsafe_allow_html=True)
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

# Player type selection
player_type = st.radio("Select Player Type:", ["Batsman", "Bowler"])

if player_type == "Batsman":
    player_name = st.text_input("Enter the player's name:")

    player_data = batters[batters['Player'] == player_name]
    if player_data.empty:
        st.write(f"Player '{player_name}' not found in the dataset.")
    else:
        player_features = player_data[['Mat', 'Inns', 'NO', 'HS', 'Ave', 'BF', 'SR', '100', '50', '0', '4s', '6s']]
        player_features['HS'] = player_features['HS'].replace({'\*': ''}, regex=True).astype(int)

        player_features_scaled = scaler.transform(player_features)

        predicted_runs = model_batsman.predict(player_features_scaled)

        previous_total_runs = player_data['Runs'].values[0]

        if previous_total_runs > predicted_runs[0]:
            new_total_runs = round(previous_total_runs - predicted_runs[0])
        else:
            new_total_runs = round(predicted_runs[0] - previous_total_runs)

        st.write(f'Next Match Runs for {player_name} after the Next Match: {new_total_runs}')

else:
    player_name_bowler = st.text_input("Enter the Bowler's name:")

    player_data_bowler = bowlers[bowlers['Player'].str.strip() == player_name_bowler.strip()]
    if player_data_bowler.empty:
        st.write(f"Player '{player_name_bowler}' not found in the bowler dataset.")
    else:
        player_features_bowler = player_data_bowler[['Wkts', 'Econ']]

        player_features_imputed_bowler = imputer_bowler.transform(player_features_bowler)

        predicted_wickets_bowler = model_bowler.predict(player_features_imputed_bowler)

        ave_column_bowler = player_data_bowler['Ave'].values[0]
        predicted_wickets_bowler /= ave_column_bowler
        rounded_predicted_wickets_bowler = round(predicted_wickets_bowler[0])

        st.write(f'Predicted wickets for {player_name_bowler} in the next match (normalized by Ave): {rounded_predicted_wickets_bowler}')
