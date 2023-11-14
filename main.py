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
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Read data
batters = pd.read_csv('/home/om/Desktop/REPOS/Cricket_Prediction_Model/Batters.csv')
bowlers = pd.read_csv('/home/om/Desktop/REPOS/Cricket_Prediction_Model/Bowlers.csv')

# Clean data
bowlers['Player'] = bowlers['Player'].str.replace(r'\(.*\)', '').str.strip()
batters['HS'] = batters['HS'].replace({'\*': ''}, regex=True).astype(int)
batters['Player'] = batters['Player'].str.replace(r'\(.*\)', '').str.strip()

# Select features for batsman
selected_features_batsman = ['Mat', 'Inns', 'NO', 'HS', 'Ave', 'BF', 'SR', '100', '50', '0', '4s', '6s']
df_selected_batsman = batters[selected_features_batsman].dropna()

# Prepare batsman data
X_batsman = df_selected_batsman[selected_features_batsman]
y_batsman = batters['Runs']

# Impute and scale batsman data
imputer_batsman = SimpleImputer(strategy='mean')
X_imputed_batsman = imputer_batsman.fit_transform(X_batsman)
scaler = StandardScaler()
X_imputed_batsman = scaler.fit_transform(X_imputed_batsman)

# Train batsman model
model_batsman = Lasso(alpha=0.1)
model_batsman.fit(X_imputed_batsman, y_batsman)

# Select features for bowler
selected_features_bowler = ['Wkts', 'Econ']
df_selected_bowler = bowlers[selected_features_bowler].dropna()

# Prepare bowler data
X_bowler = df_selected_bowler[selected_features_bowler]
y_bowler = bowlers['Wkts']

# Impute bowler data
imputer_bowler = SimpleImputer(strategy='mean')
X_imputed_bowler = imputer_bowler.fit_transform(X_bowler)

# Train bowler model
model_bowler = LinearRegression()
model_bowler.fit(X_imputed_bowler, y_bowler)

# Streamlit app styling
background_image = "CWC23-Fixtures-announcement-16x9-v2.jpeg"
background_css = f"""
    <style>
        body {{
            background-image: url({background_image});
            background-size: cover;
        }}
        .full-width {{
            width: 100%;
        }}
        .header {{
            font-size: 2em;
            text-align: center;
            color: #FFFFFF;
            margin-bottom: 20px;
        }}
        .subtitle {{
            font-size: 1.5em;
            color: #FFFFFF;
            margin-bottom: 10px;
        }}
        .input-box {{
            background-color: #4B6177;
            color: #FFFFFF;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .result-box {{
            background-color: #6A88A0;
            color: #FFFFFF;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }}
    </style>
"""

st.markdown(background_css, unsafe_allow_html=True)
st.title("Cricket Performance Prediction")

# Header
st.markdown('<div class="header">Predict Your Cricket Performance</div>', unsafe_allow_html=True)

# Player type selection
player_type = st.radio("Select Player Type:", ["Batsman", "Bowler"])

# Input box styling
st.markdown('<div class="input-box">', unsafe_allow_html=True)

if player_type == "Batsman":
    # Batsman input
    st.text_input("Enter the player's name:")
else:
    # Bowler input
    st.text_input("Enter the Bowler's name:")

# Close input box
st.markdown('</div>', unsafe_allow_html=True)

# Result box styling
st.markdown('<div class="result-box">', unsafe_allow_html=True)

# Display result
if player_type == "Batsman":
    st.success("Next Match Runs for the player: 100")
else:
    st.success("Predicted wickets for the bowler: 3")

# Close result box
st.markdown('</div>', unsafe_allow_html=True)
