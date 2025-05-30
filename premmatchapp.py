import pandas as pd
import re
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data files
FILES = [
    "epl1011matchday-results-pts.csv",
    "epl1112matchday-results-pts.csv",
    "epl1213matchday-results-pts.csv",
    "epl1314matchday-results-pts.csv",
    "epl1415matchday-results-pts.csv",
    "epl1516matchday-results-pts.csv",
    "epl1617matchday-results-pts.csv",
    "epl1718matchday-results-pts.csv",
    "epl1819matchday-results-pts.csv",
    "epl1920matchday-results-pts.csv"
]

# Weight by recency of season
SEASON_WEIGHTS = {
    "epl1011": 0.1,
    "epl1112": 0.2,
    "epl1213": 0.3,
    "epl1314": 0.4,
    "epl1415": 0.5,
    "epl1516": 0.6,
    "epl1617": 0.7,
    "epl1718": 0.8,
    "epl1819": 0.9,
    "epl1920": 1.0,
}

@st.cache_data
def load_and_prepare_data():
    match_data = []

    for file in FILES:
        try:
            df = pd.read_csv(file, index_col=0)
            teams = df.index.tolist()
            season = file[:6]

            for team in teams:
                for md in range(1, 39):
                    result_col = f"M{md}Results"
                    points_col = f"M{md}Points"

                    if result_col not in df.columns or points_col not in df.columns:
                        continue

                    result = df.loc[team, result_col]
                    points = df.loc[team, points_col]

                    match = re.match(r"(Home|Away)([WDL])", str(result))
                    if not match:
                        continue

                    loc, outcome = match.groups()
                    is_home = 1 if loc == "Home" else 0
                    win = 1 if outcome == "W" else 0

                    opponent = "Unknown"
                    for opp in teams:
                        if opp == team:
                            continue
                        opp_pts = df.loc[opp, f"M{md}Points"]
                        if (opp_pts + points == 3 and win) or (opp_pts == 1 and points == 1):
                            opponent = opp
                            break

                    season_weight = SEASON_WEIGHTS.get(season, 1.0)
                    matchday_weight = md / 38  # 1 to 38 scaled between ~0.026 and 1
                    combined_weight = season_weight * matchday_weight

                    match_data.append({
                        "Season": season,
                        "Team": team,
                        "Opponent": opponent,
                        "IsHome": is_home,
                        "Win": win,
                        "Matchday": md,
                        "Weight": combined_weight
                    })

        except Exception as e:
            st.warning(f"Failed to process {file}: {e}")

    df_all = pd.DataFrame(match_data)
    df_all = pd.get_dummies(df_all, columns=["Team", "Opponent"], drop_first=True)
    return df_all

# Load data
st.title("⚽ Premier League Match Win Predictor (Weighted by Recency)")
df = load_and_prepare_data()

# Prepare model
X = df.drop(columns=["Win", "Season", "Weight"])
y = df["Win"]
weights = df["Weight"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
sample_weights_train = weights.iloc[X_train.index]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weights_train)

# UI elements
all_teams = sorted(list(set(t.replace("Team_", "") for t in df.columns if t.startswith("Team_"))) + ["Chelsea"])
all_opponents = sorted(list(set(t.replace("Opponent_", "") for t in df.columns if t.startswith("Opponent_"))) + ["Chelsea"])

team = st.selectbox("Select your team", all_teams)
opponent = st.selectbox("Select opponent", all_opponents)
is_home = st.radio("Is your team playing at home?", ["Yes", "No"]) == "Yes"

if st.button("Predict Result"):
    input_row = {col: 0 for col in X.columns}
    input_row["IsHome"] = 1 if is_home else 0
    input_row["Matchday"] = 1  # Placeholder

    team_col = f"Team_{team}"
    opp_col = f"Opponent_{opponent}"

    if team_col in input_row:
        input_row[team_col] = 1
    else:
        st.warning(f"⚠️ Team '{team}' not in training data. Prediction might be off.")

    if opp_col in input_row:
        input_row[opp_col] = 1
    else:
        st.warning(f"⚠️ Opponent '{opponent}' not in training data.")

    input_df = pd.DataFrame([input_row])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")
    st.success("✅ Your team is likely to WIN!") if prediction == 1 else st.error("❌ Your team is unlikely to win.")
    st.write(f"Model confidence: **{prob:.2%}**")
