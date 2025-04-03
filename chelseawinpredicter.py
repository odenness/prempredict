import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

# --- Load dataset ---
df = pd.read_csv("epl1011matchday-results-pts.csv", index_col=0)

# --- Build long format for Chelsea ---
chelsea_row = df.loc['Chelsea']
teams = df.index.tolist()
match_data = []

for md in range(1, 39):
    result_col = f"M{md}Results"
    points_col = f"M{md}Points"
    result = chelsea_row[result_col]
    points = chelsea_row[points_col]

    # Parse result: HomeW, AwayL, etc.
    match = re.match(r"(Home|Away)([WDL])", result)
    if not match:
        continue
    location, outcome = match.groups()
    is_home = 1 if location == "Home" else 0
    win = 1 if outcome == "W" else 0

    # Figure out opponent: the team that gave Chelsea those points on matchday
    for team in teams:
        if team == 'Chelsea':
            continue
        if df.loc[team, f"M{md}Points"] + points == 3 or \
           (df.loc[team, f"M{md}Points"] == 1 and points == 1):
            opponent = team
            break
    else:
        opponent = "Unknown"

    match_data.append({
        "Matchday": md,
        "IsHome": is_home,
        "Opponent": opponent,
        "ChelseaWin": win
    })

chelsea_df = pd.DataFrame(match_data)
chelsea_df = pd.get_dummies(chelsea_df, columns=["Opponent"], drop_first=True)

# --- Train/test split ---
X = chelsea_df.drop("ChelseaWin", axis=1)
y = chelsea_df["ChelseaWin"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# --- Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# --- Predict a new match ---
def predict_chels_win(opponent, is_home):
    row = {col: 0 for col in X.columns}
    row["IsHome"] = 1 if is_home else 0
    if f"Opponent_{opponent}" in row:
        row[f"Opponent_{opponent}"] = 1
    else:
        print(f"⚠️ Opponent '{opponent}' not in training set. Prediction may be inaccurate.")
    df_input = pd.DataFrame([row])
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    return pred, prob

# Example prediction
opponent = input("Enter opponent (e.g. Arsenal): ").strip()
home_input = input("Is Chelsea playing at home? (yes/no): ").strip().lower()
is_home = 1 if home_input == 'yes' else 0

result, conf = predict_chels_win(opponent, is_home)
print(f"\nPrediction: Chelsea will {'WIN ✅' if result else 'NOT WIN ❌'}")
print(f"Confidence: {conf:.2%}")
