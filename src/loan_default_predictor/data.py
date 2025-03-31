import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "loan_status"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    X_train = prepare_df(X_train)
    X_test = prepare_df(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train.values, X_test_scaled, y_test.values


def prepare_df(df):
    df = df.copy()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    grade_mapping = {letter: idx + 1 for idx, letter in enumerate("ABCDEFG")}
    df["loan_grade"] = df["loan_grade"].map(grade_mapping)

    home_ownership_mapping = {
        "OTHER": 0,
        "RENT": 1,
        "MORTGAGE": 2,
        "OWN": 3
    }
    df["person_home_ownership"] = df["person_home_ownership"].map(home_ownership_mapping)

    intent_mapping = {
        "DEBTCONSOLIDATION": 0,
        "EDUCATION": 1,
        "MEDICAL": 2,
        "VENTURE": 3,
        "PERSONAL": 4,
        "HOMEIMPROVEMENT": 5
    }
    df["loan_intent"] = df["loan_intent"].map(intent_mapping)

    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"N": 0, "Y": 1})

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = df[col].astype(int)
            except ValueError:
                pass

    return df