import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    return df


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    num_imputer = SimpleImputer(strategy="median")
    cat_encoder = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_imputer, num_cols),
            ("cat", cat_encoder, cat_cols)
        ]
    )

    return preprocessor


def save_preprocessor(preprocessor, path: str):
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str):
    return joblib.load(path)
