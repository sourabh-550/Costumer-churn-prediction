from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=3000,   # increased
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model



def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model