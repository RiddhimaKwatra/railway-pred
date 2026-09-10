import json
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("dataset.csv")

y = df.Delay_Minutes
X = df.drop(["Delay_Minutes", "Actual_Arrival_Mins", "Train_ID"], axis=1)

cato_cols = list(X.select_dtypes(include=["object", "string"]).columns)
num_cols_df = X.drop(cato_cols, axis=1)
num_cols = list(num_cols_df.columns)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
OH_cols_X = pd.DataFrame(
    encoder.fit_transform(X[cato_cols]),
    columns=encoder.get_feature_names_out(cato_cols),
)
OH_cols_X.index = X.index

OH_X = pd.concat([num_cols_df, OH_cols_X], axis=1)
OH_X.columns = OH_X.columns.astype(str)
feature_columns = list(OH_X.columns)

demo_rows = df.sample(5, random_state=42)
demo_indices = demo_rows.index
OH_X_train = OH_X.drop(demo_indices)
y_train = y.drop(demo_indices)
demo_rows.to_csv("demo_examples.csv", index=False)

model = XGBRegressor(n_estimators=700, learning_rate=0.03)
scores = -1 * cross_val_score(
    model, OH_X_train, y_train, cv=6, scoring="neg_mean_absolute_error"
)
print("MAE per fold:", scores)
print("Mean MAE:", scores.mean())

model.fit(OH_X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")

with open("feature_columns.json", "w") as f:
    json.dump(
        {
            "cato_cols": cato_cols,
            "num_cols": num_cols,
            "feature_columns": feature_columns,
            "categories": {
                col: sorted(df[col].unique().tolist()) for col in cato_cols
            },
        },
        f,
        indent=2,
    )

print("Saved model.pkl, encoder.pkl, feature_columns.json")
