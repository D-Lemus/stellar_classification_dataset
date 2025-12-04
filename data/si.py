import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("star_classification.csv")

redshift_col = df['redshift']
df = df.drop(columns=["redshift"])
df["redshift"] = redshift_col

X_reg = df
y_reg = df



X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.30, random_state=0, shuffle=True
)


X_train.to_csv("star_classification_70.csv", index=False)
X_test.to_csv("star_classification_conocido.csv", index=False)

desconocido = X_test.drop("redshift",axis=1)
desconocido.to_csv("star_classification_desconocido.csv", index=False)
