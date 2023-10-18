import streamlit as st

import numpy as np
import pandas as pd

st.set_page_config(
    layout="wide",
)
st.title("🌳 Random forest 🌳")
st.subheader(":blue[Overall RMSLE]")

#결과값
def f1(s1, s2):
    if (s1-s2) < 0:
        delta_color = "inverse"
    else:
        delta_color = "normal"
    value = np.round(abs(s1-s2), 3)  
    return value, delta_color

cv_score = 0.381
valid = 0.420
test = 0.442

v1, d1 = f1(cv_score, valid)
v2, d2 = f1(cv_score, test)


col1, col2, col3 = st.columns(3)
col1.metric("CV_score", cv_score, "")
col2.metric("Test", valid, v1, delta_color=d1)
col3.metric("Kaggle (out of sample)", test, v2, delta_color=d2)

#########################################################
# st.subheader("multicollinearity를 신경 쓸 필요가 없다.")

# ###########
# st.markdown("""Does multicollinearity affect random forest?
# Random Forest uses bootstrap sampling and feature sampling, 
# i.e row sampling and column sampling. Therefore Random Forest is not affected 
# by multicollinearity that much since it is picking different set of features 
# for different models and of course every model sees a different set of data points.""")

#### 데이터셋
st.subheader(":blue[📁 Dataset used in model]")

@st.cache_data
def load_data():
    train = pd.read_csv("bike_prediction/kaggle_data/train_eda.csv")
    return train

train= load_data()
cat_col = ["season", "Year","weather", "Day of week","Month","Day_info"] #Hour
for col in cat_col:
    train[col] = train[col].astype("category")
#target, drop, y
target_col = "count"
drop_cols = ["Unnamed: 0", "datetime", "workingday", "holiday", "Day", "Year", "Hour",target_col] #"sin_hour", "cos_hour"
train = train.drop(drop_cols, axis=1)

cat_col = ["season", "weather", "Day of week", "Month", "Day_info"] #"Hour"
train = pd.get_dummies(train, columns=cat_col)
st.write(train.head(3))

####################################
st.subheader(":blue[👍 Best Hyperparameters]")

hp_name = ["n_estimators", "max_features"]
types = ["int", "string"]
ranges = ["""[500, 2000]""",
         """["auto", "sqrt", "log2", None]"""]
best_values = [568, None]
params = pd.DataFrame({'hp_name':hp_name,
                        "type": types,
                        'range':ranges,
                        'best_value':best_values})
st.write(params)

######################################
if st.button("⌨️ See code for RF Regressor with Optuna"):
    st.code("""
    def objective(trial: Trial) -> float:
        params_rf = {
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 100, 700),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None])
        }

        n_splits=5
        random_state=42
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []

        #local_train, local_valid를 5번 만들어서 수행
        for train_index, valid_index in kf.split(X=x_train, y=y_train):
            X_train, Y_train = x_train.iloc[train_index], np.log1p(y_train[train_index])
            X_valid, Y_valid = x_train.iloc[valid_index], np.log1p(y_train[valid_index])

            model = RandomForestRegressor(**params_rf)
            model.fit(X_train, Y_train )

            rf_pred = model.predict(X_valid)
            scores.append(rmsle(Y_valid, rf_pred))

        return np.mean(scores)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name="Randomforest",
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=50)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)
    """)

#############################
op = "bike_prediction/RF/op.png"
pred = "bike_prediction/RF/pred.png"

st.subheader(":blue[Optuna Result]")
st.image(op)

st.subheader(":blue[Test data prediction result]")      
st.image(pred)
##############################
st.subheader(":blue[Shap Values]")
shap = "bike_prediction/RF/Shap.png"
waterfall = "bike_prediction/RF/sharp_waterfall.png"

st.markdown("##### Overall Shap Values")
st.warning('- Random Forest 모델의 특성을 고려하여 Hour 변수를 sin_hour와 cos_hour로 나누어 학습을 진행하였다.\n')
st.image(shap)

st.markdown("##### Model explanation for first row")          
st.image(waterfall)
