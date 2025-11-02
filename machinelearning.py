import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from utils.minmax_store import load_minmax

st.set_page_config(layout="wide")

st.title("공정 불량 예측")


def load_data():
    df = pd.read_csv("2. 소성가공 품질보증 AI 데이터셋.csv")
    return df


df = load_data()
df = df.dropna()
df["passorfail"] = df["passorfail"].astype(int)

columns = df.drop(columns=["date", "passorfail"]).columns

out_df = load_minmax()

cond = False
for i in columns:
    cond |= (df[i] < out_df.loc[i, "min"]) | (df[i] > out_df.loc[i, "max"])

df["out_of_range"] = cond.astype(int)
df["passorfail"] = (df["passorfail"] | df["out_of_range"]).astype(int)
df["torque_pressure_ratio"] = df["EX1.MD_TQ"] / (df["EX1.MD_PV"] + 1e-6)
df["temp2_roll_std"] = df["EX2.MELT_TEMP"].rolling(10).std()
df["temp3_roll_std"] = df["EX3.MELT_TEMP"].rolling(10).std()
df["temp4_roll_std"] = df["EX4.MELT_TEMP"].rolling(10).std()
df["temp5_roll_std"] = df["EX5.MELT_TEMP"].rolling(10).std()
df = df.dropna()

font_path = "NanumGothic.ttf"
fm.fontManager.addfont(font_path)
plt.rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 데이터 준비
# =========================
features = [
    "EX1.MELT_P_PV",
    "EX1.H2O_PV",
    "torque_pressure_ratio",
    "temp2_roll_std",
    "temp3_roll_std",
    "temp4_roll_std",
    "temp5_roll_std",
    "out_of_range",
]
target = "passorfail"

X = df[features].values
y = df[target].astype(int).values

results = {}

# =========================
# 학습/평가 분리 (8:2, stratify)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42
)
print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# =========================
# 스케일링 (train fit → train/test transform)
# =========================
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# SMOTE (train만)
# =========================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(
    random_state=42,
    n_estimators=400,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    max_depth=15,
    bootstrap=False,
)
rf.fit(X_train, y_train)


# =========================
# 테스트셋 평가 함수
# =========================
def eval_model(m, Xte, yte, name):
    pred = m.predict(Xte)
    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, zero_division=0)
    rec = recall_score(yte, pred, zero_division=0)
    f1 = f1_score(yte, pred, zero_division=0)
    cm = confusion_matrix(yte, pred)
    print(f"[{name}] acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
    fig1, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["Normal", "Fail"],
        yticklabels=["Normal", "Fail"],
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig1)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

col1, col2, col3, col4 = st.columns(4, border=True)


col5, col6 = st.columns(2, border=True)

# =========================
# 기본 모델 평가
# =========================
with col5 :
    st.subheader('Confusion Matrix')
    results["RandomForest"] = eval_model(rf, X_test, y_test, "RandomForest")

# =========================
# 피처 중요도 시각화
# =========================
with col6:
    st.subheader('Feature Importance')
    fig2, ax = plt.subplots(figsize=(8, 8))
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    sns.barplot(x=importances[order], y=np.array(features)[order])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    st.pyplot(fig2)

# =========================
# 결과 테이블
# =========================
col1.metric('**Accuracy**', f"{results['RandomForest']['accuracy'] : .4f}")
col2.metric('**Precision**', f"{results['RandomForest']['precision'] : .4f}")
col3.metric('**Recall**', f"{results['RandomForest']['recall'] : .4f}")
col4.metric('**F1 score**', f"{results['RandomForest']['f1'] : .4f}")
