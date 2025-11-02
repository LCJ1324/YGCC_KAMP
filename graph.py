import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from utils.minmax_store import load_minmax

st.set_page_config(layout="wide")

st.title("날짜별 공정현황")


def load_data():
    df = pd.read_csv("2. 소성가공 품질보증 AI 데이터셋.csv")
    return df


df = load_data()
df = df.dropna()
df["passorfail"] = df["passorfail"].astype(int)
df["date"] = pd.to_datetime(df["date"])

columns = df.drop(columns=["date", "passorfail"]).columns

out_df = load_minmax()

cond = False
for i in columns:
    cond |= (df[i] < out_df.loc[i, "min"]) | (df[i] > out_df.loc[i, "max"])

df["out_of_range"] = cond.astype(int)
df["passorfail"] = (df["passorfail"] | df["out_of_range"]).astype(int)

font_path = "NanumGothic.ttf"
fm.fontManager.addfont(font_path)
plt.rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

with st.sidebar:
    selected_date = st.date_input("날짜 선택", value=df["date"].dt.date.min())
    feature = st.selectbox("특성 선택", columns)

col1, col2 = st.columns(2)

with col1:
    col1_1, col1_2 = st.columns([2.4, 1])

    col1_1.header(f"{feature} 시계열 그래프")

    container = st.container(border=True)
    with container:
        _, container, _ = st.columns([1, 2.5, 1])
        container.markdown(
            f"**{feature} 한계 범위 값 : {out_df.loc[feature, 'min']} ~ {out_df.loc[feature, 'max']}**"
        )

with col2:
    col2_1, col2_2 = st.columns([2.4, 1])

    col2_1.header(f"양품/불량품 시계열 그래프")

    container = st.container(border=True)
    with container:
        _, container, _ = st.columns([1, 1.5, 1])
        container.markdown(
            f"**불량수/양품수/전체 :  {len(df[df['passorfail'] == 1])}/{len(df[df['passorfail'] == 0])}/{len(df)}**"
        )

col3, col4 = st.columns(2, border=True)

with col3:
    fig1, ax = plt.subplots(figsize=(8, 4))
    plt.plot(df["date"], df[feature], color="slateblue", alpha=0.7)
    plt.axhline(out_df.loc[feature, "max"], color="r")
    plt.axhline(out_df.loc[feature, "min"], color="r")
    plt.grid(axis="y", color="grey", alpha=0.5, linestyle="--")
    plt.xlabel("시간")
    plt.ylabel("")
    plt.xticks(df["date"][::3400], rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    # for s in ["top", "right"]:
    #     ax.spines[s].set_visible(False)
    st.pyplot(fig1)

with col4:
    fig1, ax = plt.subplots(figsize=(8, 4))

    plt.plot(df["date"], df["passorfail"], color="slateblue", alpha=0.3)

    mask1 = df["passorfail"] == 0
    plt.scatter(df.loc[mask1, "date"], df.loc[mask1, "passorfail"], color="blue", s=15)

    mask2 = df["passorfail"] == 1
    plt.scatter(df.loc[mask2, "date"], df.loc[mask2, "passorfail"], color="red", s=15)

    plt.grid(axis="y", color="grey", alpha=0.5, linestyle="--")
    plt.xlabel("시간")
    plt.ylabel("")
    plt.xticks(df["date"][::3400], rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    st.pyplot(fig1)

col5, col6 = st.columns(2)

with col5:
    with st.container(border=True):
        st.markdown(f"**{feature}**")
        st.write(
            "**평균값** : "
            + f"{df[feature].mean() : .2f}"
            + "ㅤㅤㅤ"
            + "**최댓값** : "
            + f"{df[feature].max() : .2f}"
            + "ㅤㅤㅤ"
            + "**최솟값** : "
            + f"{df[feature].min() : .2f}"
        )


with col6:
    with st.expander("**불량 정보**"):
        cols = out_df.index.tolist()

        msgs = []
        for col in cols:
            cond_min = df[col] < out_df.loc[col, "min"]
            cond_max = df[col] > out_df.loc[col, "max"]

            df[f"{col}_out"] = False
            df.loc[cond_min, f"{col}_out"] = f"{col} 최소값 미만"
            df.loc[cond_max, f"{col}_out"] = f"{col} 최대값 초과"

            msgs.append(f"{col}_out")

        def build_info(row):
            v_msgs = [m for m in row[msgs] if m]
            if v_msgs:
                return v_msgs
            if row["passorfail"] == 1:
                return ["모델링 예측"]
            return []

        df["정보"] = df.apply(build_info, axis=1)

        df["불량"] = df["정보"].apply(lambda x: 1 if len(x) > 0 else 0)

        result = df[df["불량"] == 1].copy()
        result["정보"] = result["정보"].apply(lambda x: ", ".join(x))
        result = result[["date", "불량", "정보"]]
        st.dataframe(result, use_container_width=True)
