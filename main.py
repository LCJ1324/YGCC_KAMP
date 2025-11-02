import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.dates as mdates

st.set_page_config(layout="wide")

st.title("소성 공정 품질 이상 탐지 & 불량 분석 시스템")


def load_data():
    df = pd.read_csv("2. 소성가공 품질보증 AI 데이터셋.csv")
    return df


df = load_data()
df = df.dropna()
df["passorfail"] = df["passorfail"].astype(int)
df["date"] = pd.to_datetime(df["date"])

columns = df.drop(columns=["date", "passorfail"]).columns

with st.sidebar:
    feature = st.selectbox("특성 선택", columns)

font_path = "NanumGothic.ttf"
fm.fontManager.addfont(font_path)
plt.rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

pass_rate = len(df[df["passorfail"] == 0]) / len(df) * 100
fail_rate = len(df[df["passorfail"] == 1]) / len(df) * 100

col1, col2, col3, col4, col5 = st.columns(5, border=True)
col1.metric("## **생산량**", f"{len(df)}개")
col2.metric("## **양품률**", f"{pass_rate : .2f}%")
col3.metric("## **불량률**", f"{fail_rate : .2f}%")
col4.metric(feature + ' ' + "**평균값**", f"{df[feature].mean() : .2f}")
col5.metric(feature + ' ' + "**표준편차**", f"{df[feature].std() : .2f}")

with st.container(border=True):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    target_corr = corr["passorfail"].abs().sort_values(ascending=False)[1:16]
    target_corr.name = '상관계수'

    col6, col7, col8 = st.columns([1, 3, 3])

    col6.write(target_corr)

    with col7:
        fig1, ax = plt.subplots(figsize=(8, 4))
        plt.plot(df["date"], df["EX1.MD_PV"], color="slateblue", alpha=0.7)
        plt.grid(axis="y", color="grey", alpha=0.5, linestyle="--")
        plt.xlabel("시간")
        plt.ylabel("")
        plt.xticks(df["date"][::3400])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.title("EX1.MD_PV")
        # for s in ["top", "right"]:
        #     ax.spines[s].set_visible(False)
        st.pyplot(fig1)

    with col8:
        fig2, ax = plt.subplots(figsize=(8, 4))
        plt.plot(df["date"], df["EX1.MD_TQ"], color="coral", alpha=0.7)
        plt.grid(axis="y", color="grey", alpha=0.5, linestyle="--")
        plt.xlabel("시간")
        plt.ylabel("")
        plt.xticks(df["date"][::3400])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.title("EX1.MD_TQ")
        # for s in ["top", "right"]:
        #     ax.spines[s].set_visible(False)
        st.pyplot(fig2)

with st.container(border=True):
    start = pd.to_datetime("13:00:00").time()
    end   = pd.to_datetime("14:00:00").time()
    df_sub = df[(df['date'].dt.time >= start) & (df['date'].dt.time <= end)].copy()
    df_sub = df_sub.sort_values('date').reset_index(drop=True)

    df_sub['fail_flag'] = (df_sub['passorfail'] == 1).astype(int)
    df_sub['run_id'] = (df_sub['fail_flag'].ne(df_sub['fail_flag'].shift())).cumsum()
    
    fail_ranges_sub = []
    for _, g in df_sub.groupby('run_id', sort=False):
        if g['fail_flag'].iloc[0] == 1:
            fail_ranges_sub.append((g['date'].iloc[0], g['date'].iloc[-1]))

    fig3, ax = plt.subplots(figsize=(15, 4))
    plt.plot(df_sub['date'], df_sub[feature], color = 'blue', alpha = 0.5)

    for (s, e) in fail_ranges_sub:
        ax.axvspan(s, e, color='red', alpha=0.14, edgecolor='red', linestyle='--', linewidth=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(df_sub["date"][::140])
    plt.xlabel("시간")
    plt.title(f"{feature} 불량 발생 전 후 그래프")
    plt.grid(axis="y", color="grey", alpha=0.5, linestyle="--")
    st.pyplot(fig3)
