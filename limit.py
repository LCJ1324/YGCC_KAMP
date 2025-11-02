import streamlit as st
import pandas as pd
from utils.minmax_store import save_minmax
st.set_page_config(layout="wide")

st.title('공정 변수 한계 범위 설정')
def load_data():
    df = pd.read_csv("2. 소성가공 품질보증 AI 데이터셋.csv")
    return df

df = load_data()
df = df.dropna()
df['passorfail'] = df['passorfail'].astype(int)
df["date"] = pd.to_datetime(df["date"])

columns = df.drop(columns = ['date', 'passorfail']).columns

user_minmax = pd.DataFrame(index=columns, columns=["min", "max"])

col1, col2, col3 = st.columns([2, 0.5, 1])

with col1 :
    st.subheader("각 컬럼별 Min/Max 범위를 입력하세요:")

    init_minmax = pd.DataFrame({
        "min": df[columns].min(),
        "max": df[columns].max()
    })

    edited_minmax = st.data_editor(init_minmax, use_container_width=True)

with col3 :
    out_counts = {}
    for col in columns:
        col_min = float(edited_minmax.loc[col, "min"])
        col_max = float(edited_minmax.loc[col, "max"])
        out_counts[col] = ((df[col] < col_min) | (df[col] > col_max)).sum()

    out_df = pd.DataFrame({
        "min": edited_minmax["min"],
        "max": edited_minmax["max"],
        "out_of_range_count": pd.Series(out_counts)
    })

    st.subheader("Out-of-Range Count 결과")
    st.dataframe(out_df['out_of_range_count'], use_container_width=True)

    save_minmax(out_df)