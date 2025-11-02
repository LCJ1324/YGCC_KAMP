import streamlit as st
import pandas as pd

@st.cache_resource
def _minmax_store():
    # 리런/새로고침/멀티페이지 간에 공유되는 단일 저장소
    return {"table": None}

def save_minmax(df: pd.DataFrame) -> None:
    _minmax_store()["table"] = df.copy()

def load_minmax() -> pd.DataFrame | None:
    return _minmax_store()["table"]

def reset_minmax() -> None:
    _minmax_store()["table"] = None