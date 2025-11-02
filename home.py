import streamlit as st

pages = [
    st.Page("main.py", title="메인 페이지", icon="🟠", default=True),
    st.Page("limit.py", title="공정 변수 한계 범위 설정", icon="🟢"),
    st.Page('graph.py', title = '날짜별 공정현황', icon="🔵"),
    st.Page('machinelearning.py', title = '불량 예측 머신러닝', icon="🟣")
]

selected_page = st.navigation(pages)

selected_page.run()