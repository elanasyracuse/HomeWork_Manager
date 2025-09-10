import streamlit as st


st.set_page_config(page_title="🧪 Multipage Labs", page_icon="🧪", layout="centered")

HW1 = st.Page("HW_1.py", title="Home Work 1 – Document QA", icon="📄")
HW2 = st.Page("HW_2.py", title="Home Work 2 – Document QA (default)", icon="📘", default=True)

nav = st.navigation({"Home Work": [HW1, HW2]})
nav.run()