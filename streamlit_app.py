import streamlit as st

st.set_page_config(page_title="🧪 Multipage Labs", page_icon="🧪", layout="centered")

HW1 = st.Page("HW_1.py", title="Home Work 1", icon="📄")
HW2 = st.Page("HW_2.py", title="Home Work 2", icon="📘")
HW3 = st.Page("HW_3.py", title="Home Work 3", icon="📒")
HW4 = st.Page("HW_4.py", title="Home Work 4", icon="📒",default=True)

nav = st.navigation({"Home Work": [HW1, HW2, HW3, HW4]})
nav.run()