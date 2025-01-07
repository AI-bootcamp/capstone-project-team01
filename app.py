import streamlit as st

pg = st.navigation([
    st.Page("app_image.py", title="Image chess tracking", icon=":material/info:"),
    st.Page("app_upload.py", title="Upload", icon="🔗"),
    st.Page("app_live.py", title="Live chess tracking", icon=":material/info:"),
])

pg.run()