import streamlit as st

st.set_page_config(page_title="Team Profiles", layout="wide")

st.markdown(
    """
    <style>
    .profile-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-size: 2em;
        text-align: center;
        margin-bottom: 40px;
    }
    .profile-name {
        font-size: 1.5em;
        font-weight: bold;
    }
    .linkedin-link {
        color: #0077b5;
        text-decoration: none;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Welcome to My Streamlit App</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        '<div class="profile-card">' +
        '<div class="profile-name">Tariq Alshammari</div>' +
        '<a class="linkedin-link" href="https://www.linkedin.com/in/tariq-alshammari-173a91298">LinkedIn Profile</a>' +
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="profile-card">' +
        '<div class="profile-name">Lamees</div>' +
        '<a class="linkedin-link" href="https://www.linkedin.com/in/tariq-alshammari-173a91298">LinkedIn Profile</a>' +
        '</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="profile-card">' +
        '<div class="profile-name">Basel</div>' +
        '<a class="linkedin-link" href="https://www.linkedin.com/in/tariq-alshammari-173a91298">LinkedIn Profile</a>' +
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="profile-card">' +
        '<div class="profile-name">Sama</div>' +
        '<a class="linkedin-link" href="https://www.linkedin.com/in/tariq-alshammari-173a91298">LinkedIn Profile</a>' +
        '</div>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="profile-card" style="text-align: center;">' +
    '<div class="profile-name">Project Repository</div>' +
    '<a class="linkedin-link" href="https://github.com/AI-bootcamp/capstone-project-team01">Checkmate Project</a>' +
    '</div>',
    unsafe_allow_html=True
)

