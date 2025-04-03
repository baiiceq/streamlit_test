# auth/components/login_form.py
import streamlit as st
from auth.services import AuthSystem
from database import Database

def show_login_form():
    st.title("登录界面")
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")

        if st.form_submit_button("登录"):
            user, msg = AuthSystem.login(Database.get_db(), username, password)
            if user:
                st.session_state.user = user
                if user["role"] == "student" :
                    st.session_state["current_page"] = "student"
                else:
                    st.session_state["current_page"] = "teacher"
                st.success("登录成功")
                st.rerun()
            else:
                st.error(msg)
