# auth/components/register_form.py
import streamlit as st
from auth.services import AuthSystem
from database import Database

def show_register_form():
    st.title("注册界面")
    with st.form("register_form"):
        role = st.selectbox("角色", ["student", "teacher"])
        username = st.text_input("用户名")
        email = st.text_input("邮箱")
        password = st.text_input("密码", type="password")

        # 学生/教师特有字段
        if role == "student":
            student_id = st.text_input("学号（8位数字）")
        else:
            teacher_id = st.text_input("工号")

        if st.form_submit_button("发送验证码"):
            success, msg = AuthSystem.send_verification_code(email)
            if success:
                st.success(msg)
            else:
                st.error(msg)

        verification_code = st.text_input("输入验证码")

        if st.form_submit_button("注册"):
            user_data = {
                "username": username,
                "password": password,
                "role": role,
                "email": email,
                "student_id": student_id if role == "student" else None,
                "teacher_id": teacher_id if role == "teacher" else None
            }
            success, msg = AuthSystem.register(Database.get_db(), user_data, verification_code)
            if success:
                st.success("注册成功，请登录")
            else:
                st.error(msg)