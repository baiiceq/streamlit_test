import streamlit as st
from core.models import Class

class tSidebarComponent:
    @staticmethod
    def show_sidebar():
        page_options = ["班级列表", "班级详情"]

        # 映射字典
        page_mapping = {
            "班级列表": "list",
            "班级详情": "detail"
        }

        # 获取当前页面
        current_page = st.session_state.get("teacher_page", "list")
        page = st.radio("页面", page_options, index=page_options.index(
            next(key for key, value in page_mapping.items() if value == current_page)))

        st.session_state["teacher_page"] = page_mapping[page]

        # 创建新班级
        with st.expander("➕ 创建新班级", expanded=False):
            with st.form("create_class_form"):
                class_name = st.text_input("班级名称", key="new_class_name")
                if st.form_submit_button("创建"):
                    if class_name:
                        new_class = Class(
                            class_name=class_name,
                            teacher_id=st.session_state.user["teacher_id"]
                        )
                        if new_class.save():
                            st.success(f"'{class_name}' 创建成功！邀请码：{new_class.invite_code}")
                        else:
                            st.error("创建失败，请重试")
                    else:
                        st.error("请输入班级名称")
        if st.button("退出登录"):
            st.session_state.clear()
            st.session_state.user = None
            st.session_state["current_page"] = "login"
            st.rerun()