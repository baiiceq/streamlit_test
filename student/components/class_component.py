import streamlit as st
from database import Database
from student.services import ClassService
from core.models import Class

class ClassComponent:
    def __init__(self, student_id):
        self.student_id = student_id
        self.joined_classes = []
        self.ClassService = ClassService(self.student_id)

    def show_class_interface(self):
        with st.sidebar:
            self.display_sidebar()


    def display_sidebar(self):
        """显示已加入的班级"""
        st.subheader("📚 我的班级")

        self.joined_classes = self.ClassService.get_joined_classes()
        if self.joined_classes:
            for class_id in self.joined_classes:
                class_info = Class.find_by_id(class_id)
                if class_info:
                    st.markdown(f"""
                                <div class="class-card">
                                    <div class="class-header">{class_info.class_name}</div>
                                    <div class="class-meta">
                                        <span>{class_info.teacher_id}</span>
                                        <span>{class_info.get_student_count()}人</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
        else:
            st.info("尚未加入任何班级")

        self.display_class_form()

    def display_class_form(self):
        """显示加入班级的表单"""
        with st.expander("➕ 加入新班级"):
            invite_code = st.text_input("请输入班级邀请码", max_chars=6, key="class_invite_code")
            if st.button("加入班级"):
                class_service = ClassService(self.student_id)
                target_class = class_service.join_class(invite_code)
                if target_class:
                    st.success(f"成功加入班级: {target_class.class_name}")
                else:
                    st.error("无效的邀请码")
