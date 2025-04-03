import streamlit as st
from core.models import Class


class tListComponent:
    def __init__(self):
        self.teacher_id = st.session_state.user["teacher_id"]

    def show_list_interface(self):
        st.divider()
        teacher_classes = Class.find_teacher_classes(st.session_state.user["teacher_id"])

        # 检查老师是否有创建班级
        if not teacher_classes:
            st.info("您尚未创建任何班级")
        else:

            for cls in teacher_classes:
                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button(
                            f"{cls.class_name} ({cls.get_student_count()}人)",
                            key=f"cls_{cls.class_id}",
                            help=f"邀请码：{cls.invite_code}"
                    ):
                        # 当点击班级时，保存班级信息
                        st.session_state.selected_class = cls
                        st.success(f"您已成功选中 {cls.class_name}")
                with col2:
                    st.markdown(f"##### 邀请码:{cls.invite_code}")