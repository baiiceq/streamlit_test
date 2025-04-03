import streamlit as st
from core.models import Class
from teacher.services import DetailService

class tDetailComponent:
    def __init__(self):
        self.teacher_id = st.session_state.user["teacher_id"]
        self.detail_service = DetailService()
    def show_detail_interface(self):
        if not st.session_state.selected_class:
            st.error("当前未选择班级")
        else:
            selected_class = st.session_state.selected_class
            st.subheader(f"班级：{selected_class.class_name}")

            group_size = 5
            total_students = len(selected_class.students)
            total_groups = (total_students // group_size) + (1 if total_students % group_size != 0 else 0)

            # 使用selectbox选择学生组
            group_number = st.selectbox("选择第几页", range(1, total_groups + 1))

            # 计算当前组的学生索引
            start_idx = (group_number - 1) * group_size
            end_idx = min(start_idx + group_size, total_students)

            # 显示当前组的学生按钮
            for student_id in selected_class.students[start_idx:end_idx]:
                col1,col2 = st.columns([3, 2])
                with col1:
                    if st.button(f"查看学生 {student_id}", key=f"student_{student_id}"):
                        st.session_state.selected_student_id = student_id
                        st.rerun()
                with col2:
                    if st.button(f"删除学生 {student_id}"):
                        selected_class.remove_student(student_id)

            # 如果选中了学生，显示学生的详细信息
            if 'selected_student_id' in st.session_state:
                selected_student_id = st.session_state.selected_student_id
                st.title(f"学生{selected_student_id} 的详细情况")

                self.detail_service.get_student_report(selected_student_id)