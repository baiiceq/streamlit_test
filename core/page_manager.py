import streamlit as st
from auth.components.register_form import show_register_form
from auth.components.login_form import show_login_form
from student.components.sidebar_component import SidebarComponent
from student.components.chat_componets import ChatComponent
from student.components.class_component import ClassComponent
from student.components.report_component import ReportComponent
from student.components.exercise_compoent import ExerciseComponent
from teacher.components.sidebar_component import tSidebarComponent
from teacher.components.class_list_component import tListComponent
from teacher.components.class_detail_component import tDetailComponent

class PageManager:
    def __init__(self):
        # 初始化时，默认显示登录页面
        self.pages = {
            "login": self.login_page,
            "register": self.register_page,
            "student": self.student,
            "teacher": self.teacher,
        }
        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = 'login'

    def show_page(self):
        # 获取当前显示的页面
        current_page = st.session_state['current_page']
        page_function = self.pages.get(current_page)
        if page_function:
            page_function()
        else:
            st.error("Page not found!")

    def login_page(self):
        show_login_form()
        if st.button("注册"):
            st.session_state["current_page"] = "register"
            return

    def register_page(self):
        show_register_form()
        if st.button("登录"):
            st.session_state["current_page"] = "login"
            return

    def student(self):
        chat_component = ChatComponent()
        classroom_component = ClassComponent(st.session_state["user"]["student_id"])
        report_component = ReportComponent()
        exercise_component = ExerciseComponent(st.session_state["user"]["student_id"])
        st.title("学生面板")
        with st.sidebar:
            SidebarComponent.show_sidebar()
        if st.session_state["student_page"] == "chat":
            chat_component.show_chat_interface()
        elif st.session_state["student_page"] == "classroom":
            classroom_component.show_class_interface()
        elif st.session_state["student_page"] == "report":
            report_component.show_report_interface()
        elif st.session_state["student_page"] == "exercise":
            exercise_component.show_exam_interface()


    def teacher(self):
        tlist_component = tListComponent()
        tdetail_component = tDetailComponent()
        with st.sidebar:
            tSidebarComponent.show_sidebar()
        if st.session_state["teacher_page"] == "list":
            tlist_component.show_list_interface()
        elif st.session_state["teacher_page"] == "detail":
            tdetail_component.show_detail_interface()

