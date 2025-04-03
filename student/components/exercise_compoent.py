import streamlit as st
from datetime import datetime,timedelta
from student.services import ExerciseService
from tool.study_report import ExamGenerator
import os
from database import Database

class ExerciseComponent:
    def __init__(self, student_id):
        self.student_id = student_id
        self.exercise_service = ExerciseService(student_id)

    def show_exam_interface(self):
        st.title("答题系统")
        exam_generator = ExamGenerator(os.environ.get("DASHSCOPE_API_KEY"))
        knowledge_points = Database.load_knowledge_points_by_date(self.student_id, datetime.now() - timedelta(weeks=1), datetime.now())
        x = []
        for points in knowledge_points:
            for point in points[1]:
                x.append(point)
        knowledge_points = list(set(x))
        default_knowledge = ",".join(knowledge_points)
        weak_knowledge_input = st.text_input("请输入学生薄弱知识点（用逗号分隔）", default_knowledge)
        knowledge_list = [k.strip() for k in weak_knowledge_input.split(",") if k.strip()]
        selected_knowledge = st.multiselect("请选择知识点", options=knowledge_list, default=knowledge_list)
        question_types = ["选择题", "问答题", "程序设计题", "判断题"]
        selected_types = st.multiselect("请选择题型", options=question_types, default=["选择题", "判断题"])
        question_count = st.slider("选择试题数量", min_value=1, max_value=20, value=5)
        if st.button("生成试题"):
            if not selected_knowledge or not selected_types:
                st.error("请至少选择一个知识点和题型！")
            else:
                with st.spinner("正在生成试题，请稍候..."):
                    exam_text = exam_generator.generate_exam_questions(
                        ", ".join(selected_knowledge),
                        ", ".join(selected_types),
                        question_count
                    )
                    questions = exam_generator.parse_exam_questions(exam_text)
                    st.session_state["questions"] = questions
                    st.session_state["user_answers"] = [None] * len(questions)
                    st.success("试题生成完成，请开始作答！")

        if "questions" in st.session_state and st.session_state["questions"]:
            st.markdown("### 试题列表")
            for i, q in enumerate(st.session_state["questions"]):
                st.markdown(f"**第 {i + 1} 题**（{q['type']}）：{q['question']}")
                if q["type"] == "选择题":
                    st.session_state["user_answers"][i] = st.radio("选项", q["options"], key=f"q{i}")
                elif q["type"] == "判断题":
                    st.session_state["user_answers"][i] = st.radio("判断", q["options"], key=f"q{i}")
                else:
                    st.session_state["user_answers"][i] = st.text_input("答案", key=f"q{i}")
