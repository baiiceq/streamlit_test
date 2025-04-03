import streamlit as st
from student.services import ReportService
from datetime import datetime,timedelta
import pandas as pd


class ListService:
    def get_class_data(self, start_date, end_date):
        selected_class = st.session_state.selected_class
        json_text = []
        for student_id in selected_class.students:
            dict = {}
            dict["name"] = student_id
            dict["weak_points"] = self.get_student_weak_points(student_id, start_date, end_date)
            dict["study_period"] = self.get_student_study_period(student_id, start_date, end_date)
            dict["question_count"] = self.get_student_questions_cnt(student_id, start_date, end_date)
            json_text.append(dict)

        data = pd.DataFrame(json_text)
        return data


    @classmethod
    def get_student_weak_points(self, student_id, start_date, end_date):
        report_service = ReportService(student_id)
        report_service.report_service_init(start_date, end_date)
        weak_points = [item[0] for item in report_service.freq.most_common(5)]

        return weak_points

    @classmethod
    def get_student_study_period(self, student_id, start_date, end_date):
        report_service = ReportService(student_id)
        report_service.report_service_init(start_date, end_date)
        report_service.df['timestamp'] = pd.to_datetime(report_service.df['timestamp'])
        report_service.df['hour'] = report_service.df['timestamp'].dt.hour

        def get_time_period(hour):
            if 5 <= hour < 12:
                return '上午 (5:00-11:59)'
            elif 12 <= hour < 18:
                return '下午 (12:00-17:59)'
            else:
                return '晚上 (18:00-4:59)'

        report_service.df['time_period'] = report_service.df['hour'].apply(get_time_period)
        period_counts = report_service.df['time_period'].value_counts().reset_index()
        period_counts.columns = ['时间段', '提问数量']

        most_active_period = period_counts.iloc[0]['时间段']
        return most_active_period

    @classmethod
    def get_student_questions_cnt(self, student_id, start_date, end_date):
        report_service = ReportService(student_id)
        report_service.report_service_init(start_date, end_date)
        return len(report_service.knowledge_points)

class DetailService:

    def get_student_report(self, student_id):
        report_service = ReportService(student_id)
        current_date = datetime.now()
        time_col1, time_col2, time_col3 = st.columns([2, 2, 3])
        with time_col1:
            start_date = st.date_input("起始日期",
                                       value=current_date - timedelta(weeks=1),
                                       max_value=current_date - timedelta(weeks=1))
        with time_col2:
            end_date = st.date_input("结束日期",
                                     value=datetime.now(),
                                     min_value=start_date + timedelta(weeks=1))

        if st.button("确定时间段"):
            report_service.report_service_init(start_date, end_date)

        main_tab1, main_tab2, main_tab3 = st.tabs(["核心分析", "高级功能", "生成学习报告"])

        with main_tab1:
            tab1, tab2, tab3, tab4 = st.tabs(["📊 知识点频率分析",
                                              "🔄 知识点共现分析",
                                              "⏰ 时序趋势分析",
                                              "⏱️ 时间偏好分析"])

            with tab1:
                report_service.knowledge_points_freqency_report()

            with tab2:
                report_service.collinear_report()

            with tab3:
                report_service.timing_report()

            with tab4:
                report_service.time_preference_report()

        with main_tab2:
            tab5, tab6, tab7, tab8, tab9 = st.tabs(["📈 每日知识点组成",
                                                    "🧠 因果知识图谱",
                                                    "✨ 个性化反馈",
                                                    "🔍 深度时序分析",
                                                    "🧠 记忆持久性"])
            with tab5:
                report_service.daily_points_report()

            with tab6:
                report_service.causal_knowledge_report()

            with tab7:
                report_service.personalize_report()

            with tab8:
                report_service.depth_timing_report()

            with tab9:
                report_service.memory_report()

        with main_tab3:
            if st.button("生成学习报告"):
                report_service.generator_report()
