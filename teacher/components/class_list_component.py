import streamlit as st
from core.models import Class
from teacher.services import ListService
from datetime import datetime, timedelta
import tool.teacher_report as tr


class tListComponent:
    def __init__(self):
        self.teacher_id = st.session_state.user["teacher_id"]
        self.list_service  = ListService()

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



        if st.session_state.selected_class:
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
            df_data = self.list_service.get_class_data(start_date, end_date)

            if st.button("确定时间"):
                df_data = self.list_service.get_class_data(start_date, end_date)

            if len(df_data) <= 4:
                st.error("班级人数不足5，无法生成")
            else:
                st.title("📈 班级学情分析系统")
                st.title(st.session_state.selected_class.class_name)
                st.markdown("本系统通过多维度数据分析，帮助教师快速掌握班级整体学习情况，识别共性薄弱环节。")

                tab1, tab2, tab3, tab4, tab5 = st.tabs(["薄弱知识点",
                                                  "学习行为",
                                                  "学生群体",
                                                  "知识点关联",
                                                  "AI报告"])
                with tab1:
                    self.weak_result = tr.analyze_weak_points(df_data)

                with tab2:
                    self.behavior_result = tr.analyze_learning_behavior(df_data)

                with tab3:
                    self.cluster_result = tr.analyze_student_clusters(df_data)

                with tab4:
                    self.network_result = tr.analyze_knowledge_network(df_data)

                with tab5:
                    analysis_results = {
                        "top_weak_points": self.weak_result['top_weak_points'],
                        "weak_distribution": self.weak_result['weak_distribution'],
                        "study_period_dist": self.behavior_result['period_dist'],
                        "question_stats": self.behavior_result['question_stats'],
                        "cluster_summary": {
                            "total_clusters": self.cluster_result['cluster_stats']['total_clusters'],
                            "cluster_dist": self.cluster_result['cluster_stats']['size_distribution'],
                            "silhouette": self.cluster_result['cluster_stats']['silhouette_score'],
                            "features": self.cluster_result['cluster_stats']['features']
                        },
                        "knowledge_network": {
                            "top_combinations": self.network_result['top_combinations'],
                            "network_density": self.network_result['network_density'],
                            "isolated_points": self.network_result.get('isolated', [])
                        }
                    }
                    tr.generate_analysis_report(df_data, analysis_results)

            st.title("班级原始数据")
            st.dataframe(df_data)



