import logging

from database import Database
import uuid
from datetime import datetime,timedelta
from tool import chat_agent, analysis
import streamlit as st
import json
from core.models import Class
import pandas as pd
from tool.study_report import LearningReportGenerator
import os
from fpdf import FPDF

class ConversationService:
    def __init__(self, student_id):
        self.student_id = student_id
        self.conversation_manager = chat_agent.ConversationManager()

        self.title_prompt_template  = """
            请阅读以下对话，提取一个不超过6个字的题目
            
            [对话记录]
            {coversation_history}
        """

        self.knowledge_points_template = """
            请根据以下对话记录，从知识点列表中提取出相符合的知识点（4个以内），并严格按照JSON返回(仅包含points字段)
            提取知识点必须是知识点列表中的，不允许产生新知识点
            [知识点列表]
            {points}
            
            [对话记录]
            {history}
            
            返回JSON形式(仅包含points字段,只需要输出知识点的名字)
        """

    def load_history(self):
        """加载历史对话"""
        return Database.load_conversation_history(self.student_id)

    def create_conversation(self):
        """创建新对话"""
        new_conversation = {
            "conversation_id": str(uuid.uuid4()),
            "student_id": self.student_id,
            "title": "新对话",
            "messages": [],
            "knowledges": [],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        return new_conversation

    def save_conversation(self, conversation):
        """保存对话"""
        Database.save_conversation(conversation)

    def get_ai_response(self, prompt):
        use_web_search = st.session_state["web_search"]
        if(st.session_state["deep_thought"]):
            ai_response, web_summary, thought_text = self.conversation_manager.process_conversation(
                prompt, use_web_search=use_web_search
            )
        else:
            ai_response, web_summary, reasoning_text = self.conversation_manager.thinking_input(
                prompt, use_web_search=use_web_search
            )
        return ai_response

    def update_coversation_title(self):
        length = len(st.session_state.current_conversation["messages"])
        if  length <= 2 or  length % 6 ==0:
            last_msgs = " ".join(
                [m["content"] for m in st.session_state.current_conversation["messages"][-6:]])
            title_prompt = self.title_prompt_template.format(coversation_history=last_msgs)
            new_title = self.conversation_manager.get_direct_response(title_prompt)
            st.session_state.current_conversation["title"] = new_title

    def get_knowledge_points(self, question, answer):
        message = f"student : {question}\nassitent : {answer}"

        with open("c_language.json", 'r', encoding='utf-8') as f:
            points = json.dumps(json.load(f), ensure_ascii=False, indent=4)
        points_prompt = self.knowledge_points_template.format(history=message,points=points)
        response = self.conversation_manager.get_direct_response(points_prompt)

        try:
            cleaned = response.replace("```json", "").replace("json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            self.save_knowlegdes_points(data["points"])
        except:
            logging.error("知识点提取失败")

    def save_knowlegdes_points(self, points):
        st.session_state.current_conversation["knowledges"].append((datetime.now(),points))


    def clear_memory(self):
        self.conversation_manager.clear_memory()



class ReportService:
    def __init__(self, student_id):
        self.student_id = student_id
        self.knowledge_points = None
        self.report_service_init(datetime.now()-timedelta(weeks=1),datetime.now())
        self.learning_report_generator = LearningReportGenerator(os.environ.get("DASHSCOPE_API_KEY"))

    def report_service_init(self, start_date, end_date):
        self.knowledge_points = Database.load_knowledge_points_by_date(self.student_id, start_date, end_date)
        self.messages = Database.load_coversations_by_date(self.student_id, start_date, end_date)
        self.df = pd.DataFrame(self.knowledge_points, columns=['timestamp', 'knowledge_points'])
        self.data = []
        for points in self.knowledge_points:
            self.data.append(points[1])
        self.freq = analysis.knowledge_frequency_analysis(self.data)

    def messages_to_json(self):
        message_list = []
        for i in range(0, len(self.messages), 2):
            message_list.append({
                "timestamp": self.messages[i]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "question": self.messages[i]["content"],
                "answer": ""
            })

        # 转换为 JSON 格式字符串
        conversation_json = json.dumps(message_list, indent=4, ensure_ascii=False)

        return conversation_json

    def generator_report(self):
        json_text = self.messages_to_json()

        with st.spinner("正在生成学习报告，请稍候..."):
            final_report, keywords = self.learning_report_generator.generate_report(json_text)
            if final_report:
                st.markdown("### 生成的学习报告")
                print(final_report)
                st.markdown(final_report)
                st.success("报告生成完成！")
                st.session_state["keywords"] = keywords
                # 记录报告：编号为 日期+序号
                today_str = datetime.now().strftime("%Y%m%d")
                count_today = sum(
                    1 for report in st.session_state.report_history if report["id"].startswith(today_str)) + 1
                report_id = f"{today_str}-{count_today:03d}"
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.report_history.append({
                    "id": report_id,
                    "timestamp": timestamp_str,
                    "report": final_report
                })
                # PDF 导出，使用支持 Unicode 的字体
                pdf = FPDF()
                pdf.add_page()
                font_path = os.path.join(os.path.dirname(__file__), "chinese.simhei.ttf")
                pdf.add_font("DejaVu", "", font_path, uni=True)
                pdf.set_font("DejaVu", size=12)
                for line in final_report.split('\n'):
                    pdf.multi_cell(0, 10, line)
                pdf_output = pdf.output(dest="S").encode("latin1", errors="replace")
                st.download_button("下载 PDF", data=pdf_output, file_name=f"{report_id}.pdf",
                                   mime="application/pdf")

    def knowledge_points_freqency_report(self):
        st.header("2️⃣ 知识点频率分析")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 知识点频率表格")
            analysis.plot_frequency_table(self.freq)
        with col2:
            st.subheader("☁️ 知识点词云图")
            analysis.plot_frequency_wordcloud_streamlit(self.freq)
        st.markdown("---")
        st.subheader("3️⃣ 知识点频率TOP10图表")
        chart_type = st.radio("选择图表类型：", options=["柱状图", "饼图"], horizontal=True, key="chart_type")
        if chart_type == "柱状图":
            analysis.plot_top_frequency_bar(self.freq, top_n=10)
        else:
            analysis.plot_top_frequency_pie(self.freq, top_n=10)

    def collinear_report(self):
        st.header("🔄 知识点共现分析")
        with st.expander("📖 分析说明与解读指南", expanded=True):
            st.markdown("""
                               ### 如何解读共现分析？
                               1. **共现频率表**：显示知识点两两组合的出现次数，高频组合提示教学中的常见知识关联
                               2. **热力图**：颜色越深表示共现频率越高，对角线显示单个知识点出现频次
                               3. **网络图**：
                                  - 节点大小反映知识点出现频率
                                  - 连线粗细表示共现强度
                                  - 紫色节点表示核心枢纽知识点
                                  - 紧密连接的群落提示知识模块

                               ### 教学应用价值：
                               ✅ 发现高频组合 → 优化课程设计中的知识点搭配  
                               ✅ 识别核心节点 → 加强重点知识点的教学  
                               ✅ 发现知识群落 → 建立模块化教学体系  
                               ✅ 定位薄弱环节 → 发现应加强关联的教学点

                               *示例：若"三角函数"与"向量"高频共现，建议在教学中强化二者的综合应用训练*
                               """)
        analysis.analyze_knowledge_cooccurrence(self.data)
        st.markdown("---")
        st.subheader("🕸️ 知识点共现网络")
        col1, col2 = st.columns(2)
        with col1:
            min_cooccurrence = st.slider("最小共现阈值", min_value=1, max_value=10, value=1,
                                         help="只显示共现次数大于等于此值的知识点对")
        with col2:
            max_nodes = st.slider("最大节点数量", min_value=5, max_value=30, value=15,
                                  help="限制网络图中显示的知识点数量")
        analysis.build_knowledge_network(self.data, self.freq, min_cooccurrence, max_nodes)

    def timing_report(self):
        if 'timestamp' in self.df.columns:
            st.header("⏰ 时序趋势分析")
            with st.expander("🔍 分析维度说明", expanded=True):
                st.markdown("""
                ### 本模块分析维度包括：
                1. **学习会话分析**：识别连续学习时段和间隔
                2. **学习强度分析**：分析每日/每周学习规律
                3. **知识焦点迁移**：跟踪知识点关注度变化
                """)

            df_enriched, session_stats = analysis.analyze_learning_sessions(self.df)
            analysis.analyze_knowledge_learning_curve(df_enriched)
            analysis.analyze_learning_intensity(df_enriched)

        else:
            st.error("无法进行时序分析，数据中缺少 timestamp 列")

    def time_preference_report(self):
        if 'timestamp' in self.df.columns:
            st.header("⏱️ 学生提问时间偏好分析")
            analysis.analyze_time_preference(self.df)
        else:
            st.error("无法进行时间分析，数据中缺少 timestamp 列")

    def daily_points_report(self):
        if 'timestamp' in self.df.columns:
            st.header("📈 每日知识点组成分析")
            analysis.analyze_daily_knowledge_composition(self.df)
        else:
            st.error("无法进行每日分析，数据中缺少 timestamp 列")

    def causal_knowledge_report(self):
        if 'timestamp' in self.df.columns:
            analysis.analyze_causal_relationships(self.data)
        else:
            st.error("需要时间戳数据进行因果时序分析")

    def personalize_report(self):
        st.header("✨ 个性化反馈")
        analysis.create_learning_profile(self.df, self.data)

    def depth_timing_report(self):
        analysis.advanced_time_series_analysis(self.df)

    def memory_report(self):
        if 'timestamp' in self.df.columns:
            analysis.analyze_memory_persistence(self.df.copy())
        else:
            st.error("需要时间戳数据进行记忆持久性分析")

class ClassService:
    def __init__(self, student_id):
        self.student_id = student_id
        self.student_info = None

    def get_joined_classes(self):
        """获取学生加入的班级"""
        if self.student_info == None:
            self.student_info = Database.find_user({"student_id": self.student_id})

        return self.student_info.get("classes", [])

    def join_class(self, invite_code):
        """加入班级"""
        target_class = Class.find_by_invite_code(invite_code.upper())
        if target_class:
            if target_class.class_id in self.get_joined_classes():
                st.error("您已加入该班级")
            else:
                try:
                    # 添加学生到班级
                    target_class.add_student(st.session_state.user["student_id"])
                    # 更新学生信息
                    Database.add_user_class(st.session_state.user["user_id"], target_class.class_id)
                    st.success(f"成功加入班级：{target_class.class_name}")
                    st.rerun()
                except Exception as e:
                    st.error("加入班级失败，请稍后重试")
        else:
            st.error("无效的邀请码")

class ExerciseService:
    def __init__(self, student_id):
        self.student_id = student_id
