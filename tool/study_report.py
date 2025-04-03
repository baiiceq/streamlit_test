import json
import requests
from langchain import LLMChain, PromptTemplate
from langchain_community.llms import Tongyi
import os
from fpdf import  FPDF
import datetime
class LearningReportGenerator:
    def __init__(self, api_key):
        """初始化学习报告生成器，设置大模型和搜索工具"""
        self.llm = Tongyi(
            model="qwen-max-2025-01-25",
            temperature=0.8,
            openai_api_key=api_key,
            openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )
        self.search_tool = self._bocha_websearch_tool

    def _bocha_websearch_tool(self, query: str, count: int = 20) -> str:
        """Bocha API 联网搜索功能实现"""
        url = 'https://api.bochaai.com/v1/web-search'
        headers = {
            'Authorization': f'Bearer sk-782dce9336b549299c48f4607376d0f2',  # 请替换为实际 API 密钥
            'Content-Type': 'application/json'
        }
        data = {
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": count
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            json_data = response.json()
            if json_data.get("code") != 200 or not json_data.get("data"):
                return "搜索服务暂时不可用"
            webpages = json_data["data"]["webPages"]["value"]
            return "\n\n".join(
                f"**标题**：[{p['name']}]({p['url']})\n\n摘要：{p['summary']}"
                for p in webpages[:5]
            )
        except Exception as e:
            return f"搜索失败：{str(e)}"

    def parse_conversation(self, json_text):
        """解析对话记录 JSON 文本"""
        try:
            logs = json.loads(json_text)
            if isinstance(logs, list) and all("timestamp" in log and "question" in log and "answer" in log for log in logs):
                return logs
            return None
        except Exception:
            return None

    def format_conversation(self, logs):
        """格式化对话记录"""
        formatted = ""
        for log in logs:
            formatted += f"**时间**：{log['timestamp']}\n\n**提问**：{log['question']}\n\n**回答**：{log['answer']}\n\n---\n\n"
        return formatted

    def generate_analysis(self, conversation_logs):
        """生成学习行为分析和知识掌握情况评估"""
        analysis_prompt_template = """
        请仔细阅读以下学生与大模型的对话记录，并完成以下两部分内容：
        ### 【一、学习行为分析】
        - **提问频次与时间分布**：统计学生提问的频次，指出高峰时段；
        - **常见提问主题**：列举学生常见的提问主题和问题类型。
        - **行为分析**：结合上述信息，给出详细的学生学习行为分析结果
        ### 【二、知识掌握情况评估】
        - **核心知识点掌握情况**：指出学生对核心知识点的理解程度；
        - **常见错误分析**：分析学生常见的错误和疑问，判断其薄弱环节，并列出薄弱知识点。
        - **行为分析**：结合上述信息，给出详细的学生学习行为分析结果
        请以 Markdown 格式输出，注意格式以及美观
        **对话记录**：
        {conversation_logs}
        """
        analysis_prompt = PromptTemplate(input_variables=["conversation_logs"], template=analysis_prompt_template)
        analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
        return analysis_chain.run(conversation_logs=conversation_logs)

    def extract_keywords(self, analysis):
        """提取薄弱知识点关键词"""
        keyword_prompt_template = """
        根据以下学习情况分析，提取出学生薄弱环节中涉及的核心知识点或主题，
        请只输出关键词，用逗号分隔：
        {analysis}
        """
        keyword_prompt = PromptTemplate(input_variables=["analysis"], template=keyword_prompt_template)
        keyword_chain = LLMChain(llm=self.llm, prompt=keyword_prompt)
        return keyword_chain.run(analysis=analysis).strip()

    def search_online_resources(self, query):
        """搜索在线学习资源"""
        return self.search_tool(query)

    def generate_suggestions(self, analysis, online_resources):
        """生成个性化学习建议"""
        suggestion_prompt_template = """
        根据以下学生的学习情况分析：
        {analysis}
        并结合实时在线搜索得到的学习资源摘要：
        {online_resources}
        请为学生提供个性化学习建议，包括：
            - **学习情况说明**：根据对话记录的情况，对学生的学习情况进行详细的分析和说明。
            - **学习资源推荐**：给出具体的网络课程、视频、文章等链接，附上简短的资源简介和推荐理由；
            - **专项训练建议**：针对学生薄弱环节生成一些涉及相关知识点的具体训练习题及练习建议；
            - **学习计划调整**：结合学生目前的学习情况，给出调整学习计划的具体思路和建议。
        请以 Markdown 格式输出建议,注意格式以及美观。
        """
        suggestion_prompt = PromptTemplate(input_variables=["analysis", "online_resources"], template=suggestion_prompt_template)
        suggestion_chain = LLMChain(llm=self.llm, prompt=suggestion_prompt)
        return suggestion_chain.run(analysis=analysis, online_resources=online_resources)

    def generate_planning(self, analysis):
        """生成学习进步趋势与未来规划"""
        planning_prompt_template = """
        根据以下学生的学习情况分析：
        {analysis}
        请为学生总结其学习进步趋势，并给出未来规划，包括：
            - **学习进步趋势**：分析学生近期的进步和存在的问题；
            - **未来目标规划**：结合学生的知识点掌握情况和水平，制定符合实际的短期目标（1-3个月）和长期目标（半年到一年），并简要说明每个目标的重要性和具体实现路径。
        请以 Markdown 格式输出规划，注意格式以及美观。
        """
        planning_prompt = PromptTemplate(input_variables=["analysis"], template=planning_prompt_template)
        planning_chain = LLMChain(llm=self.llm, prompt=planning_prompt)
        return planning_chain.run(analysis=analysis)

    def generate_report(self, json_input):
        """生成完整的学习报告"""
        logs = self.parse_conversation(json_input)
        if logs is None:
            return None, None
        formatted_logs = self.format_conversation(logs)
        analysis_output = self.generate_analysis(formatted_logs)
        analysis_output = "\n".join(analysis_output.splitlines()[1:-1])
        keywords_output = self.extract_keywords(analysis_output)
        search_query = f"针对薄弱知识点 {keywords_output} 的学习资源推荐"
        online_resources = self.search_online_resources(search_query)
        suggestions_output = self.generate_suggestions(analysis_output, online_resources)
        suggestions_output = "\n".join(suggestions_output.splitlines()[1:-1])
        planning_output = self.generate_planning(analysis_output)
        planning_output = "\n".join(planning_output.splitlines()[1:-1])
        final_report = f"""
# 学习报告
## 一、学习行为分析
{analysis_output}
---
## 二、个性化学习建议
{suggestions_output}
---
## 三、学习进步趋势与未来规划
{planning_output}
        """
        return final_report, keywords_output
import re
from langchain import LLMChain, PromptTemplate
from langchain_community.llms import Tongyi

class ExamGenerator:
    def __init__(self, api_key):
        """初始化试题生成器，设置大模型"""
        self.llm = Tongyi(
            model="qwen-max-2025-01-25",
            temperature=0.8,
            openai_api_key=api_key,
            openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

    def generate_exam_questions(self, knowledge_points, question_types, question_count):
        """根据知识点、题型和数量生成试题"""
        exam_prompt_template = """
        请根据以下要求生成试题：
        知识点：{knowledge_points}
        题型：{question_types}
        题目数量：{question_count}
        格式要求：
        1. 如果题型中有判断题，则判断题必须包含明确的正误判断，答案只能写"对"或"错"
        2. 如果题型中有选择题，选择题选项请用A. B. C. D.开头
        3. 所有题型必须包含题目、答案、解析三部分
        示例：
        **题目**：水的沸点是100摄氏度（标准大气压下）
        **答案**：对
        **解析**：在标准大气压（1atm）下，水的沸点是100℃...
        请以 Markdown 格式输出试题列表，每道题之间用分隔线（---）隔开。
        请严格遵循上面对知识点、题型、题目数量的限制与要求
        """
        exam_prompt = PromptTemplate(
            input_variables=["knowledge_points", "question_types", "question_count"],
            template=exam_prompt_template
        )
        exam_chain = LLMChain(llm=self.llm, prompt=exam_prompt)
        return exam_chain.run(
            knowledge_points=knowledge_points,
            question_types=question_types,
            question_count=question_count
        )

    def parse_exam_questions(self, text):
        """解析生成的试题文本"""
        questions = []
        pattern = r'\*\*题目\*\*：(.*?)\n(.*?)\*\*答案\*\*：(.*?)\n\*\*解析\*\*：(.*?)\n---'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            question = match[0].strip()
            options_text = match[1].strip()
            answer = match[2].strip()
            explanation = match[3].strip()

            if "A." in options_text and "B." in options_text:
                question_type = "选择题"
                options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]
                answer = answer.split(".")[0].strip() if "." in answer else answer
            elif answer.lower() in ["对", "正确", "true", "yes", "√"]:
                question_type = "判断题"
                answer = "对"
                options = ["对", "错"]
            else:
                question_type = "问答题"
                options = None

            questions.append({
                "question": question,
                "options": options,
                "answer": answer,
                "explanation": explanation,
                "type": question_type
            })
        return questions
import streamlit as st

# 主函数
def main():
    st.sidebar.title("导航")
    page = st.sidebar.radio("选择页面", ("学习报告生成", "试题生成"))
    api_key = 'sk-e2492dea19b945059a9a05abb4d0fc0b'  # 请替换为实际 API 密钥

    if page == "学习报告生成":
        st.title("学习报告生成系统")
        report_generator = LearningReportGenerator(api_key)
        json_input = st.text_area("请粘贴对话记录（JSON 格式）：", height=300)
        if st.button("生成学习报告"):
            if json_input.strip() == "":
                st.error("请输入有效的对话记录 JSON 数据。")
            else:
                with st.spinner("正在生成学习报告，请稍候..."):
                    final_report, keywords = report_generator.generate_report(json_input)
                    if final_report:
                        st.markdown("### 生成的学习报告")
                        st.markdown(final_report)
                        st.success("报告生成完成！")
                        st.session_state["keywords"] = keywords
                        # 记录报告：编号为 日期+序号
                        today_str = datetime.datetime.now().strftime("%Y%m%d")
                        count_today = sum(
                            1 for report in st.session_state.report_history if report["id"].startswith(today_str)) + 1
                        report_id = f"{today_str}-{count_today:03d}"
                        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.report_history.append({
                            "id": report_id,
                            "timestamp": timestamp_str,
                            "report": final_report
                        })
                        # PDF 导出，使用支持 Unicode 的字体
                        pdf = FPDF()
                        pdf.add_page()
                        font_path = os.path.join(os.path.dirname(__file__), "SimSun.ttf")
                        pdf.add_font("DejaVu", "", font_path, uni=True)
                        pdf.set_font("DejaVu", size=12)
                        for line in final_report.split('\n'):
                            pdf.multi_cell(0, 10, line)
                        pdf_output = pdf.output(dest="S").encode("latin1", errors="replace")
                        st.download_button("下载 PDF", data=pdf_output, file_name=f"{report_id}.pdf",
                                           mime="application/pdf")
    elif page == "试题生成":
        st.title("答题系统")
        exam_generator = ExamGenerator(api_key)
        default_knowledge = st.session_state.get("keywords", "牛顿第一定律, 惯性")
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

if __name__ == "__main__":
    main()