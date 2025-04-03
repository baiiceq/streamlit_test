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
        keywords_output = self.extract_keywords(analysis_output)
        search_query = f"针对薄弱知识点 {keywords_output} 的学习资源推荐"
        online_resources = self.search_online_resources(search_query)
        suggestions_output = self.generate_suggestions(analysis_output, online_resources)
        planning_output = self.generate_planning(analysis_output)
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
        exam_prompt_template = """请根据以下要求生成c语言试题：
知识点：{knowledge_points}
题型：{question_types}
题目数量：{question_count}

严格按照以下JSON格式输出：
[
  {{
    "question": "题干内容（包含任何代码）",
    "options": ["A.选项1", "B.选项2", ...], // 选择题/判断题必填，其他题型为null
    "answer": "正确答案", // 选择题用大写字母，判断题用"对/错"
    "explanation": "答案解析",
    "type": "题型名称"
  }}
]

生成规则：
1. 题干中的代码直接包含在question字段中，不要放在options里
2. 选择题选项必须用A. B. C. D.开头
3. 判断题必须有且只有"对"和"错"两个选项
4. 问答题的options设为null
5. 确保生成严格有效的JSON格式

示例：
[
  {{
    "question": "以下Python代码的输出是什么？\\nprint(len({{'a': 1, 'b': 2}}))",
    "options": ["A. 1", "B. 2", "C. 报错", "D. 4"],
    "answer": "B",
    "explanation": "字典的len()方法返回键的数量",
    "type": "选择题"
  }},
  {{
    "question": "HTTP协议是无状态的",
    "options": ["对", "错"],
    "answer": "对",
    "explanation": "HTTP协议本身不保存客户端状态",
    "type": "判断题"
  }}
]"""

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
        """解析生成的试题JSON文本"""
        try:
            # 提取JSON代码块
            json_str = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if json_str:
                text = json_str.group(1).strip()

            questions = json.loads(text)
            processed = []

            for q in questions:
                # 字段校验
                required_fields = ["question", "answer", "explanation", "type"]
                if not all(field in q for field in required_fields):
                    continue

                # 答案标准化
                q_type = q["type"].strip()
                answer = str(q["answer"]).strip()
                options = q.get("options")

                # 处理选择题
                if q_type == "选择题":
                    if not options or len(options) < 2:
                        continue
                    # 提取答案字母
                    answer = re.sub(r'[^A-Da-d]', '', answer).upper()
                    if not answer:
                        continue
                    # 标准化选项前缀
                    options = [f"{chr(65 + i)}. {opt.split('. ')[1]}"
                               for i, opt in enumerate(options[:4])]

                # 处理判断题
                elif q_type == "判断题":
                    answer = "对" if answer.lower() in ["对", "正确", "true", "yes", "√"] else "错"
                    options = ["对", "错"]

                # 处理问答题
                else:
                    options = None

                processed.append({
                    "question": q["question"].strip(),
                    "options": options,
                    "answer": answer,
                    "explanation": q["explanation"].strip(),
                    "type": q_type
                })

            return processed

        except json.JSONDecodeError:
            st.error("试题解析失败，请重试或检查提示词")
            return []
        except Exception as e:
            st.error(f"解析异常：{str(e)}")
            return []

import streamlit as st

# 主函数
def main():
    # 初始化session state
    if "report_history" not in st.session_state:
        st.session_state.report_history = []
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = []
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
        st.title("智能试题生成系统")
        exam_generator = ExamGenerator(api_key)

        # 知识点输入
        col1, col2 = st.columns([3, 2])
        with col1:
            default_knowledge = st.session_state.get("keywords", "Python基础, 数据结构")
            knowledge_input = st.text_input("输入知识点（逗号分隔）", default_knowledge)
            knowledge_list = [k.strip() for k in knowledge_input.split(",") if k.strip()]

        # 题型和数量选择
        with col2:
            question_types = st.multiselect(
                "选择题型",
                options=["选择题", "判断题", "问答题", "编程题"],
                default=["选择题", "判断题"]
            )
            question_count = st.slider("题目数量", 1, 20, 5)

        # 生成按钮
        if st.button("生成试题", help="点击生成个性化试题"):
            if not knowledge_list or not question_types:
                st.error("请至少选择一个知识点和题型！")
            else:
                with st.spinner("正在生成试题，请稍候..."):
                    exam_text = exam_generator.generate_exam_questions(
                        ", ".join(knowledge_list),
                        ", ".join(question_types),
                        question_count
                    )
                    questions = exam_generator.parse_exam_questions(exam_text)
                    if questions:
                        st.session_state.questions = questions
                        st.session_state.user_answers = [None] * len(questions)
                        st.success("试题生成成功！")
                    else:
                        st.error("试题生成失败，请调整参数后重试")

        # 显示试题
        if st.session_state.questions:
            st.divider()
            st.subheader("生成试题列表")

            for i, q in enumerate(st.session_state.questions):
                with st.expander(f"第 {i + 1} 题（{q['type']}）", expanded=True):
                    st.markdown(f"**题干**：{q['question']}")

                    # 选项显示
                    if q["options"]:
                        options = q["options"]
                        if q["type"] == "选择题":
                            user_ans = st.radio(
                                "选项",
                                options,
                                key=f"q_{i}",
                                index=options.index(q["answer"]) if q["answer"] in options else 0
                            )
                        else:  # 判断题
                            user_ans = st.radio(
                                "判断",
                                options,
                                key=f"q_{i}",
                                index=options.index(q["answer"])
                            )
                        st.session_state.user_answers[i] = user_ans[0] if q["type"] == "选择题" else user_ans
                    else:
                        st.session_state.user_answers[i] = st.text_area(
                            "填写答案",
                            key=f"q_{i}",
                            height=100
                        )

            # 提交和批改
            if st.button("提交试卷"):
                correct_count = 0
                results = []

                for i, q in enumerate(st.session_state.questions):
                    user_ans = st.session_state.user_answers[i]
                    correct = False

                    if q["type"] == "选择题":
                        correct = (user_ans.upper() == q["answer"].upper())
                    elif q["type"] == "判断题":
                        correct = (user_ans == q["answer"])
                    else:
                        # 简答题简单匹配关键词
                        keywords = re.findall(r'\w+', q["answer"].lower())
                        user_words = re.findall(r'\w+', user_ans.lower())
                        correct = len(set(keywords) & set(user_words)) / len(keywords) > 0.6

                    if correct:
                        correct_count += 1

                    results.append({
                        "status": "正确" if correct else "错误",
                        "user_answer": user_ans,
                        "correct_answer": q["answer"],
                        "explanation": q["explanation"]
                    })

                # 显示结果
                st.divider()
                st.subheader(f"测验结果：{correct_count}/{len(results)} 正确")

                for i, result in enumerate(results):
                    with st.expander(f"第 {i + 1} 题解析", expanded=False):
                        st.markdown(f"""
                        - 你的答案：`{result['user_answer']}`
                        - 正确答案：`{result['correct_answer']}`
                        - 解析：{result['explanation']}
                        """)
                        st.markdown(
                            f"**结果**：<span style='color: {'green' if result['status'] == '正确' else 'red'}'>{result['status']}</span>",
                            unsafe_allow_html=True)

if __name__ == "__main__":
    main()