import streamlit as st
import uuid
from datetime import datetime
import logging
import json
from wordcloud import WordCloud
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 导入自定义模块
import config  # 加载环境变量和日志配置
import db  # 数据库操作
import api  # API调用及标题生成
import ui  # UI 辅助函数
import util
import analysis



def main():
    st.title("📚 智能课业辅导系统")
    ui.inject_custom_css()  # 注入更新后的 CSS 样式

    # 状态初始化
    required_states = {
        "current_conversation": {
            "conversation_id": str(uuid.uuid4()),
            "student_id": "",
            "title": "新对话",
            "messages": [],
            "timestamp": datetime.now(),
            "summary": ""
        },
        "selected_page": "chat",
        "report_generated": False,
        "analysis_content": None
    }
    for key, value in required_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

    with st.sidebar:
        # 页面切换
        page_options = ["💬 即时问答", "📊 学情报告"]
        page = st.radio("页面", page_options, index=0 if st.session_state.selected_page == "chat" else 1)
        st.session_state.selected_page = "chat" if page == page_options[0] else "report"

        # 学号输入与验证
        st.header("历史对话")
        student_id = st.text_input("学号（8位数字）", key="student_id")
        if student_id and not ui.validate_student_id(student_id):
            st.error("学号格式错误")
            st.stop()

        # 加载历史对话
        if student_id:
            st.session_state.current_conversation["student_id"] = student_id
            conversations = db.load_conversation_history(student_id)

            # 新建对话按钮
            if st.button("➕ 新建对话"):
                st.session_state.current_conversation = {
                    "conversation_id": str(uuid.uuid4()),
                    "student_id": student_id,
                    "title": "新对话",
                    "messages": [],
                    "timestamp": datetime.now(),
                    "summary": ""
                }
                st.rerun()

            # 显示历史记录
            st.subheader("历史记录")
            st.markdown('<div class="history-container">', unsafe_allow_html=True)
            for conv in conversations:
                col1, col2 = st.columns([5, 1])
                btn_text = f"{conv['title']} - {conv['timestamp'].strftime('%m-%d %H:%M')}"

                with col1:
                    if st.button(btn_text, key=f"hist_{conv['conversation_id']}", help="点击查看该历史对话"):
                        selected_conv = db.db.conversations.find_one({"conversation_id": conv["conversation_id"]})
                        if selected_conv:
                            st.session_state.current_conversation = selected_conv
                            st.rerun()

                with col2:
                    if conv.get("summary"):
                        st.markdown("ℹ️", help=conv["summary"])

            st.markdown('</div>', unsafe_allow_html=True)

    # 主界面：分为聊天内容和底部固定的输入栏
    try:
        if st.session_state.selected_page == "chat":
            st.markdown(f'<div class="compact-header">📌 {st.session_state.current_conversation["title"]}</div>',
                        unsafe_allow_html=True)
            # 聊天内容区：滚动显示所有消息
            st.markdown('<div class="main-content">', unsafe_allow_html=True)
            for msg in st.session_state.current_conversation["messages"]:
                ui.display_chat_message(msg["role"], msg["content"])
            st.markdown('</div>', unsafe_allow_html=True)

            prompt = st.chat_input("输入你的C语言问题...")

            if prompt:
                user_msg = {"role": "user", "content": prompt}
                st.session_state.current_conversation["messages"].append(user_msg)
                st.session_state.current_conversation["timestamp"] = datetime.now()
                db.save_conversation(st.session_state.current_conversation)
                st.session_state.refresh_sidebar = True

                with st.spinner("🧠 老师思考中..."):
                    try:
                        full_prompt = util.get_recent_context(st.session_state.current_conversation) + prompt
                        ai_response = api.get_deepseek_response(full_prompt)
                        ai_msg = {"role": "assistant", "content": ai_response}
                        st.session_state.current_conversation["messages"].append(ai_msg)

                        # 更新标题和摘要
                        current_conv = st.session_state.current_conversation
                        if len(st.session_state.current_conversation["messages"]) <= 1:
                            new_title = api.generate_conversation_title(prompt + ai_response)
                            st.session_state.current_conversation["title"] = new_title
                            summary = api.generate_conversation_summary(current_conv )
                            current_conv["summary"] = summary
                        elif len(st.session_state.current_conversation["messages"]) % 3 == 0:
                            last_msgs = " ".join(
                                [m["content"] for m in st.session_state.current_conversation["messages"][-6:]])
                            new_title = api.generate_conversation_title(last_msgs)
                            st.session_state.current_conversation["title"] = new_title
                            summary = api.generate_conversation_summary(current_conv)
                            current_conv["summary"] = summary

                        st.session_state.current_conversation["timestamp"] = datetime.now()

                        db.save_conversation(current_conv)
                        st.rerun()
                    except Exception as e:
                        st.error("回答生成失败")
                        logging.error(f"生成标题失败1: {str(e)}")
        else:
            # 学情报告页面的代码逻辑
            st.header("📊 学习分析报告")

            # 新增时间选择行
            time_col1, time_col2, time_col3 = st.columns([2, 2, 3])
            with time_col1:
                start_date = st.date_input("起始日期",
                                           value=datetime.now(),
                                           max_value=datetime.now())
            with time_col2:
                end_date = st.date_input("结束日期",
                                         value=datetime.now(),
                                         min_value=start_date)
            with time_col3:
                st.write("")  # 占位对齐
                analyze_btn = st.button("🚀 生成时段报告",
                                        help="分析选定时间段内的所有对话",
                                        disabled=not st.session_state.current_conversation["student_id"])

            # 错误处理
            if start_date > end_date:
                st.error("错误：结束日期不能早于开始日期")
                st.stop()

            # 核心分析逻辑
            if analyze_btn:
                with st.spinner(
                        "正在分析{}至{}的对话记录...".format(start_date.strftime("%m/%d"), end_date.strftime("%m/%d"))):
                    try:
                        # 获取时间范围内的对话
                        conversations = db.load_conversations_by_date(
                            st.session_state.current_conversation["student_id"],
                            start_date,
                            end_date
                        )

                        if not conversations:
                            st.warning("该时间段内没有可分析的对话记录")
                            st.stop()

                        report = analysis.get_student_report(conversations)

                        st.session_state.analysis_content = report
                        st.session_state.report_generated = True
                    except Exception as e:
                        st.error("分析失败：" + str(e))

            # 展示分析结果
            if st.session_state.report_generated:
                st.session_state.report_generated = False
                # 报告概览卡片
                with st.container(border=True):
                    cols = st.columns([2,1,1,1])
                    cols[0].metric("分析时段",
                                   f"{start_date.strftime('%m/%d')} - {end_date.strftime('%m/%d')}")
                    cols[1].metric("涉及对话数",
                                   st.session_state.analysis_content["conversations_cnt"])
                    cols[2].metric("总消息量",
                                   st.session_state.analysis_content["messages_cnt"])
                    cols[3].metric("核心知识点",
                                   len(st.session_state.analysis_content["knowledges"]))

                # 交互式分析面板
                tab1, tab2, tab3 = st.tabs(["📚 知识点分析", "❗ 错误模式", "📈 学习趋势"])

                with tab1:
                    word_cloud = ui.plot_knowledge_timeline(st.session_state.analysis_content)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(word_cloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                    

                # 下载功能增强
                report_json = json.dumps(st.session_state.analysis_content, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 下载完整分析报告 (JSON)",
                    data=report_json,
                    file_name=f"learning_report_{start_date}_{end_date}.json",
                    mime="application/json",
                    key="full_report_download"
                )
    except Exception as e:
            st.error("系统发生错误，请刷新页面: " + str(e))
            logging.exception("系统异常")

if __name__ == "__main__":
    main()


