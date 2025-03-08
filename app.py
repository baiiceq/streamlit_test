import streamlit as st
import uuid
from datetime import datetime
import logging

# 导入自定义模块
import config  # 加载环境变量和日志配置
import db  # 数据库操作
import api  # API调用及标题生成
import ui  # UI 辅助函数
import util


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
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("生成报告", disabled=st.session_state.report_generated):
                    with st.spinner("分析中..."):
                        try:
                            history = "\n".join([f"{m['role']}: {m['content']}" for m in
                                                 st.session_state.current_conversation["messages"]])
                            if st.session_state.current_conversation.get("summary"):
                                history = f"【对话摘要】\n{st.session_state.current_conversation['summary']}\n\n【完整对话】\n" + history
                            prompt = f"""根据以下对话生成学习报告：
{history}

要求：
- 分[掌握知识点][常见错误][学习建议]三部分
- 每个部分3-5个条目
- 使用Markdown格式"""
                            st.session_state.analysis_content = api.get_deepseek_response(prompt)
                            st.session_state.report_generated = True
                            st.rerun()
                        except Exception as e:
                            st.error("报告生成失败")
            if st.session_state.report_generated:
                st.markdown(st.session_state.analysis_content)
            else:
                st.info("点击上方按钮生成学习报告")
    except Exception as e:
        st.error("系统发生错误，请刷新页面")
        logging.exception("系统异常")


if __name__ == "__main__":
    main()


