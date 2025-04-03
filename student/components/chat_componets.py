import streamlit as st
from database import Database
from datetime import datetime
import logging
from student.services import ConversationService


class ChatComponent:
    def __init__(self):
        self.student_id = st.session_state.user["student_id"]
        self.conversation_service = ConversationService(self.student_id)

    def display_chat_message(self, role, content):
        with st.chat_message(role):
            if role == "assistant" and any(s in content for s in ["#include", "printf"]):
                st.markdown(f"\n{content}\n```", unsafe_allow_html=True)
            else:
                st.markdown(content, unsafe_allow_html=True)

    def show_chat_interface(self):

        if self.student_id:
            st.session_state.current_conversation["student_id"] = self.student_id
            conversations = self.conversation_service.load_history()

            with st.sidebar:
                # 新建对话按钮
                if st.button("➕ 新建对话"):
                    st.session_state.current_conversation = self.conversation_service.create_conversation()
                    self.conversation_service.clear_memory()
                    st.rerun()

                # 显示历史记录
                st.header("历史对话")
                for conv in conversations:
                    col1, col2 = st.columns([5, 1])
                    btn_text = f"{conv['title']} - {conv['updated_at'].strftime('%m-%d %H:%M')}"
                    with col1:
                        if st.button(btn_text, key=f"hist_{conv['conversation_id']}", help="点击查看该历史对话"):
                            db = Database.get_db()
                            selected_conv = db.conversations.find_one({"conversation_id": conv["conversation_id"]})
                            if selected_conv:
                                st.session_state.current_conversation = selected_conv
                                self.conversation_service.clear_memory()
                                st.rerun()

            # 显示对话内容区
            st.markdown(f'<div class="compact-header">📌 {st.session_state.current_conversation["title"]}</div>', unsafe_allow_html=True)
            for msg in st.session_state.current_conversation["messages"]:
                self.display_chat_message(msg["role"], msg["content"])
            st.markdown('</div>', unsafe_allow_html=True)



            prompt = st.chat_input("输入你的C语言问题...")

            if prompt:
                user_msg = {"role": "user", "content": prompt, "timestamp": datetime.now()}
                st.session_state.current_conversation["messages"].append(user_msg)
                st.session_state.current_conversation["updated_at"] = datetime.now()
                self.conversation_service.save_conversation(st.session_state.current_conversation)
                st.session_state.refresh_sidebar = True

                with st.spinner("🧠 老师思考中..."):
                    try:
                        ai_response = self.conversation_service.get_ai_response(prompt)

                        ai_msg = {"role": "assistant", "content": ai_response, "timestamp": datetime.now()}
                        st.session_state.current_conversation["messages"].append(ai_msg)
                        st.session_state.current_conversation["updated_at"] = datetime.now()

                        try:
                            self.conversation_service.update_coversation_title()
                        except Exception as e:
                            st.error("标题生成失败")
                            logging.error(f"标题生成失败: {str(e)}")
                        try:
                            self.conversation_service.get_knowledge_points(prompt,ai_response)
                        except Exception as e:
                            st.error("知识点提取失败")
                            logging.error(f"知识点提取失败: {str(e)}")

                        try:
                            Database.save_conversation(st.session_state.current_conversation)
                        except Exception as e:
                            st.error("保存失败")
                            logging.error(f"保存失败: {str(e)}")

                        # 保存更新后的对话


                        st.rerun()

                    except Exception as e:
                        st.error("回答生成失败")
                        logging.error(f"生成标题失败1: {str(e)}")