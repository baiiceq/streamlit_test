import streamlit as st


class SidebarComponent:
    @staticmethod
    def show_sidebar():
        page_options = ["💬 即时问答", "📊 学情报告", "📅 我的班级", "📝 题目练习"]

        # 映射字典
        page_mapping = {
            "💬 即时问答": "chat",
            "📊 学情报告": "report",
            "📅 我的班级": "classroom",
            "📝 题目练习": "exercise"
        }

        # 获取当前页面
        current_page = st.session_state.get("student_page", "chat")
        page = st.radio("页面", page_options, index=page_options.index(
            next(key for key, value in page_mapping.items() if value == current_page)))

        # 更新 session_state
        st.session_state["student_page"] = page_mapping[page]
        if(st.session_state["student_page"] == "chat"):
            def toggle_deep_thought():
                st.session_state.deep_thought = not st.session_state.deep_thought

            def toggle_web_search():
                st.session_state.web_search = not st.session_state.web_search

            input_container = st.empty()
            with input_container:
                with st.container():
                    cols = st.columns([2, 2])  # 调整列宽比例

                    with cols[0]:
                        deep_label = ("✅ 深度思考" if st.session_state.deep_thought
                                      else "🔘 深度思考")
                        st.button(
                            deep_label,
                            on_click=toggle_deep_thought,
                            use_container_width=True
                        )

                    with cols[1]:
                        web_label = ("✅ 网络搜索" if st.session_state.web_search
                                     else "🔘 网络搜索")
                        st.button(
                            web_label,
                            on_click=toggle_web_search,
                            use_container_width=True
                        )
        if st.button("退出登录"):
            st.session_state.clear()
            st.session_state.user = None
            st.session_state["current_page"] = "login"
            st.rerun()