import streamlit as st
from core.page_manager import PageManager
from datetime import datetime
import uuid


def init():
    required_states = {
        "current_conversation": {
            "conversation_id": str(uuid.uuid4()),
            "student_id": "",
            "title": "新对话",
            "messages": [],
            "knowledges": [],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "summary": ""
        },
        "student_page": "chat",
        "teacher_page": "list",
        "current_page": "login",
        "report_generated": False,
        "analysis_content": None,
        "user": None,
        "deep_thought" : False,
        "web_search" : False,
        "report_history" : [],
        "selected_class" : None
    }
    for key, value in required_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    init()
    st.title("📚 智能课业辅导系统")
    page_manager = PageManager()
    page_manager.show_page()

if __name__ == "__main__":
    main()