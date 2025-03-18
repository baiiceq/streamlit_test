import streamlit as st
import pandas as pd
import plotly_express as px
from wordcloud import WordCloud

def display_chat_message(role, content):
    with st.chat_message(role):
        if role == "assistant" and any(s in content for s in ["#include", "printf"]):
            st.markdown(f"```c\n{content}\n```")
        else:
            st.markdown(content)

def validate_student_id(student_id):
    return len(student_id) == 8 and student_id.isdigit()

def inject_custom_css():
    st.markdown("""
    <style>
        /* 主内容区增加底部内边距，避免被固定的输入栏遮挡 */
        .main-content {
            padding: 1rem;
            padding-bottom: 10px; /* 根据输入栏高度调整 */
            overflow-y: auto;
        }
        /* 固定在页面底部的输入栏 */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #ffffff;
            padding: 16px 20px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            z-index: 999;
        }
        /* 历史记录容器样式 */
        .history-container {
            max-height: 60vh;
            overflow-y: auto;
            padding-right: 10px;
        }
        .history-container::-webkit-scrollbar {
            width: 8px;
        }
        .history-container::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        .history-container::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        .history-container::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        /* 标题样式 */
        .compact-header {
            margin: 0 0 0.5rem !important;
            padding: 0 !important;
        }
        /* 消息间隔 */
        .stChatMessage {
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def plot_knowledge_timeline(report):
    """绘制知识点出现时间线"""
    text = " ".join(report["knowledges"])
    wordcloud = WordCloud(
        width=800, height=400,
        background_color="white",
        font_path="chinese.simhei.ttf"  # 中文字体路径（根据你的系统调整）
    ).generate(text)

    return wordcloud

def create_activity_heatmap(conversations):
    return 1



