from pymongo import MongoClient
import streamlit as st
from datetime import datetime
import uuid
import util  # 假定 util.py 中包含 MONGO_URI、DB_NAME 等常量

@st.cache_resource
def init_db():
    client = MongoClient(util.MONGO_URI)
    db = client[util.DB_NAME]
    db.conversations.create_index([("student_id", 1), ("timestamp", -1)])
    return db

# 全局数据库实例
db = init_db()

def load_conversation_history(student_id):
    return list(db.conversations.find(
        {"student_id": student_id},
        {"_id": 0, "conversation_id": 1, "title": 1, "timestamp": 1}
    ).sort("timestamp", -1))


def save_conversation(conversation):
    try:
        db.conversations.update_one(
            {"conversation_id": conversation["conversation_id"]},
            {"$set": conversation},
            upsert=True
        )
    except Exception as e:
        import logging, streamlit as st
        logging.error(f"保存失败: {str(e)}")
        st.error("自动保存失败，请及时截图")
