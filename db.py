from pymongo import MongoClient
import streamlit as st
from datetime import datetime
import uuid
import util  # 假定 util.py 中包含 MONGO_URI、DB_NAME 等常量
from datetime import datetime
import logging

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


def cleanup_old_summaries(max_length=500):
    """自动截断过长的对话摘要"""
    from pymongo import UpdateMany

    try:
        # 1. 批量获取需要清理的对话
        filter_criteria = {"summary": {"$exists": True}}
        projections = {"summary": 1}

        # 2. 使用批量操作提升效率
        update_operations = []
        for conv in db.conversations.find(filter_criteria, projections):
            if len(conv.get("summary", "")) > max_length:
                new_summary = conv["summary"][:max_length] + "..."
                update_operations.append(
                    UpdateMany(
                        {"_id": conv["_id"]},
                        {"$set": {"summary": new_summary}}
                    )
                )

        # 3. 批量执行更新
        if update_operations:
            db.conversations.bulk_write(update_operations)

        return len(update_operations)  # 返回清理数量

    except Exception as e:
        print(f"摘要清理失败: {str(e)}")
        return 0

def load_conversations_by_date(student_id, start_date, end_date):
    """根据时间范围查询对话"""
    try:
        return list(db.conversations.find({
            "student_id": student_id,
            "timestamp": {
                "$gte": datetime.combine(start_date, datetime.min.time()),
                "$lte": datetime.combine(end_date, datetime.max.time())
            }
        }, sort=[("timestamp", 1)]))  # 按时间正序排列
    except Exception as e:
        logging.error(f"时间范围查询失败: {str(e)}")
        return []