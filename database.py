import mongoengine as me
from core.config import settings  # 用于获取数据库配置
from pymongo import MongoClient
from datetime import datetime
from datetime import timedelta
import uuid
import logging

class Database:
    _db = None

    @staticmethod
    def connect():
        """连接数据库"""
        if Database._db is None:
            # 仅在第一次连接时初始化MongoClient
            client = MongoClient(settings.MONGODB_URI)
            Database._db = client[settings.DB_NAME]

    @staticmethod
    def get_db():
        """获取数据库连接"""
        if Database._db is None:
            # 如果没有客户端实例化，则进行连接
            Database.connect()
        return Database._db

    @staticmethod
    def save_verification_code(email: str, code: str):
        """存储验证码"""
        db = Database.get_db()
        expiration = datetime.utcnow() + timedelta(minutes=5)  # 5 分钟有效期
        db.email_verifications.update_one(
            {"email": email},
            {"$set": {"code": code, "expiration": expiration}},
            upsert=True
        )

    @staticmethod
    def get_verification_code(email: str):
        """获取验证码"""
        db = Database.get_db()
        record = db.email_verifications.find_one({"email": email})

        if not record or datetime.utcnow() > record["expiration"]:
            return None  # 没有找到验证码 或 过期

        return record["code"]  # 返回有效验证码

    @staticmethod
    def find_user(query: dict):
        """查找用户"""
        db = Database.get_db()
        return db.users.find_one(query)

    @staticmethod
    def create_user(user_data: dict):
        """创建用户"""
        db = Database.get_db()
        user_data["user_id"] = str(uuid.uuid4())
        user_data["created_at"] = datetime.now()
        db.users.insert_one(user_data)

    @staticmethod
    def load_conversation_history(student_id):
        db = Database.get_db()
        return list(db.conversations.find(
            {"student_id": student_id},
            {"_id": 0, "conversation_id": 1, "title": 1, "created_at": 1, "updated_at": 1}
        ).sort("updated_at", -1))

    @staticmethod
    def save_conversation(conversation):
        db = Database.get_db()
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

    def add_user_class(user_id, class_id):
        """为用户添加班级关联"""
        db = Database.get_db()
        db.users.update_one(
            {"user_id": user_id},
            {"$addToSet": {"classes": class_id}}
        )

    def load_coversations_by_date(student_id, start_date, end_date):
        """根据时间范围查询对话的知识点"""
        db = Database.get_db()
        try:
            result = list(db.conversations.find({
            "student_id": student_id,
            "$or": [
                {"created_at": {"$gte": datetime.combine(start_date, datetime.min.time()),
                                "$lte": datetime.combine(end_date, datetime.max.time())}},
                {"updated_at": {"$gte": datetime.combine(start_date, datetime.min.time()),
                                "$lte": datetime.combine(end_date, datetime.max.time())}}
            ]
        }).sort("created_at", 1))
            result = [conversation["messages"] for conversation in result]
            messages = []
            for coversation_message in result:
                for message in coversation_message:
                    messages.append(message)
            return messages
        except Exception as e:
            logging.error(f"时间范围查询失败: {str(e)}")
            return []

    def load_knowledge_points_by_date(student_id, start_date, end_date):
        """根据时间范围查询对话的知识点"""
        db = Database.get_db()
        try:
            result = list(db.conversations.find({
            "student_id": student_id,
            "$or": [
                {"created_at": {"$gte": datetime.combine(start_date, datetime.min.time()),
                                "$lte": datetime.combine(end_date, datetime.max.time())}},
                {"updated_at": {"$gte": datetime.combine(start_date, datetime.min.time()),
                                "$lte": datetime.combine(end_date, datetime.max.time())}}
            ]
        }).sort("created_at", 1))
            result = [conversation["knowledges"] for conversation in result]
            knowledge_points = []
            for coversation_points in result:
                for points in coversation_points:
                    knowledge_points.append(points)
            return knowledge_points
        except Exception as e:
            logging.error(f"时间范围查询失败: {str(e)}")
            return []