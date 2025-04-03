import uuid
from datetime import datetime
import logging
from database import Database

class Class:
    def __init__(self, class_name, teacher_id, class_id=None, invite_code=None, students=None):
        self.class_id = class_id or str(uuid.uuid4())
        self.class_name = class_name
        self.teacher_id = teacher_id
        self.students = students or []
        self.invite_code = invite_code or self.generate_invite_code()
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def get_student_count(self):
        return len(self.students)

    @staticmethod
    def generate_invite_code():
        """生成6位大写字母数字混合邀请码"""
        return str(uuid.uuid4()).replace("-", "")[:6].upper()

    def to_dict(self):
        """转换为字典格式（用于数据库存储）"""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "teacher_id": self.teacher_id,
            "students": self.students,
            "invite_code": self.invite_code,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    def save(self):
        """保存到数据库"""
        try:
            result = Database.get_db().classes.update_one(
                {"class_id": self.class_id},
                {"$set": self.to_dict()},
                upsert=True
            )
            return result.upserted_id is not None
        except Exception as e:
            logging.error(f"班级保存失败: {str(e)}")
            return False

    def add_student(self, student_id):
        """添加学生到班级"""
        if student_id not in self.students:
            self.students.append(student_id)
            self.updated_at = datetime.now()
            return self.save()
        return False

    def remove_student(self, student_id):
        """从班级移除学生"""
        if student_id in self.students:
            student = Database.find_user({"student_id": student_id})
            self.students.remove(student_id)
            self.updated_at = datetime.now()
            student["classes"].remove(self.class_id)
            return self.save()
        return False

    def refresh_invite_code(self):
        """刷新邀请码"""
        self.invite_code = self.generate_invite_code()
        return self.save()

    @classmethod
    def find_by_id(cls, class_id):
        """通过ID查找班级"""
        data = Database.get_db().classes.find_one({"class_id": class_id})
        if data:
            return cls.from_dict(data)
        return None

    @classmethod
    def find_by_invite_code(cls, invite_code):
        """通过邀请码查找班级"""
        data = Database.get_db().classes.find_one({"invite_code": invite_code})
        if data:
            return cls.from_dict(data)
        return None

    @classmethod
    def find_teacher_classes(cls, teacher_id):
        """获取教师的所有班级"""
        classes = []
        for data in Database.get_db().classes.find({"teacher_id": teacher_id}):
            classes.append(cls.from_dict(data))
        return classes

    @classmethod
    def from_dict(cls, data):
        """从字典创建实例"""
        return cls(
            class_name=data["class_name"],
            teacher_id=data["teacher_id"],
            class_id=data["class_id"],
            invite_code=data["invite_code"],
            students=data["students"]
        )
