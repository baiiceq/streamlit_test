# auth/models.py
from pydantic import BaseModel, EmailStr
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt

# MongoDB 用户模型
class User(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str
    student_id: Optional[str] = None
    teacher_id: Optional[str] = None

    class Config:
        orm_mode = True

    @classmethod
    def find_by_username(cls, db: AsyncIOMotorClient, username: str):
        return db.users.find_one({"username": username})

    @classmethod
    def find_by_email(cls, db: AsyncIOMotorClient, email: str):
        return db.users.find_one({"email": email})

    @classmethod
    def find_by_student_id(cls, db: AsyncIOMotorClient, student_id: str):
        return db.users.find_one({"student_id": student_id})

    @classmethod
    def find_by_teacher_id(cls, db: AsyncIOMotorClient, teacher_id: str):
        return db.users.find_one({"teacher_id": teacher_id})

    @classmethod
    def create(cls, db: AsyncIOMotorClient, user_data: dict):
        user = cls(**user_data)
        user_dict = user.dict(exclude_unset=True)
        user_dict["password"] = bcrypt.hashpw(user_data["password"].encode('utf-8'), bcrypt.gensalt())
        db.users.insert_one(user_dict)
        return user
