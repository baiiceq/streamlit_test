# auth/services.py
import random
import string
from datetime import datetime
from passlib.context import CryptContext
from auth.models import User
import random
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from database import Database

# 创建密码加密的上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthSystem:
    @staticmethod
    def send_verification_code(email: str):
        """发送 6 位随机验证码到用户邮箱"""
        verification_code = str(random.randint(100000, 999999))

        # 存储验证码（可以存入 Redis 或数据库）
        Database.save_verification_code(email, verification_code)

        # 邮件内容
        msg = MIMEText(f"您的验证码是：{verification_code}，有效期 5 分钟。", "plain", "utf-8")
        msg["From"] = formataddr(("注册系统", "2579351233@qq.com"))
        msg["To"] = formataddr((email, email))
        msg["Subject"] = "您的注册验证码"

        # SMTP 服务器配置
        smtp_server = "smtp.qq.com"
        smtp_port = 465
        sender_email = "2579351233@qq.com"
        sender_password = "wapckmguoumndhih"

        try:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [email], msg.as_string())
            server.quit()
            return True, "验证码已发送"
        except Exception as e:
            return False, f"验证码发送失败：{e}"

    @staticmethod
    def verify_code(email: str, verification_code: str):
        """验证验证码"""
        stored_code = Database.get_verification_code(email)  # 从数据库中获取存储的验证码
        print(stored_code,verification_code,email)
        if stored_code == verification_code:
            return True
        return False

    @staticmethod
    def register(db, user_data: dict, verification_code: str):
        """注册新用户"""
        # 校验验证码
        if not AuthSystem.verify_code(user_data["email"], verification_code):
            return False, "验证码错误或已过期"

        # 校验用户名和邮箱是否存在
        if User.find_by_username(db, user_data["username"]):
            return False, "用户名已存在"
        if User.find_by_email(db, user_data["email"]):
            return False, "该邮箱已被注册"
        if user_data["role"] == "student" and User.find_by_student_id(db, user_data.get("student_id")):
            return False, "学号已被注册"

        # 注册用户
        user_data["password_hash"] = pwd_context.hash(user_data["password"])

        Database.create_user(user_data)
        return True, "注册成功，请登录"

    @staticmethod
    def login(db, username: str, password: str):
        """用户登录验证"""
        user = User.find_by_username(db, username)
        if not user:
            return None, "用户名不存在"

        if not pwd_context.verify(password, user["password_hash"]):
            return None, "密码错误"

        return user, "登录成功"


