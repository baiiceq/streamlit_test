from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

class Settings:
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/your_database_name")
    DB_NAME: str = os.getenv("DB_NAME", "db_test")
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY")

settings = Settings()