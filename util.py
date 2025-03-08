# MongoDB连接信息
MONGO_URI="mongodb+srv://testaccess:123321@wzy.idzma.mongodb.net/?retryWrites=true&w=majority&appName=wzy"

# 数据库名称
DB_NAME='db_test'

# DeepSeek API密钥
DEEPSEEK_API_KEY='sk-719621e0d17c4d39b0aa6a0ad3f21421'

def get_recent_context(conversation):
    summary = conversation.get("summary", "暂无摘要")
    last_five_msgs = conversation["messages"][-5:]
    context = "\n".join([f"{m['role']}: {m['content']}" for m in last_five_msgs])
    return f"【对话摘要】\n{summary}\n\n【近期对话】\n{context}\n\n【学生提问】"


