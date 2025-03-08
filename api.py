import logging
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
import util
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(api_key=util.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_deepseek_response(prompt, temperature = 0.8):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一名计算机系的老师，具有丰富的C语言课程的教学经验，负责解决学生关于C语言课程的问题。回答学生问题时你需要从不同的角度来思考，还需要注意学生可能不知道如何去问的问题，同时要多使用例子和实际代码，并且要考虑学生的拓展性。仅仅代码部分使用markdown语法，其他回复不要使用"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature = temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"API请求失败: {str(e)}")
        st.error("服务暂时不可用，请稍后再试")
        return ""

def generate_conversation_title(prompt):
    title_prompt = f"用不超过6个汉字概括以下内容(最好带有c语言的知识点)：{prompt}"
    try:
        response = get_deepseek_response(title_prompt)
        return response
    except Exception as e:
        logging.error(f"生成标题失败: {str(e)},{prompt}")
        return "新对话"


def generate_conversation_summary(conversation):
    # 将最近对话拼接成文本
    recent_dialogue = "\n".join([f"{m['role']}: {m['content']}" for m in conversation["messages"][-6:]])  # 取最近3轮对话
    # 构造摘要提示词（针对C语言辅导场景优化）
    prompt = f"""你是一名资深的C语言教师，请根据以下对话内容生成结构化摘要(500字以内)：
【对话记录】
{recent_dialogue}
【要求】
1. 识别核心知识点（如：指针、结构体等）
2. 标注常见错误模式（如：内存泄漏、空指针等）
3. 总结学习进展
4. 使用以下格式：
---
【知识点】 
- 知识点1
- 知识点2

【易错点】
- 错误类型1
- 错误类型2

【进展】
- 已掌握...
- 正在学习...
---
"""
    try:
        response = get_deepseek_response(prompt,0.3)
        return response.strip()
    except Exception as e:
        logging.error(f"摘要生成失败: {str(e)}")
        return "暂无摘要"