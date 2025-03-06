import logging
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
import util
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(api_key=util.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_deepseek_response(prompt):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一名计算机系的老师，具有丰富的C语言课程的教学经验，负责解决学生关于C语言课程的问题。回答学生问题时你需要从不同的角度来思考，还需要注意学生可能不知道如何去问的问题，同时要多使用例子和实际代码，并且要考虑学生的拓展性"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"API请求失败: {str(e)}")
        st.error("服务暂时不可用，请稍后再试")
        raise

def generate_conversation_title(prompt):
    title_prompt = f"用不超过6个汉字概括以下内容(最好带有c语言的知识点)：{prompt}"
    try:
        response = get_deepseek_response(title_prompt)
        return response.strip()[:15] or "新对话"
    except Exception as e:
        logging.error(f"生成标题失败: {str(e)}")
        return "新对话"
