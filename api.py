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
                {"role": "system", "content": '''你是一名计算机系的老师，具有丰富的C语言课程的教学经验以及深厚的理论基础和丰富的实际经验。负责解决学生关于C语言课程的问题，并总是能够提供准确，严谨同时易懂的的答案。接下来是对你回答内容的一些规范和要求
【回答要求】
1.站在一位学生的角度来思考该问题，分析出学生在该问题上可能遇到的难点和易错点，在回答时从多角度来来思考该问题的答案和涉及知识点。
2.考虑到答案的丰富度，希望你在回答和解释问题时，多使用恰当的示例和代码，对于代码部分，使用markdown来表示以显示区别。
3.逐步展现对问题的思考，并展示思考过程，使得学生能够学习到解决该问题的思路，具体思路可以参照如下描述，也可以适当添加和丰富一些内容：（1）如何阅读和分析题目，将复杂的问题拆分成一个个简单的任务。（2）如何从问题中发掘关键词，（3）如何通过关键词对应到学科知识点，发掘该问题的本质是在考察什么知识点，（4）如何调用这些知识点来一步步解决问题（5）回溯整个思考过程检查是否存在漏洞或者考虑步骤，并复盘整个思考流程，并用流程图的形式显示出来整个思考过程。我希望可以显示上述思考的内容和过程，并以 thinking 思考内容 thinking的格式显示在问题回答的最上面。
5.结合该问题的特点和涉及知识点，为该学生在提出一些相关或者更深入的问题。'''},
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
