from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
import json
import util


class BaseAgent:
    """Agent基类"""

    def __init__(self, llm=None):
        # 修改点1：配置Deepseek模型参数
        self.llm = llm or ChatOpenAI(
            temperature=0.2,
            model="deepseek-chat",
            openai_api_base="https://api.deepseek.com/v1",
            openai_api_key=util.DEEPSEEK_API_KEY,
            max_retries = 3
        )
        self.chain = None  # 由子类初始化

    def run(self, inputs):
        """执行链式调用"""
        return self.chain(inputs)