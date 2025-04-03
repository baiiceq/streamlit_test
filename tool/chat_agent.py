import os
import requests
import json
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain.tools import StructuredTool
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType


class ConversationManager:
    def __init__(self, model="qwen-max", temperature=0.5):
        """
        初始化对话管理器，包括模型和对话内存。

        Args:
            model (str): 使用的语言模型名称
            temperature (float): 模型的随机性参数
        """
        # 初始化对话内存
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 初始化语言模型
        self.qwen = Tongyi(
            model=model,
            temperature=temperature,
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY")
        )

        # 初始化思维链和搜索工具
        self.chain_cot = self.init_cot_chain()
        self.search_tool, self.summary_chain, self.agent, self.agent_prompt = self.init_search_tool()

    def clear_memory(self):
        """
        清除对话历史记忆，确保新对话不受之前对话影响。
        """
        self.conversation_memory.clear()

    def init_cot_chain(self):
        """
        初始化用于生成思维链的链条，分步思考问题。

        Returns:
            LLMChain: 已配置的思维链链条。
        """
        cot_template = '''
    你是一个逻辑推理专家，负责将问题进行逐步思考并生成推理过程。
    请一步一步思考题目: {question}，生成详细的推理步骤(COT)
    要求：
    - 分步骤解释核心步骤 
    - 尽可能详细分析问题各种可能情况
    - 将问题分解成一个个小任务，体现思维链特点
    - 对于不清楚的知识或概念不要随便假设
    我不需要问题答案，只需要你思考问题的过程。
        '''
        prompt = PromptTemplate(input_variables=["question"], template=cot_template)

        llm_cot = Tongyi(
            model="qwen-max",
            temperature=0.8,
            openai_api_key='sk-e2492dea19b945059a9a05abb4d0fc0b',
            openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )
        chain_cot = LLMChain(llm=llm_cot, prompt=prompt)
        return chain_cot

    def init_search_tool(self):
        """
        初始化网络搜索工具、摘要链及 Agent，用于执行网络搜索和生成摘要。

        Returns:
            tuple: (search_tool, summary_chain, agent, agent_prompt)
        """

        def bocha_websearch_tool(query: str, count: int = 20) -> str:
            """
            通过 Bocha API 执行网络搜索。

            Args:
                query (str): 搜索关键词。
                count (int): 返回结果数量（默认20）。

            Returns:
                str: 格式化后的搜索结果摘要。
            """
            url = 'https://api.bochaai.com/v1/web-search'
            headers = {
                'Authorization': f'Bearer sk-782dce9336b549299c48f4607376d0f2',  # 替换为实际 API 密钥
                'Content-Type': 'application/json'
            }
            data = {
                "query": query,
                "freshness": "noLimit",
                "summary": True,
                "count": count
            }
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                json_data = response.json()
                if json_data.get("code") != 200 or not json_data.get("data"):
                    return "搜索服务暂时不可用"
                webpages = json_data["data"]["webPages"]["value"]
                return "\n\n".join(
                    f"标题：{p['name']}\n链接：{p['url']}\n摘要：{p['summary']}"
                    for p in webpages[:5]
                )
            except Exception as e:
                return f"搜索失败：{str(e)}"

        search_tool = StructuredTool.from_function(
            func=bocha_websearch_tool,
            name="web_search",
            description="""
    用于执行网络搜索的工具。
    输入应为包含以下键的 JSON 对象：
    - query: 搜索关键词（例如 "人字路口 真话 假话"）
    - count: 可选，结果数量（默认20）
            """
        )

        summary_prompt = PromptTemplate(
            input_variables=["question", "web_results"],
            template="""你是一个擅长总结网络信息的信息分析师，请结合下列搜索结果生成准确、清晰的摘要。
    问题：{question}
    搜索结果：
    {web_results}
    请提炼关键信息供参考。"""
        )
        summary_chain = LLMChain(llm=self.qwen, prompt=summary_prompt)

        agent_prompt_template = """你是一个智能搜索助手，请按以下步骤处理问题：
    1. 分析问题并提取3-5个核心关键词；
    2. 使用 web_search 工具进行搜索；
    3. 综合搜索结果生成结构化报告。
    当前问题：
    {question}
    请以 JSON 格式返回如下字段：
    {{
      "keywords": ["关键词1", "关键词2"],
      "tool_input": "要搜索的关键词组合"
    }}"""
        agent_prompt = PromptTemplate(input_variables=['question'], template=agent_prompt_template)

        agent = initialize_agent(
            tools=[search_tool],
            llm=self.qwen,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

        return search_tool, summary_chain, agent, agent_prompt

    def get_web_search_summary(self, input_text):
        """
        根据用户问题调用网络搜索工具，并生成搜索摘要。

        Args:
            input_text (str): 用户问题。

        Returns:
            str: 生成的网络搜索摘要文本。
        """
        formatted_input = self.agent_prompt.format(question=input_text)
        search_response = self.agent.invoke({"input": formatted_input})
        summary_result = self.summary_chain.invoke({
            'question': input_text,
            'web_results': search_response['output']
        })
        return summary_result['text']

    def process_conversation(self, input_text, use_web_search=True):
        """
        处理对话，支持记忆管理和可选的网络搜索。

        Args:
            input_text (str): 用户输入文本
            use_web_search (bool): 是否使用网络搜索

        Returns:
            tuple: (最终回答, 网络搜索摘要, 思维链文本)
        """
        # 调用思维链生成思维过程
        cot_result = self.chain_cot.invoke({"question": input_text})
        thought_text = cot_result["text"] if "text" in cot_result else cot_result

        # 根据标志判断是否调用网络搜索生成摘要
        web_summary = self.get_web_search_summary(input_text) if use_web_search else ""

        # 将问题、思维链和网络摘要合并成单一输入变量
        combined_input = f"### 问题:\n{input_text}\n\n### 思维链 / 推理:\n{thought_text}\n\n### 网络信息:\n{web_summary}"

        # 创建最终答案生成链，采用单一输入变量，并注入对话历史
        final_prompt = PromptTemplate(
            input_variables=['chat_history', 'combined_input'],
            template='''
{chat_history}
你是一位优秀的C语言教授，擅长从学生角度讲解编程问题。请阅读下面整合后的内容，并生成详细解答：

{combined_input}

回答要求:
1. 详细讲解问题解决过程，条理清晰；
2. 针对复杂问题提供示例和代码说明；
3. 分析学生可能遇到的难点及易错点；
4. 根据问题特点提出相关拓展问题。
'''
        )
        chain_result = LLMChain(llm=self.qwen, prompt=final_prompt, memory=self.conversation_memory)
        response = chain_result.invoke({
            'chat_history': self.conversation_memory.buffer,
            'combined_input': combined_input
        })
        self.conversation_memory.save_context({"input": input_text}, {"output": response['text']})
        return response['text'], web_summary, thought_text

    def thinking_input(self, input_text, use_web_search=True):
        """
        通过 DeepSeek 深度推理，并结合（可选）网络搜索摘要和对话记忆生成答案。

        Args:
            input_text (str): 用户提问。
            use_web_search (bool): 是否调用网络搜索（默认 True）。

        Returns:
            tuple: (最终回答, 网络搜索摘要, DeepSeek 推理文本)
        """
        # 调用 DeepSeek 获取深度推理结果
        client = OpenAI(api_key="sk-251abe6e76f64f79a2321611c6f67bc6", base_url="https://api.deepseek.com")
        messages = [{"role": "user", "content": input_text}]
        deepseek_response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
        )
        reasoning_content = deepseek_response.choices[0].message.reasoning_content

        # 根据标志判断是否调用网络搜索生成摘要
        web_summary = self.get_web_search_summary(input_text) if use_web_search else ""

        # 合并 DeepSeek 推理结果、问题和网络摘要为单一输入变量
        combined_input = f"### 问题:\n{input_text}\n\n### 深度推理:\n{reasoning_content}\n\n### 网络信息:\n{web_summary}"

        # 创建最终答案生成链，注入对话历史
        final_prompt = PromptTemplate(
            input_variables=['chat_history', 'combined_input'],
            template='''
{chat_history}
你是一位优秀的C语言教授，擅长从学生角度讲解编程问题。请阅读下面整合后的内容，并生成详细解答：

{combined_input}

回答要求:
1. 详细讲解问题解决过程，条理清晰；
2. 针对复杂问题提供示例和代码说明；
3. 分析学生可能遇到的难点及易错点；
4. 根据问题特点提出相关拓展问题。
'''
        )
        chain_result = LLMChain(llm=self.qwen, prompt=final_prompt, memory=self.conversation_memory)
        response = chain_result.invoke({
            'chat_history': self.conversation_memory.buffer,
            'combined_input': combined_input
        })
        self.conversation_memory.save_context({"input": input_text}, {"output": response['text']})
        return response['text'], web_summary, reasoning_content

    def create_mind_map(self, input_text: str):
        """
        调用外部 API 生成思维导图。

        Args:
            input_text (str): 用于生成思维导图的输入文本。

        Returns:
            dict: API 返回的思维导图数据。格式为{"output"："url"}
        """
        API_URL = 'https://api.coze.cn/v1/workflow/run'
        API_KEY = "pat_lFeh61WZGRkTegrzgDe7VmMhpOZFElKT5tJwlMsdRiQJywMXUplHMLe6W65E9KEL"
        WORKFLOW_ID = "7482050366412095539"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "workflow_id": WORKFLOW_ID,
            'parameters': {'input_2': input_text},
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()['data']

    def get_direct_response(self, input_text: str):
        """
        直接将用户的输入传递给 Tongyi 大模型，返回生成的回答。

        Args:
            input_text (str): 用户输入的提示词。

        Returns:
            str: 大模型生成的回答。
        """
        response = self.qwen.generate([input_text])
        return response.generations[0][0].text  # 返回生成的文本内容

#-------------------------------
#示例代码
#-------------------------------
def main():
    # 创建对话管理器
    conversation_manager = ConversationManager()

    # 第一个对话
    question1 = "什么是动态规划？"
    response1, web_summary1, thought_text1 = conversation_manager.process_conversation(
        question1, use_web_search=True
    )
    print("第一个对话响应：", response1)

    # 第二个对话（相同问题）
    question2 = "重复上一个问题"
    response2, web_summary2, thought_text2 = conversation_manager.process_conversation(
        question2, use_web_search=False
    )
    print("第二个对话响应：", response2)

    conversation_manager_2=ConversationManager()

    # 第三个对话（再次相同问题）
    question3 = "重复上一个问题"
    response3, web_summary3, thought_text3 = conversation_manager_2.process_conversation(
        question3, use_web_search=False
    )
    print("第三个对话响应：", response3)

    # 使用深度推理方法
    question4 = "深度学习的基本原理是什么？"
    response4, web_summary4, reasoning_text4 = conversation_manager.thinking_input(
        question4, use_web_search=True
    )
    print("深度推理对话响应：", response4)

    # 生成思维导图
    mind_map_result = conversation_manager.create_mind_map(question4)
    print("思维导图链接：", mind_map_result)


if __name__ == "__main__":
    main()