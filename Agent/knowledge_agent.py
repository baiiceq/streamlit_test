from Agent.base import BaseAgent
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from pathlib import Path
import json


class KnowledgeRAG:
    """本地知识点检索器"""

    def __init__(self, knowledge_path="knowledge/c_language.json"):
        self.knowledge = self._load_knowledge(knowledge_path)
        self.available_concepts = list(self.knowledge.keys())

    def _load_knowledge(self, path):
        """加载本地知识库"""
        try:
            with open(Path(__file__).parent / path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"知识库加载失败: {str(e)}")
            return {}

    def retrieve(self, concepts):
        """检索知识点详情"""
        valid_concepts = []

        # 过滤非字符串类型
        for concept in concepts:
            if isinstance(concept, (str, int, float)):
                valid_concepts.append(str(concept))

        # 返回检索结果
        return {
            k: self.knowledge.get(k, "未知知识点")
            for k in valid_concepts
            if k in self.knowledge
        }

class KnowledgeAgent(BaseAgent):
    description = "用于分析C语言知识点关系的专用工具"

    def __init__(self, llm=None, rag_enabled=True):
        super().__init__(llm)
        self.rag = KnowledgeRAG() if rag_enabled else None

        system_template = """你是一位C语言专家，请完成：
        1. 从对话中识别显性知识点（explicit）
        
        可用知识点列表：{concepts}
        对话记录：{conversations}
        
        返回严格JSON格式（仅包含explicit字段）："""

        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        prompt = ChatPromptTemplate.from_messages([system_prompt])
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def analyze(self, conversation):
        """执行分析"""
        try:
            # 获取可用知识点列表
            concepts = self.rag.available_concepts if self.rag else []

            # 调用LLM
            response = self.run({
                "conversations": conversation,
                "concepts": ", ".join(concepts)
            })

            # 解析结果
            result = self._parse_response(response['text'])

            return result
        except Exception as e:
            return {"error": str(e)}

    def _parse_response(self, text):
        default = {"explicit": []}
        try:
            cleaned = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)

            # 强制类型转换和过滤
            return {
                "explicit": [
                    str(item) for item in data.get("explicit", [])
                    if isinstance(item, (str, int, float))
                ]
            }
        except Exception as e:
            print(f"解析失败: {str(e)}")
            return default