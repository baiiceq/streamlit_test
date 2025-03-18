import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

FEATURE_SCHEMA = {
    "student_id": str,
    "analysis_time": str,
    "features": {
        "knowledge_points": list,
        "error_patterns": list,
        "learning_preference": {
            "question_type": str,  # 概念型/调试型/项目型
            "depth_level": str     # 基础/进阶
        },
        "interaction_pattern": {
            "question_length": int,
            "follow_up_frequency": float
        }
    }
}