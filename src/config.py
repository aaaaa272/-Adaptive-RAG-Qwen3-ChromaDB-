"""
配置管理模块
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# 模型配置
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen3-8b")
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4")

# LangSmith
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "kaoyan-rag")

# ChromaDB
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "kaoyan_knowledge")

# RAG 参数
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "10"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "10"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
DOCS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
