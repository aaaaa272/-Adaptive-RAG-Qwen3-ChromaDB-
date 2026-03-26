"""
离线索引建立模块
- 支持 PDF / TXT / Markdown 文档
- 使用 Qwen3-Embedding + Multi-representation 技术
- 层级索引：原始 chunk 存向量，同时存 summary 表征
- 分布式存储于 ChromaDB
"""
import os
import json
import hashlib
import base64
from pathlib import Path
from typing import List, Optional
from loguru import logger
import jieba
import fitz  # PyMuPDF

from openai import OpenAI
import chromadb
from chromadb.config import Settings

from src.config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    QWEN_EMBEDDING_MODEL, QWEN_CHAT_MODEL,
    CHROMA_DB_PATH, CHROMA_COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR
)


def get_embedding_client() -> OpenAI:
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False)
    )


def embed_texts(texts: List[str], client: Optional[OpenAI] = None) -> List[List[float]]:
    """批量文本向量化（阿里云百炼 Embedding）"""
    if client is None:
        client = get_embedding_client()
    embeddings = []
    batch_size = 25  # 百炼 API 单次最多25条
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            model=QWEN_EMBEDDING_MODEL,
            input=batch,
            encoding_format="float"
        )
        embeddings.extend([item.embedding for item in resp.data])
    return embeddings


# ─── 文本分块 ────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """按字符窗口分块，中文友好"""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ─── 文档加载 ────────────────────────────────────────────────────────────────

def _ocr_page_with_qwen(page: fitz.Page) -> str:
    """将 PDF 页面渲染为图片，调用 qwen-vl-ocr 识别文字"""
    try:
        mat = fitz.Matrix(2.0, 2.0)  # 2x 分辨率，提升识别率
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        resp = client.chat.completions.create(
            model="qwen-vl-ocr",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            "min_pixels": 28 * 28 * 4,
                            "max_pixels": 1280 * 784,
                        },
                        {"type": "text", "text": "请提取图片中所有文字内容，保持原有段落结构，不要添加任何解释。"},
                    ],
                }
            ],
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OCR 识别失败: {e}")
        return ""


def load_pdf(path: str) -> List[dict]:
    """加载 PDF，返回 [{page: int, text: str}]
    自动检测扫描件，调用 qwen-vl-ocr 识别
    """
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            # 扫描件：用 OCR 识别
            logger.info(f"第 {i+1} 页为扫描件，调用 OCR 识别...")
            text = _ocr_page_with_qwen(page)
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def load_txt(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return [{"page": 1, "text": text}] if text else []


def load_document(path: str) -> List[dict]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".txt", ".md"):
        return load_txt(path)
    else:
        logger.warning(f"不支持的文件类型: {ext}，跳过 {path}")
        return []


# ─── 摘要生成（Multi-representation） ────────────────────────────────────────

def generate_chunk_summary(chunk: str, client: Optional[OpenAI] = None) -> str:
    """
    为每个 chunk 生成摘要表征（用于 Multi-representation 索引）
    用 Qwen3 生成语义更浓缩的摘要，提升检索精度
    """
    if client is None:
        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    try:
        resp = client.chat.completions.create(
            model=QWEN_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是考研知识整理助手。请用1-2句话概括以下文本的核心知识点，便于检索。"
                },
                {"role": "user", "content": chunk}
            ],
            max_tokens=128,
            temperature=0.1,
            extra_body={"enable_thinking": False}
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"摘要生成失败: {e}")
        return chunk[:200]  # fallback


# ─── 主索引函数 ───────────────────────────────────────────────────────────────

class KnowledgeIndexer:
    """知识库索引器"""

    def __init__(self):
        self.embed_client = get_embedding_client()
        self.llm_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        self.chroma = get_chroma_client()

        # 原始 chunk 集合（用于检索返回）
        self.raw_collection = self.chroma.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        # 摘要向量集合（Multi-representation：用摘要的向量检索原始文本）
        self.summary_collection = self.chroma.get_or_create_collection(
            name=f"{CHROMA_COLLECTION_NAME}_summary",
            metadata={"hnsw:space": "cosine"}
        )

    def _doc_id(self, filepath: str, chunk_idx: int) -> str:
        h = hashlib.md5(f"{filepath}:{chunk_idx}".encode()).hexdigest()[:8]
        return f"{Path(filepath).stem}_{chunk_idx}_{h}"

    def index_file(self, filepath: str, use_summary: bool = True, progress_cb=None):
        """对单个文件建立索引"""
        logger.info(f"索引文件: {filepath}")
        pages = load_document(filepath)
        if not pages:
            logger.warning(f"文件内容为空: {filepath}")
            return 0

        all_chunks = []
        all_metas = []

        for page_info in pages:
            chunks = chunk_text(page_info["text"])
            for idx, chunk in enumerate(chunks):
                chunk_id = self._doc_id(filepath, len(all_chunks))
                all_chunks.append(chunk)
                all_metas.append({
                    "source": str(filepath),
                    "filename": Path(filepath).name,
                    "page": page_info["page"],
                    "chunk_index": idx,
                    "chunk_id": chunk_id,
                    "subject": self._infer_subject(Path(filepath).name),
                })

        if not all_chunks:
            return 0

        logger.info(f"共 {len(all_chunks)} 个 chunk，开始向量化...")

        # 1. 原始 chunk 向量化 & 存储
        raw_embeddings = embed_texts(all_chunks, self.embed_client)
        ids = [m["chunk_id"] for m in all_metas]

        # 过滤已存在的 ID
        existing = set(self.raw_collection.get(ids=ids)["ids"])
        new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing]

        if new_indices:
            self.raw_collection.upsert(
                ids=[ids[i] for i in new_indices],
                embeddings=[raw_embeddings[i] for i in new_indices],
                documents=[all_chunks[i] for i in new_indices],
                metadatas=[all_metas[i] for i in new_indices],
            )

        # 2. 摘要向量化（Multi-representation）
        if use_summary:
            logger.info("生成摘要表征（Multi-representation）...")
            summaries = []
            for i, chunk in enumerate(all_chunks):
                if progress_cb:
                    progress_cb(i + 1, len(all_chunks))
                summaries.append(generate_chunk_summary(chunk, self.llm_client))

            summary_embeddings = embed_texts(summaries, self.embed_client)
            summary_ids = [f"summary_{id_}" for id_ in ids]

            existing_s = set(self.summary_collection.get(ids=summary_ids)["ids"])
            new_s_indices = [i for i, id_ in enumerate(summary_ids) if id_ not in existing_s]

            if new_s_indices:
                # 摘要集合存储：向量是摘要的，document 是原始 chunk（检索后返回原文）
                s_metas = []
                for i in new_s_indices:
                    m = dict(all_metas[i])
                    m["summary"] = summaries[i]
                    m["raw_chunk_id"] = ids[i]
                    s_metas.append(m)

                self.summary_collection.upsert(
                    ids=[summary_ids[i] for i in new_s_indices],
                    embeddings=[summary_embeddings[i] for i in new_s_indices],
                    documents=[all_chunks[i] for i in new_s_indices],  # 存原始文本
                    metadatas=s_metas,
                )

        logger.info(f"文件 {Path(filepath).name} 索引完成，共 {len(all_chunks)} 个 chunk")
        return len(all_chunks)

    def index_directory(self, directory: str = None, use_summary: bool = True, progress_cb=None):
        """批量索引目录下所有文档"""
        if directory is None:
            directory = str(DOCS_DIR)
        doc_dir = Path(directory)
        files = list(doc_dir.glob("**/*.pdf")) + \
                list(doc_dir.glob("**/*.txt")) + \
                list(doc_dir.glob("**/*.md"))

        if not files:
            logger.warning(f"目录 {directory} 中没有找到文档")
            return 0

        total = 0
        for f in files:
            n = self.index_file(str(f), use_summary=use_summary, progress_cb=progress_cb)
            total += n
        logger.info(f"全部索引完成，共 {total} 个 chunk")
        return total

    def _infer_subject(self, filename: str) -> str:
        """根据文件名推断学科"""
        fn = filename.lower()
        if any(k in fn for k in ["数据结构", "data_structure", "ds"]):
            return "数据结构"
        elif any(k in fn for k in ["操作系统", "os", "operating"]):
            return "操作系统"
        elif any(k in fn for k in ["计算机网络", "network", "net"]):
            return "计算机网络"
        elif any(k in fn for k in ["组成原理", "architecture", "计组"]):
            return "计算机组成原理"
        elif any(k in fn for k in ["数学", "math", "高数", "线代", "概率"]):
            return "数学"
        elif any(k in fn for k in ["英语", "english"]):
            return "英语"
        elif any(k in fn for k in ["政治", "politics", "马原"]):
            return "政治"
        return "综合"

    def get_collection_stats(self) -> dict:
        return {
            "raw_chunks": self.raw_collection.count(),
            "summary_chunks": self.summary_collection.count(),
        }

    def get_bm25_corpus(self) -> List[dict]:
        """获取所有 chunk 用于 BM25 索引构建"""
        results = self.raw_collection.get(include=["documents", "metadatas"])
        corpus = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            corpus.append({"text": doc, "metadata": meta})
        return corpus
