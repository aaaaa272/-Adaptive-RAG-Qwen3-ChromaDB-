"""
混合检索模块（Adaptive-RAG）
- 学科感知路由：根据问题类型动态调度
- BM25 关键词检索（中文分词）
- 向量检索（Qwen3-Embedding）
- 问题重写（上下文压缩）
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import jieba
from rank_bm25 import BM25Okapi
from openai import OpenAI
import chromadb

from src.config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    QWEN_EMBEDDING_MODEL, QWEN_CHAT_MODEL,
    CHROMA_COLLECTION_NAME,
    TOP_K_VECTOR, TOP_K_BM25
)
from src.indexer import embed_texts, get_chroma_client


@dataclass
class RetrievedDoc:
    """检索结果文档"""
    doc_id: str
    content: str
    source: str
    page: int
    subject: str
    score: float
    retrieval_type: str  # "vector" | "bm25" | "summary_vector"


# ─── 问题重写（上下文压缩 + 语义理解） ────────────────────────────────────────

class QueryRewriter:
    """
    问题重写器
    - 将模糊/口语化的问题转化为结构化检索指令
    - 生成多个子问题（多角度检索）
    """

    def __init__(self):
        self.client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    def rewrite(self, query: str, chat_history: List[Dict] = None) -> Tuple[str, List[str]]:
        """
        返回 (重写后的主问题, 子问题列表)
        """
        history_ctx = ""
        if chat_history:
            recent = chat_history[-3:]  # 最近3轮
            history_ctx = "\n".join([
                f"用户: {h['user']}\n助手: {h['assistant'][:100]}..."
                for h in recent
            ])

        history_section = f"对话历史:\n{history_ctx}" if history_ctx else ""
        prompt = f"""你是考研知识问答系统的查询优化器。

{history_section}

用户原始问题: {query}

请完成以下任务：
1. 将问题重写为更精确的学术查询语句（去除口语化，补全缩写）
2. 如果问题涉及多个知识点，拆分为2-3个子问题

严格按照以下JSON格式输出：
{{
  "rewritten": "重写后的主问题",
  "sub_queries": ["子问题1", "子问题2"]
}}

只输出JSON，不要其他内容。"""

        try:
            resp = self.client.chat.completions.create(
                model=QWEN_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1,
                extra_body={"enable_thinking": False}
            )
            raw = resp.choices[0].message.content.strip()
            # 提取 JSON
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                import json
                data = json.loads(match.group())
                return data.get("rewritten", query), data.get("sub_queries", [])
        except Exception as e:
            logger.warning(f"问题重写失败: {e}")
        return query, []


# ─── 学科路由器（Adaptive-RAG 核心） ─────────────────────────────────────────

class SubjectRouter:
    """
    学科感知路由器
    - 判断问题属于哪个学科
    - 动态选择检索策略
    """

    SUBJECTS = ["数据结构", "操作系统", "计算机网络", "计算机组成原理", "数学", "英语", "政治", "综合"]

    SUBJECT_KEYWORDS = {
        "数据结构": ["链表", "树", "图", "排序", "查找", "栈", "队列", "堆", "hash", "哈希", "复杂度"],
        "操作系统": ["进程", "线程", "调度", "内存", "页面置换", "文件系统", "死锁", "信号量", "互斥"],
        "计算机网络": ["TCP", "UDP", "HTTP", "IP", "路由", "协议", "socket", "握手", "拥塞", "DNS"],
        "计算机组成原理": ["CPU", "指令", "流水线", "cache", "总线", "存储器", "寻址", "中断", "DMA"],
        "数学": ["导数", "积分", "极限", "矩阵", "行列式", "概率", "随机变量", "方差", "特征值"],
        "英语": ["语法", "阅读", "写作", "翻译", "词汇", "作文"],
        "政治": ["马克思", "毛泽东", "习近平", "党史", "哲学", "政治经济学", "时事"],
    }

    def route(self, query: str) -> Tuple[str, str]:
        """
        返回 (subject, retrieval_strategy)
        retrieval_strategy: "hybrid" | "vector_only" | "bm25_focus"
        """
        query_lower = query.lower()
        scores = {}
        for subject, keywords in self.SUBJECT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            if score > 0:
                scores[subject] = score

        if scores:
            subject = max(scores, key=scores.get)
            # 代码/公式类问题，关键词权重更高
            if any(k in query for k in ["代码", "实现", "写出", "算法", "公式"]):
                return subject, "bm25_focus"
            return subject, "hybrid"

        return "综合", "hybrid"


# ─── BM25 检索器 ──────────────────────────────────────────────────────────────

class BM25Retriever:
    """中文 BM25 检索器"""

    def __init__(self, corpus: List[Dict]):
        self.corpus = corpus
        tokenized = [list(jieba.cut(doc["text"])) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 索引构建完成，共 {len(corpus)} 个文档")

    def retrieve(self, query: str, top_k: int = TOP_K_BM25) -> List[RetrievedDoc]:
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self.corpus[idx]
            meta = doc.get("metadata", {})
            results.append(RetrievedDoc(
                doc_id=meta.get("chunk_id", str(idx)),
                content=doc["text"],
                source=meta.get("filename", "unknown"),
                page=meta.get("page", 0),
                subject=meta.get("subject", "综合"),
                score=float(scores[idx]),
                retrieval_type="bm25"
            ))
        return results


# ─── 向量检索器 ───────────────────────────────────────────────────────────────

class VectorRetriever:
    """ChromaDB 向量检索器，支持原始 chunk 和摘要向量双路检索"""

    def __init__(self):
        self.chroma = get_chroma_client()
        self.embed_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        self.raw_col = self.chroma.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        self.summary_col = self.chroma.get_or_create_collection(
            name=f"{CHROMA_COLLECTION_NAME}_summary",
            metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, text: str) -> List[float]:
        return embed_texts([text], self.embed_client)[0]

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_VECTOR,
        subject_filter: Optional[str] = None,
        use_summary: bool = True
    ) -> List[RetrievedDoc]:
        query_emb = self._embed(query)
        where = {"subject": subject_filter} if subject_filter and subject_filter != "综合" else None

        results = []

        # 1. 原始 chunk 向量检索
        kwargs = {"query_embeddings": [query_emb], "n_results": min(top_k, self.raw_col.count() or 1)}
        if where:
            kwargs["where"] = where
        raw_res = self.raw_col.query(include=["documents", "metadatas", "distances"], **kwargs)
        for doc, meta, dist in zip(
            raw_res["documents"][0],
            raw_res["metadatas"][0],
            raw_res["distances"][0]
        ):
            results.append(RetrievedDoc(
                doc_id=meta.get("chunk_id", ""),
                content=doc,
                source=meta.get("filename", "unknown"),
                page=meta.get("page", 0),
                subject=meta.get("subject", "综合"),
                score=1 - dist,  # cosine distance -> similarity
                retrieval_type="vector"
            ))

        # 2. 摘要向量检索（Multi-representation）
        if use_summary and self.summary_col.count() > 0:
            s_kwargs = {"query_embeddings": [query_emb], "n_results": min(top_k // 2, self.summary_col.count())}
            if where:
                s_kwargs["where"] = where
            s_res = self.summary_col.query(include=["documents", "metadatas", "distances"], **s_kwargs)
            for doc, meta, dist in zip(
                s_res["documents"][0],
                s_res["metadatas"][0],
                s_res["distances"][0]
            ):
                results.append(RetrievedDoc(
                    doc_id=meta.get("chunk_id", ""),
                    content=doc,
                    source=meta.get("filename", "unknown"),
                    page=meta.get("page", 0),
                    subject=meta.get("subject", "综合"),
                    score=1 - dist,
                    retrieval_type="summary_vector"
                ))

        return results


# ─── 混合检索器 ───────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Adaptive-RAG 混合检索器
    - 学科感知路由
    - 动态调度 BM25 + 向量检索
    - 支持多子问题并行检索
    """

    def __init__(self, bm25_corpus: Optional[List[Dict]] = None):
        self.query_rewriter = QueryRewriter()
        self.router = SubjectRouter()
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = None
        if bm25_corpus:
            self._init_bm25(bm25_corpus)

    def _init_bm25(self, corpus: List[Dict]):
        self.bm25_retriever = BM25Retriever(corpus)

    def retrieve(
        self,
        query: str,
        chat_history: List[Dict] = None,
        top_k_each: int = TOP_K_VECTOR
    ) -> Tuple[List[RetrievedDoc], Dict]:
        """
        主检索入口
        返回: (检索文档列表, 调试信息)
        """
        debug_info = {}

        # 1. 问题重写
        rewritten, sub_queries = self.query_rewriter.rewrite(query, chat_history)
        debug_info["rewritten_query"] = rewritten
        debug_info["sub_queries"] = sub_queries
        logger.info(f"原始问题: {query}")
        logger.info(f"重写问题: {rewritten}")

        # 2. 学科路由
        subject, strategy = self.router.route(rewritten)
        debug_info["subject"] = subject
        debug_info["strategy"] = strategy
        logger.info(f"学科路由: {subject}, 策略: {strategy}")

        all_docs: List[RetrievedDoc] = []

        # 3. 主问题检索
        queries_to_search = [rewritten] + sub_queries[:2]
        for q in queries_to_search:
            # 向量检索
            v_docs = self.vector_retriever.retrieve(
                q, top_k=top_k_each,
                subject_filter=subject if strategy != "综合" else None,
                use_summary=True
            )
            all_docs.extend(v_docs)

            # BM25 检索
            if self.bm25_retriever and strategy in ("hybrid", "bm25_focus"):
                b_docs = self.bm25_retriever.retrieve(q, top_k=top_k_each)
                all_docs.extend(b_docs)

        debug_info["total_retrieved"] = len(all_docs)
        return all_docs, debug_info
