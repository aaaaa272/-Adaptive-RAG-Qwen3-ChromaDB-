"""
Agentic RAG 主流水线
整合：问题重写 → 学科路由 → 混合检索 → Self-RAG → RRF → 重排序 → CoT 生成
"""
from typing import List, Dict, Iterator, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from src.config import TOP_K_FINAL
from src.indexer import KnowledgeIndexer
from src.retriever import HybridRetriever, RetrievedDoc
from src.reranker import PostProcessor, reciprocal_rank_fusion
from src.generator import AnswerGenerator


@dataclass
class RAGResponse:
    """RAG 完整响应"""
    answer: str = ""
    sources: List[Dict] = field(default_factory=list)
    debug_info: Dict = field(default_factory=dict)
    rewritten_query: str = ""
    subject: str = ""
    retrieval_count: int = 0


class AgenticRAGPipeline:
    """
    Agentic RAG 完整流水线

    流程：
    1. 问题重写（上下文压缩 + 语义理解）
    2. Adaptive-RAG 路由（学科感知）
    3. 混合检索（BM25 + 向量检索 + 摘要向量）
    4. Self-RAG 多轮评估（必要时补充检索）
    5. RRF 融合去重
    6. Cohere/LLM 重排序
    7. CoT 生成 + Citation 机制
    """

    def __init__(self, use_self_rag: bool = True):
        logger.info("初始化 Agentic RAG 流水线...")
        self.indexer = KnowledgeIndexer()
        self.retriever: Optional[HybridRetriever] = None
        self.post_processor = PostProcessor(use_self_rag=use_self_rag)
        self.generator = AnswerGenerator()
        self.use_self_rag = use_self_rag
        self._initialized = False
        logger.info("RAG 流水线初始化完成")

    def initialize_retriever(self):
        """初始化检索器（需要在索引建立后调用）"""
        bm25_corpus = self.indexer.get_bm25_corpus()
        self.retriever = HybridRetriever(bm25_corpus=bm25_corpus if bm25_corpus else None)
        self._initialized = True
        logger.info(f"检索器初始化完成，BM25 语料: {len(bm25_corpus)} 条")

    def is_ready(self) -> bool:
        """检查是否可以回答问题"""
        stats = self.indexer.get_collection_stats()
        return stats["raw_chunks"] > 0

    def get_stats(self) -> Dict:
        """获取系统状态"""
        stats = self.indexer.get_collection_stats()
        stats["retriever_ready"] = self._initialized
        return stats

    def query_stream(
        self,
        question: str,
        chat_history: List[Dict] = None,
        use_cot: bool = True,
        use_thinking: bool = False,
        max_retry: int = 1,
    ) -> Iterator[str]:
        """
        流式问答入口
        支持多轮对话、Self-RAG 补充检索
        """
        if not self._initialized:
            self.initialize_retriever()

        chat_history = chat_history or []

        # ─── 无知识库时的 Fallback ────────────────────────────────────────
        if not self.is_ready():
            yield from self.generator.generate_no_context(question, chat_history)
            return

        # ─── 步骤1: 混合检索 ──────────────────────────────────────────────
        yield "🔍 正在检索相关知识...\n\n"
        all_docs, debug_info = self.retriever.retrieve(
            question, chat_history=chat_history
        )

        # 分离向量和 BM25 结果（用于 RRF）
        vector_docs = [d for d in all_docs if "vector" in d.retrieval_type]
        bm25_docs = [d for d in all_docs if d.retrieval_type == "bm25"]

        # ─── 步骤2: 后处理（RRF + Self-RAG + 重排序） ────────────────────
        final_docs, need_more = self.post_processor.process(
            debug_info.get("rewritten_query", question),
            vector_docs,
            bm25_docs,
            top_k=TOP_K_FINAL
        )

        # ─── 步骤3: Self-RAG 补充检索 ────────────────────────────────────
        if need_more and max_retry > 0:
            yield "🔄 补充检索中...\n\n"
            # 扩展检索：使用子问题
            for sub_q in debug_info.get("sub_queries", [])[:2]:
                extra_docs, _ = self.retriever.retrieve(sub_q)
                extra_vector = [d for d in extra_docs if "vector" in d.retrieval_type]
                extra_bm25 = [d for d in extra_docs if d.retrieval_type == "bm25"]
                extra_final, _ = self.post_processor.process(
                    sub_q, extra_vector, extra_bm25, top_k=3
                )
                final_docs = list({d.doc_id: d for d in final_docs + extra_final}.values())
            final_docs = final_docs[:TOP_K_FINAL]

        subject = debug_info.get("subject", "综合")
        rewritten = debug_info.get("rewritten_query", question)
        yield f"📚 找到 **{len(final_docs)}** 条相关资料（学科: {subject}）\n\n"
        yield "---\n\n"

        # ─── 步骤4: 生成答案 ──────────────────────────────────────────────
        yield from self.generator.generate_stream(
            question,
            final_docs,
            chat_history=chat_history,
            use_cot=use_cot,
            use_thinking=use_thinking,
        )

    def query(
        self,
        question: str,
        chat_history: List[Dict] = None,
        use_cot: bool = True,
    ) -> RAGResponse:
        """
        同步问答接口
        """
        if not self._initialized:
            self.initialize_retriever()

        chat_history = chat_history or []
        response = RAGResponse()

        if not self.is_ready():
            answer, _ = self.generator.generate_no_context.__wrapped__(question, chat_history) \
                if hasattr(self.generator.generate_no_context, '__wrapped__') \
                else ("知识库为空，请先上传文档。", {})
            response.answer = answer
            return response

        all_docs, debug_info = self.retriever.retrieve(question, chat_history=chat_history)
        vector_docs = [d for d in all_docs if "vector" in d.retrieval_type]
        bm25_docs = [d for d in all_docs if d.retrieval_type == "bm25"]

        final_docs, _ = self.post_processor.process(
            debug_info.get("rewritten_query", question),
            vector_docs, bm25_docs
        )

        answer, citation_map = self.generator.generate(
            question, final_docs, chat_history=chat_history, use_cot=use_cot
        )

        response.answer = answer
        response.sources = [
            {
                "index": idx,
                "source": info["source"],
                "page": info["page"],
                "subject": info["subject"],
                "preview": info["content_preview"]
            }
            for idx, info in citation_map.items()
        ]
        response.debug_info = debug_info
        response.rewritten_query = debug_info.get("rewritten_query", question)
        response.subject = debug_info.get("subject", "综合")
        response.retrieval_count = len(final_docs)
        return response
