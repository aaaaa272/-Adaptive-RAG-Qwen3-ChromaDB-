"""
检索后处理模块（Self-RAG + RRF + Cohere 重排序）
- RRF 算法：融合多路检索结果并去重
- Self-RAG：多轮检索评估，判断是否需要补充检索
- Cohere 语义重排序（可降级到本地 LLM 重排序）
"""
import os
from typing import List, Tuple, Optional, Dict
from loguru import logger
from openai import OpenAI

from src.config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    QWEN_CHAT_MODEL, COHERE_API_KEY, TOP_K_FINAL
)
from src.retriever import RetrievedDoc


# ─── RRF 融合算法 ─────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    doc_lists: List[List[RetrievedDoc]],
    k: int = 60
) -> List[RetrievedDoc]:
    """
    Reciprocal Rank Fusion - 融合多路检索结果
    RRF(d) = Σ 1/(k + rank_i(d))
    自动去重（按 content 哈希）
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, RetrievedDoc] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            # 用内容哈希去重
            key = hash(doc.content[:200])
            key_str = str(key)
            rrf_score = 1.0 / (k + rank + 1)
            scores[key_str] = scores.get(key_str, 0) + rrf_score
            if key_str not in doc_map:
                doc_map[key_str] = doc

    # 按 RRF 分数排序
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    result = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.score = scores[key]
        result.append(doc)
    return result


# ─── Self-RAG 相关性评估 ──────────────────────────────────────────────────────

class SelfRAGEvaluator:
    """
    Self-RAG 多轮检索策略
    - 评估每个 chunk 与问题的相关性
    - 决定是否需要补充检索
    """

    def __init__(self):
        self.client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    def is_relevant(self, query: str, doc_content: str) -> Tuple[bool, float]:
        """判断文档是否与问题相关，返回 (is_relevant, confidence)"""
        prompt = f"""判断以下文档片段是否包含回答问题所需的信息。

问题: {query}

文档片段:
{doc_content[:400]}

请只回答 "相关" 或 "不相关"，再给出置信度（0.0-1.0）。
格式: 相关|0.9 或 不相关|0.2"""

        try:
            resp = self.client.chat.completions.create(
                model=QWEN_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                temperature=0.0,
                extra_body={"enable_thinking": False}
            )
            text = resp.choices[0].message.content.strip()
            is_rel = "相关" in text and "不相关" not in text
            conf_match = __import__('re').search(r'(\d+\.?\d*)', text)
            confidence = float(conf_match.group(1)) if conf_match else (0.8 if is_rel else 0.2)
            return is_rel, min(confidence, 1.0)
        except Exception as e:
            logger.warning(f"Self-RAG 评估失败: {e}")
            return True, 0.5  # 默认保留

    def filter_relevant(
        self,
        query: str,
        docs: List[RetrievedDoc],
        max_eval: int = 15
    ) -> List[RetrievedDoc]:
        """过滤不相关文档，保留高置信度相关文档"""
        if not docs:
            return []

        # 只对前 max_eval 个文档做 LLM 评估（节省 API 调用）
        to_eval = docs[:max_eval]
        filtered = []

        for doc in to_eval:
            is_rel, conf = self.is_relevant(query, doc.content)
            if is_rel and conf >= 0.5:
                doc.score = doc.score * 0.7 + conf * 0.3  # 融合分数
                filtered.append(doc)

        # 剩余文档（超出 max_eval 的）直接保留
        filtered.extend(docs[max_eval:])
        return filtered

    def needs_more_retrieval(self, query: str, docs: List[RetrievedDoc]) -> bool:
        """判断当前检索结果是否足够回答问题"""
        if not docs:
            return True
        if len(docs) < 2:
            return True

        context = "\n\n".join([d.content[:300] for d in docs[:3]])
        prompt = f"""根据以下文档片段，能否充分回答问题？

问题: {query}

文档片段:
{context}

请只回答 "充分" 或 "不充分"。"""

        try:
            resp = self.client.chat.completions.create(
                model=QWEN_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0.0,
                extra_body={"enable_thinking": False}
            )
            return "不充分" in resp.choices[0].message.content
        except Exception:
            return False


# ─── 重排序器 ────────────────────────────────────────────────────────────────

class Reranker:
    """
    语义重排序器
    优先使用 Cohere，降级到 LLM 本地重排序
    """

    def __init__(self):
        self.cohere_available = bool(COHERE_API_KEY)
        if self.cohere_available:
            try:
                import cohere
                self.co = cohere.Client(api_key=COHERE_API_KEY)
                logger.info("Cohere 重排序器初始化成功")
            except Exception as e:
                logger.warning(f"Cohere 初始化失败，使用本地重排序: {e}")
                self.cohere_available = False

        if not self.cohere_available:
            self.llm_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
            logger.info("使用 Qwen3 本地重排序")

    def rerank(
        self,
        query: str,
        docs: List[RetrievedDoc],
        top_k: int = TOP_K_FINAL
    ) -> List[RetrievedDoc]:
        """语义重排序，返回 top_k 个最相关文档"""
        if not docs:
            return []

        if self.cohere_available:
            return self._cohere_rerank(query, docs, top_k)
        else:
            return self._llm_rerank(query, docs, top_k)

    def _cohere_rerank(self, query: str, docs: List[RetrievedDoc], top_k: int) -> List[RetrievedDoc]:
        """Cohere 语义重排序"""
        try:
            texts = [d.content[:512] for d in docs]
            results = self.co.rerank(
                query=query,
                documents=texts,
                model="rerank-multilingual-v3.0",
                top_n=min(top_k, len(docs))
            )
            reranked = []
            for r in results.results:
                doc = docs[r.index]
                doc.score = r.relevance_score
                reranked.append(doc)
            return reranked
        except Exception as e:
            logger.warning(f"Cohere 重排序失败，降级到本地: {e}")
            return self._llm_rerank(query, docs, top_k)

    def _llm_rerank(self, query: str, docs: List[RetrievedDoc], top_k: int) -> List[RetrievedDoc]:
        """使用 Qwen3 进行本地重排序"""
        if len(docs) <= top_k:
            return docs[:top_k]

        # 构建排序 prompt
        doc_texts = "\n\n".join([
            f"[文档{i+1}] {doc.content[:300]}"
            for i, doc in enumerate(docs[:10])  # 最多评估10个
        ])

        prompt = f"""根据问题，对以下文档按相关性从高到低排序。

问题: {query}

{doc_texts}

请输出最相关的文档编号列表，用逗号分隔（例如: 3,1,5,2,4）。
只输出数字列表，不要其他内容。"""

        try:
            resp = self.llm_client.chat.completions.create(
                model=QWEN_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.0,
                extra_body={"enable_thinking": False}
            )
            raw = resp.choices[0].message.content.strip()
            import re
            indices = [int(x.strip()) - 1 for x in re.findall(r'\d+', raw)]
            indices = [i for i in indices if 0 <= i < len(docs)]

            # 按排序结果重组
            seen = set()
            reranked = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    reranked.append(docs[idx])

            # 补充未被排序的文档
            for i, doc in enumerate(docs):
                if i not in seen and len(reranked) < top_k:
                    reranked.append(doc)

            return reranked[:top_k]
        except Exception as e:
            logger.warning(f"LLM 重排序失败: {e}")
            return docs[:top_k]


# ─── 后处理流水线 ──────────────────────────────────────────────────────────────

class PostProcessor:
    """
    完整后处理流水线
    RRF 融合 → Self-RAG 过滤 → Cohere/LLM 重排序
    """

    def __init__(self, use_self_rag: bool = True):
        self.self_rag = SelfRAGEvaluator() if use_self_rag else None
        self.reranker = Reranker()
        self.use_self_rag = use_self_rag

    def process(
        self,
        query: str,
        vector_docs: List[RetrievedDoc],
        bm25_docs: List[RetrievedDoc],
        top_k: int = TOP_K_FINAL
    ) -> Tuple[List[RetrievedDoc], bool]:
        """
        完整后处理流程
        返回: (最终文档列表, 是否需要补充检索)
        """
        # 1. RRF 融合去重
        fused = reciprocal_rank_fusion(
            [vector_docs, bm25_docs],
            k=60
        )
        logger.info(f"RRF 融合后: {len(fused)} 个候选文档")

        # 2. Self-RAG 相关性过滤
        need_more = False
        if self.use_self_rag and fused:
            fused = self.self_rag.filter_relevant(query, fused)
            need_more = self.self_rag.needs_more_retrieval(query, fused)
            logger.info(f"Self-RAG 过滤后: {len(fused)} 个文档，需要补充检索: {need_more}")

        # 3. 语义重排序
        final = self.reranker.rerank(query, fused, top_k=top_k)
        logger.info(f"重排序后: {len(final)} 个最终文档")

        return final, need_more
