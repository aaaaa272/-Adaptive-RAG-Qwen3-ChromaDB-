"""
生成模块
- Prompt Template + CoT（思维链）推理
- Citation 机制：关键信息可追溯验证
- 流式输出支持
"""
from typing import List, Iterator, Optional
from loguru import logger
from openai import OpenAI

from src.config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, QWEN_CHAT_MODEL
)
from src.retriever import RetrievedDoc


# ─── Prompt 模板 ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位专业的考研辅导老师，精通计算机专业课（数据结构、操作系统、计算机网络、计算机组成原理）和数学、英语、政治等公共课。

你的回答规范：
1. **严谨准确**：基于提供的参考资料回答，不编造内容
2. **思维链推理**：对复杂问题逐步分析，展示推理过程
3. **引用标注**：在关键结论后用 [来源N] 标注信息来源
4. **结构清晰**：使用标题、列表、代码块等格式组织答案
5. **考研导向**：突出考点、易错点和解题技巧

如果参考资料不足以回答问题，请如实说明并给出已知的部分分析。"""


COT_PROMPT_TEMPLATE = """## 参考资料

{context}

---

## 学生问题

{query}

## 回答要求

请按照以下步骤回答：

**【分析】** 首先分析这道题/问题考查的核心知识点

**【推导】** 逐步展开分析过程（对于计算题需写出推导步骤）

**【结论】** 给出最终答案，在关键知识点后标注引用来源，如 [来源1]

**【考研提示】** 指出该知识点的常见考法和易错点（如果适用）"""


def build_context_with_citations(docs: List[RetrievedDoc]) -> tuple:
    """
    构建带引用编号的上下文
    返回 (context_str, citation_map)
    """
    context_parts = []
    citation_map = {}

    for i, doc in enumerate(docs, 1):
        citation_map[i] = {
            "source": doc.source,
            "page": doc.page,
            "subject": doc.subject,
            "content_preview": doc.content[:100],
        }
        context_parts.append(
            f"[来源{i}] 文件: {doc.source} | 第{doc.page}页 | 学科: {doc.subject}\n"
            f"{doc.content}"
        )

    return "\n\n".join(context_parts), citation_map


def format_citations(citation_map: dict) -> str:
    """格式化引用列表"""
    lines = ["\n\n---\n**📚 参考来源**"]
    for idx, info in citation_map.items():
        lines.append(
            f"[来源{idx}] {info['source']} · 第{info['page']}页 · {info['subject']}"
        )
    return "\n".join(lines)


# ─── 生成器 ───────────────────────────────────────────────────────────────────

class AnswerGenerator:
    """
    基于 Qwen3 的问答生成器
    支持 CoT 推理 + Citation 机制 + 流式输出
    """

    def __init__(self):
        self.client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    def generate(
        self,
        query: str,
        docs: List[RetrievedDoc],
        chat_history: List[dict] = None,
        use_cot: bool = True,
        use_thinking: bool = False,
    ) -> tuple:
        """
        同步生成答案
        返回 (answer_text, citation_map)
        """
        context, citation_map = build_context_with_citations(docs)

        if use_cot:
            user_content = COT_PROMPT_TEMPLATE.format(context=context, query=query)
        else:
            user_content = f"参考资料:\n{context}\n\n问题: {query}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 加入历史对话（最近2轮）
        if chat_history:
            for h in chat_history[-2:]:
                messages.append({"role": "user", "content": h["user"]})
                messages.append({"role": "assistant", "content": h["assistant"]})

        messages.append({"role": "user", "content": user_content})

        # 构建请求参数
        req_params = {
            "model": QWEN_CHAT_MODEL,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.3,
        }

        # Qwen3 支持 enable_thinking（深度思考模式）
        if use_thinking:
            req_params["extra_body"] = {"enable_thinking": True}
        else:
            req_params["extra_body"] = {"enable_thinking": False}

        try:
            resp = self.client.chat.completions.create(**req_params)
            answer = resp.choices[0].message.content.strip()

            # 附加引用列表
            if citation_map:
                answer += format_citations(citation_map)

            return answer, citation_map
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return f"生成回答时出错: {e}", {}

    def generate_stream(
        self,
        query: str,
        docs: List[RetrievedDoc],
        chat_history: List[dict] = None,
        use_cot: bool = True,
        use_thinking: bool = False,
    ) -> Iterator[str]:
        """
        流式生成答案
        """
        context, citation_map = build_context_with_citations(docs)

        if use_cot:
            user_content = COT_PROMPT_TEMPLATE.format(context=context, query=query)
        else:
            user_content = f"参考资料:\n{context}\n\n问题: {query}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            for h in chat_history[-2:]:
                messages.append({"role": "user", "content": h["user"]})
                messages.append({"role": "assistant", "content": h["assistant"]})
        messages.append({"role": "user", "content": user_content})

        req_params = {
            "model": QWEN_CHAT_MODEL,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.3,
            "stream": True,
            "extra_body": {"enable_thinking": use_thinking},
        }

        try:
            full_response = ""
            thinking_content = ""
            in_thinking = False

            for chunk in self.client.chat.completions.create(**req_params):
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # 处理 thinking 内容（Qwen3 深度思考）
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    if not in_thinking:
                        in_thinking = True
                        yield "\n\n> 💭 **思考过程**\n> "
                    thinking_content += delta.reasoning_content
                    yield f"> {delta.reasoning_content.replace(chr(10), chr(10) + '> ')}"
                    continue

                if hasattr(delta, 'content') and delta.content:
                    if in_thinking:
                        in_thinking = False
                        yield "\n\n---\n\n"
                    full_response += delta.content
                    yield delta.content

            # 流式结束后追加引用
            if citation_map:
                citation_str = format_citations(citation_map)
                yield citation_str

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"\n\n❌ 生成出错: {e}"

    def generate_no_context(self, query: str, chat_history: List[dict] = None) -> Iterator[str]:
        """无检索上下文时的直接回答（知识库为空时的 fallback）"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if chat_history:
            for h in chat_history[-2:]:
                messages.append({"role": "user", "content": h["user"]})
                messages.append({"role": "assistant", "content": h["assistant"]})
        messages.append({"role": "user", "content": f"问题: {query}\n\n注意：当前知识库为空，请基于你的考研知识直接回答，并标注这是通用知识而非来自上传资料。"})

        try:
            for chunk in self.client.chat.completions.create(
                model=QWEN_CHAT_MODEL,
                messages=messages,
                max_tokens=1024,
                temperature=0.5,
                stream=True,
                extra_body={"enable_thinking": False}
            ):
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and hasattr(delta, 'content') and delta.content:
                    yield delta.content
        except Exception as e:
            yield f"❌ 生成出错: {e}"
