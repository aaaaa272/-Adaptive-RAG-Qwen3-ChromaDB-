"""
Gradio Web 界面
考研知识智能问答系统
"""
import os
import sys
import time
import shutil
import threading
from pathlib import Path
from typing import List, Optional

import gradio as gr
from loguru import logger

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DOCS_DIR, DASHSCOPE_API_KEY
from src.rag_pipeline import AgenticRAGPipeline


# ─── 全局状态 ────────────────────────────────────────────────────────────────

pipeline: Optional[AgenticRAGPipeline] = None
pipeline_lock = threading.Lock()


def get_pipeline() -> AgenticRAGPipeline:
    global pipeline
    if pipeline is None:
        with pipeline_lock:
            if pipeline is None:
                pipeline = AgenticRAGPipeline(use_self_rag=True)
    return pipeline


# ─── 功能函数 ────────────────────────────────────────────────────────────────

def upload_and_index(files, use_summary: bool, progress=gr.Progress()):
    """上传文件并建立索引"""
    if not files:
        return "⚠️ 请先上传文件", get_db_stats()

    p = get_pipeline()
    total_chunks = 0
    logs = []

    for i, file in enumerate(files):
        filename = Path(file.name).name
        dest = DOCS_DIR / filename
        shutil.copy2(file.name, dest)
        progress((i + 0.5) / len(files), desc=f"索引 {filename}...")
        logs.append(f"📄 正在处理: **{filename}**")

        try:
            n = p.indexer.index_file(str(dest), use_summary=use_summary)
            total_chunks += n
            logs.append(f"  ✅ 完成，生成 {n} 个知识块")
        except Exception as e:
            logs.append(f"  ❌ 失败: {e}")
            logger.error(f"索引 {filename} 失败: {e}")

        progress((i + 1) / len(files), desc=f"完成 {filename}")

    # 重新初始化检索器
    p.initialize_retriever()

    msg = "\n".join(logs)
    msg += f"\n\n✅ **索引完成！** 共生成 {total_chunks} 个知识块"
    return msg, get_db_stats()


def get_db_stats() -> str:
    """获取知识库状态"""
    try:
        p = get_pipeline()
        stats = p.get_stats()
        files = list(DOCS_DIR.glob("**/*.pdf")) + \
                list(DOCS_DIR.glob("**/*.txt")) + \
                list(DOCS_DIR.glob("**/*.md"))
        return (
            f"📊 **知识库状态**\n"
            f"- 文档数量: {len(files)} 个文件\n"
            f"- 原始知识块: {stats['raw_chunks']} 条\n"
            f"- 摘要索引: {stats['summary_chunks']} 条\n"
            f"- 检索器: {'✅ 就绪' if stats['retriever_ready'] else '⏳ 待初始化'}"
        )
    except Exception as e:
        return f"❌ 获取状态失败: {e}"


def clear_database():
    """清空知识库"""
    global pipeline
    try:
        import chromadb
        from src.config import CHROMA_DB_PATH
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
        pipeline = None
        get_pipeline()  # 重新初始化
        return "✅ 知识库已清空", get_db_stats()
    except Exception as e:
        return f"❌ 清空失败: {e}", get_db_stats()


def chat(
    message: str,
    history: list,
    use_cot: bool,
    use_thinking: bool,
):
    """聊天接口（流式）"""
    if not message.strip():
        yield history, ""
        return

    p = get_pipeline()

    # 转换历史格式
    chat_history = []
    for h in history[-4:]:
        if isinstance(h, dict):
            user_msg = h.get("role") == "user" and h.get("content")
            # Gradio 6 新格式，跳过，下面单独处理
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            user_msg, ai_msg = h
            if user_msg and ai_msg:
                chat_history.append({"user": str(user_msg), "assistant": str(ai_msg)})

    # 流式生成（Gradio 6 新消息格式）
    accumulated = ""
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]

    try:
        for chunk in p.query_stream(
            message,
            chat_history=chat_history,
            use_cot=use_cot,
            use_thinking=use_thinking,
        ):
            accumulated += chunk
            history[-1]["content"] = accumulated
            yield history, ""
    except Exception as e:
        error_msg = f"❌ 发生错误: {e}"
        history[-1]["content"] = error_msg
        yield history, ""


def init_retriever():
    """手动初始化检索器"""
    try:
        p = get_pipeline()
        p.initialize_retriever()
        return "✅ 检索器初始化完成", get_db_stats()
    except Exception as e:
        return f"❌ 初始化失败: {e}", get_db_stats()


# ─── Gradio 界面 ──────────────────────────────────────────────────────────────

CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.chat-message { font-size: 15px; line-height: 1.6; }
.source-box { background: #f0f4ff; border-radius: 8px; padding: 12px; }
footer { display: none !important; }
"""

def create_ui():
    with gr.Blocks(
        title="考研知识智能问答系统",
    ) as demo:
        gr.Markdown(
            """
            # 🎓 考研知识智能问答系统
            **基于 Adaptive-RAG + Qwen3 + ChromaDB 的智能考研辅导助手**

            > 支持计算机专业课（数据结构、操作系统、网络、计组）及数学、英语、政治
            """
        )

        with gr.Tabs():
            # ─── Tab 1: 问答 ──────────────────────────────────────────────
            with gr.TabItem("💬 智能问答"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="对话",
                            height=550,
                            elem_classes=["chat-message"]
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="输入考研问题，例如：请解释TCP三次握手的过程...",
                                label="",
                                scale=4,
                                lines=2,
                                max_lines=4,
                            )
                            send_btn = gr.Button("发送 ▶", variant="primary", scale=1)

                        with gr.Row():
                            clear_chat_btn = gr.Button("🗑️ 清空对话", size="sm")
                            gr.Markdown("**提示**: 按 Enter 发送，Shift+Enter 换行")

                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 生成设置")
                        use_cot = gr.Checkbox(
                            label="启用 CoT 思维链", value=True,
                            info="逐步推理，适合计算和分析题"
                        )
                        use_thinking = gr.Checkbox(
                            label="启用深度思考 (Qwen3)", value=False,
                            info="更深入的推理，速度较慢"
                        )

                        gr.Markdown("### 📊 知识库状态")
                        db_stats_display = gr.Markdown(value=get_db_stats())
                        refresh_stats_btn = gr.Button("🔄 刷新状态", size="sm")

                        gr.Markdown("### 💡 示例问题")
                        examples = gr.Examples(
                            examples=[
                                ["请详细解释TCP三次握手和四次挥手的过程及原因"],
                                ["二叉树的前序、中序、后序遍历有什么区别？请给出代码"],
                                ["操作系统中的死锁是什么？如何检测和避免？"],
                                ["虚拟内存的工作原理是什么？什么是页面置换算法？"],
                                ["计算机组成原理中流水线的冒险有哪几种类型？"],
                                ["高数中格林公式的使用条件和计算方法"],
                            ],
                            inputs=msg_input,
                            label=""
                        )

            # ─── Tab 2: 知识库管理 ─────────────────────────────────────────
            with gr.TabItem("📚 知识库管理"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 上传教材文档")
                        file_upload = gr.File(
                            label="支持 PDF / TXT / Markdown",
                            file_types=[".pdf", ".txt", ".md"],
                            file_count="multiple",
                        )
                        use_summary_cb = gr.Checkbox(
                            label="启用 Multi-representation（生成摘要索引）",
                            value=True,
                            info="提高检索精度，但索引速度较慢（需要调用 LLM）"
                        )
                        with gr.Row():
                            index_btn = gr.Button("📥 上传并建立索引", variant="primary")
                            init_retriever_btn = gr.Button("🔧 初始化检索器")

                        index_log = gr.Markdown("等待上传文档...")

                    with gr.Column(scale=1):
                        gr.Markdown("### 知识库状态")
                        db_stats_tab = gr.Markdown(value=get_db_stats())

                        gr.Markdown("### 危险操作")
                        clear_db_btn = gr.Button("🗑️ 清空知识库", variant="stop")
                        clear_db_log = gr.Markdown("")

            # ─── Tab 3: 系统说明 ──────────────────────────────────────────
            with gr.TabItem("ℹ️ 系统说明"):
                gr.Markdown("""
                ## 🏗️ 系统架构

                ### 技术栈
                - **LLM**: Qwen3-8b（阿里云百炼）
                - **Embedding**: text-embedding-v3（Qwen3 Embedding）
                - **向量数据库**: ChromaDB（本地持久化）
                - **关键词检索**: BM25 + 中文分词（jieba）
                - **重排序**: Cohere / Qwen3 本地重排序
                - **Web 界面**: Gradio

                ### 核心流程

                ```
                用户问题
                    ↓
                [1] 问题重写（Qwen3 上下文压缩 + 语义理解）
                    ↓
                [2] Adaptive-RAG 路由（学科感知，动态策略）
                    ↓
                [3] 混合检索
                    ├── BM25 关键词检索（jieba 中文分词）
                    ├── 向量检索（Qwen3-Embedding）
                    └── 摘要向量检索（Multi-representation）
                    ↓
                [4] Self-RAG 多轮评估（相关性过滤 + 补充检索）
                    ↓
                [5] RRF 融合去重（倒数排名融合）
                    ↓
                [6] Cohere/Qwen3 语义重排序
                    ↓
                [7] CoT 生成 + Citation 机制（Qwen3-8b）
                    ↓
                最终答案（含参考来源标注）
                ```

                ### 快速开始

                1. 在 `.env` 文件中配置 `DASHSCOPE_API_KEY`
                2. 在「知识库管理」标签页上传教材 PDF
                3. 点击「上传并建立索引」
                4. 在「智能问答」标签页提问

                ### 配置说明

                | 参数 | 默认值 | 说明 |
                |------|--------|------|
                | `CHUNK_SIZE` | 512 | 文本分块大小（字符数） |
                | `TOP_K_FINAL` | 5 | 最终返回文档数 |
                | `COHERE_API_KEY` | 空 | 留空使用 Qwen3 本地重排序 |
                """)

        # ─── 事件绑定 ────────────────────────────────────────────────────

        # 发送消息
        def submit_chat(message, history, cot, thinking):
            if not message.strip():
                return history, ""
            return None, ""

        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot, use_cot, use_thinking],
            outputs=[chatbot, msg_input],
            show_progress=False,
        )
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot, use_cot, use_thinking],
            outputs=[chatbot, msg_input],
            show_progress=False,
        )
        clear_chat_btn.click(lambda: [], outputs=[chatbot])

        # 知识库管理
        index_btn.click(
            fn=upload_and_index,
            inputs=[file_upload, use_summary_cb],
            outputs=[index_log, db_stats_tab],
            show_progress=True,
        ).then(fn=get_db_stats, outputs=[db_stats_display])

        init_retriever_btn.click(
            fn=init_retriever,
            outputs=[index_log, db_stats_tab],
        ).then(fn=get_db_stats, outputs=[db_stats_display])

        clear_db_btn.click(
            fn=clear_database,
            outputs=[clear_db_log, db_stats_tab],
        ).then(fn=get_db_stats, outputs=[db_stats_display])

        refresh_stats_btn.click(fn=get_db_stats, outputs=[db_stats_display])

    return demo


if __name__ == "__main__":
    if not DASHSCOPE_API_KEY:
        print("⚠️  警告：未设置 DASHSCOPE_API_KEY")
        print("请复制 .env.example 为 .env 并填入你的 API Key")
        print()

    # 预初始化
    logger.info("启动考研知识智能问答系统...")
    p = get_pipeline()
    if p.is_ready():
        logger.info("检测到已有知识库，初始化检索器...")
        p.initialize_retriever()

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CSS,
    )
