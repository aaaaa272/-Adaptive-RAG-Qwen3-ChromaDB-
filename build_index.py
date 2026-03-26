"""
命令行索引工具（用于批量建立知识库）
用法：
    python build_index.py --dir ./data/docs --no-summary
    python build_index.py --file ./data/docs/数据结构.pdf
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.indexer import KnowledgeIndexer


def main():
    parser = argparse.ArgumentParser(description="考研知识库索引工具")
    parser.add_argument("--dir", type=str, default="./data/docs", help="文档目录")
    parser.add_argument("--file", type=str, help="单个文件路径")
    parser.add_argument("--no-summary", action="store_true", help="跳过摘要生成（更快但精度略低）")
    parser.add_argument("--stats", action="store_true", help="查看知识库状态")
    args = parser.parse_args()

    indexer = KnowledgeIndexer()

    if args.stats:
        stats = indexer.get_collection_stats()
        print(f"知识库状态: {stats}")
        return

    use_summary = not args.no_summary

    if args.file:
        print(f"索引文件: {args.file}")
        n = indexer.index_file(args.file, use_summary=use_summary)
        print(f"完成，共 {n} 个知识块")
    else:
        print(f"索引目录: {args.dir}")
        n = indexer.index_directory(args.dir, use_summary=use_summary)
        print(f"全部完成，共 {n} 个知识块")

    stats = indexer.get_collection_stats()
    print(f"知识库状态: {stats}")


if __name__ == "__main__":
    main()
