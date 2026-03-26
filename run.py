#!/usr/bin/env python
"""
快速启动脚本
"""
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent

def check_env():
    env_file = ROOT / ".env"
    if not env_file.exists():
        example = ROOT / ".env.example"
        if example.exists():
            import shutil
            shutil.copy(example, env_file)
            print("✅ 已自动创建 .env 文件（从 .env.example 复制）")
            print("⚠️  请编辑 .env 文件，填入你的 DASHSCOPE_API_KEY")
        else:
            print("❌ 未找到 .env.example，请手动创建 .env 文件")
        return False

    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if not key or key == "your_dashscope_api_key_here":
        print("⚠️  DASHSCOPE_API_KEY 未设置！请编辑 .env 文件")
        print("   获取方式：https://bailian.console.aliyun.com/")
        return False
    return True


def install_deps():
    print("📦 安装依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"], check=True)
    print("✅ 依赖安装完成")


if __name__ == "__main__":
    os.chdir(ROOT)

    if "--install" in sys.argv:
        install_deps()

    ready = check_env()
    if not ready and "--force" not in sys.argv:
        print("\n请先配置 .env 文件后重新运行")
        sys.exit(1)

    print("\n🚀 启动考研知识智能问答系统...")
    print("   访问地址: http://localhost:7861")
    print("   按 Ctrl+C 退出\n")

    subprocess.run([sys.executable, "app.py"])
