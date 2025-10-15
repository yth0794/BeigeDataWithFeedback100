#!/usr/bin/env python3
"""
快速启动脚本
一键启动所有功能
"""

import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        return False

def check_file_exists(file_path):
    """检查文件是否存在"""
    return os.path.exists(file_path)

def main():
    print("🚀 GitHub数据集转Hugging Face格式 - 快速启动")
    print("=" * 50)
    
    # 检查必要文件
    excel_file = "BeigeDataWithFeedback100.xlsx"
    if not check_file_exists(excel_file):
        print(f"❌ 找不到Excel文件: {excel_file}")
        return
    
    # 1. 转换Excel为Hugging Face格式
    if not check_file_exists("huggingface_dataset/dataset"):
        print("\n📊 步骤1: 转换Excel为Hugging Face格式")
        success = run_command(
            f"python excel_to_huggingface.py --excel_file {excel_file} --output_dir huggingface_dataset",
            "转换Excel文件"
        )
        if not success:
            return
    else:
        print("✅ 数据集已存在，跳过转换步骤")
    
    # 2. 验证数据集
    print("\n🔍 步骤2: 验证数据集")
    success = run_command(
        "python -c \"from datasets import load_from_disk; dataset = load_from_disk('huggingface_dataset/dataset'); print(f'✅ 数据集验证成功: {len(dataset[\\\"train\\\"])} 个样本')\"",
        "验证数据集"
    )
    
    # 3. 启动Web界面
    print("\n🌐 步骤3: 启动Web界面查看器")
    print("正在启动Streamlit应用...")
    print("📱 Web界面将在浏览器中自动打开")
    print("🔗 如果没有自动打开，请访问: http://localhost:8501")
    print("\n按 Ctrl+C 停止服务器")
    
    try:
        # 延迟2秒后打开浏览器
        time.sleep(2)
        webbrowser.open("http://localhost:8501")
        
        # 启动Streamlit
        subprocess.run(["streamlit", "run", "dataset_viewer.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("请手动运行: streamlit run dataset_viewer.py")

def show_help():
    """显示帮助信息"""
    print("""
📚 GitHub数据集转Hugging Face格式 - 使用指南

🎯 功能特性:
- ✅ Excel转Hugging Face格式
- ✅ Web界面数据查看器
- ✅ 上传到Hugging Face Hub
- ✅ 数据统计和可视化

🚀 快速开始:
1. 运行此脚本: python quick_start.py
2. 等待转换完成
3. 在浏览器中查看数据

📁 生成的文件:
- huggingface_dataset/dataset/     # Hugging Face数据集
- huggingface_dataset/README.md    # 数据集说明
- dataset_viewer.py               # Web查看器
- upload_to_hub.py               # 上传脚本

🔧 手动操作:
- 转换数据: python excel_to_huggingface.py --excel_file BeigeDataWithFeedback100.xlsx
- 启动查看器: streamlit run dataset_viewer.py
- 上传到Hub: python upload_to_hub.py --dataset_path huggingface_dataset/dataset --repo_name your-username/dataset-name

📞 技术支持:
- 检查依赖: pip install pandas openpyxl datasets huggingface_hub streamlit plotly
- 查看日志: 运行脚本时会显示详细进度信息
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_help()
    else:
        main()
