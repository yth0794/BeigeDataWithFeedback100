#!/usr/bin/env python3
"""
Hugging Face Hub Uploader
将本地数据集上传到Hugging Face Hub
"""

import os
import json
from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo, login
import argparse
from typing import Optional

class HuggingFaceUploader:
    def __init__(self, dataset_path: str, repo_name: str, token: Optional[str] = None):
        self.dataset_path = dataset_path
        self.repo_name = repo_name
        self.token = token
        self.api = HfApi()
        
    def login_to_hub(self):
        """登录到Hugging Face Hub"""
        try:
            if self.token:
                login(token=self.token)
            else:
                login()  # 使用交互式登录
            print("✅ 成功登录到Hugging Face Hub")
            return True
        except Exception as e:
            print(f"❌ 登录失败: {str(e)}")
            return False
    
    def create_repository(self):
        """创建数据集仓库"""
        try:
            create_repo(
                repo_id=self.repo_name,
                repo_type="dataset",
                exist_ok=True
            )
            print(f"✅ 成功创建数据集仓库: {self.repo_name}")
            return True
        except Exception as e:
            print(f"❌ 创建仓库失败: {str(e)}")
            return False
    
    def upload_dataset(self):
        """上传数据集"""
        try:
            # 检查数据集路径
            if not os.path.exists(self.dataset_path):
                print(f"❌ 数据集路径不存在: {self.dataset_path}")
                return False
            
            # 加载数据集以验证
            dataset = load_from_disk(self.dataset_path)
            print(f"📊 数据集包含 {len(dataset['train'])} 个样本")
            
            # 上传数据集
            self.api.upload_folder(
                folder_path=self.dataset_path,
                repo_id=self.repo_name,
                repo_type="dataset"
            )
            
            print(f"✅ 数据集上传成功: https://huggingface.co/datasets/{self.repo_name}")
            return True
            
        except Exception as e:
            print(f"❌ 上传失败: {str(e)}")
            return False
    
    def upload_readme(self):
        """上传README文件"""
        try:
            readme_path = os.path.join(os.path.dirname(self.dataset_path), "README.md")
            if os.path.exists(readme_path):
                self.api.upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=self.repo_name,
                    repo_type="dataset"
                )
                print("✅ README文件上传成功")
            else:
                print("⚠️ README文件不存在，跳过上传")
            return True
        except Exception as e:
            print(f"❌ README上传失败: {str(e)}")
            return False
    
    def run_upload(self):
        """运行完整的上传流程"""
        print("🚀 开始上传数据集到Hugging Face Hub...")
        
        # 1. 登录
        if not self.login_to_hub():
            return False
        
        # 2. 创建仓库
        if not self.create_repository():
            return False
        
        # 3. 上传数据集
        if not self.upload_dataset():
            return False
        
        # 4. 上传README
        self.upload_readme()
        
        print("🎉 上传完成！")
        print(f"🔗 数据集链接: https://huggingface.co/datasets/{self.repo_name}")
        return True

def main():
    parser = argparse.ArgumentParser(description="上传数据集到Hugging Face Hub")
    parser.add_argument("--dataset_path", "-d", required=True, help="本地数据集路径")
    parser.add_argument("--repo_name", "-r", required=True, help="Hugging Face仓库名称 (格式: username/dataset-name)")
    parser.add_argument("--token", "-t", help="Hugging Face访问令牌 (可选)")
    
    args = parser.parse_args()
    
    uploader = HuggingFaceUploader(args.dataset_path, args.repo_name, args.token)
    success = uploader.run_upload()
    
    if success:
        print("\n📋 后续步骤:")
        print("1. 访问数据集页面查看上传结果")
        print("2. 添加数据集标签和描述")
        print("3. 设置数据集可见性")
        print("4. 邀请协作者 (如需要)")

if __name__ == "__main__":
    main()
