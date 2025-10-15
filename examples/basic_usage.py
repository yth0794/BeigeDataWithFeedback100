#!/usr/bin/env python3
"""
Basic Usage Example
基本使用示例
"""

from datasets import load_dataset
import pandas as pd

def main():
    print("🚀 加载数据集...")
    
    # 从GitHub加载数据集
    dataset = load_dataset("yth0794/BeigeDataWithFeedback100")
    
    print(f"📊 数据集信息: {dataset}")
    print(f"📈 训练集样本数: {len(dataset['train'])}")
    
    # 查看特征
    features = dataset['train'].features
    print(f"🔧 特征列表: {list(features.keys())}")
    
    # 查看第一个样本
    first_sample = dataset['train'][0]
    print(f"📝 第一个样本: {first_sample}")
    
    # 转换为pandas
    df = dataset['train'].to_pandas()
    print(f"📊 DataFrame形状: {df.shape}")
    print(f"📋 前5行数据:")
    print(df.head())

if __name__ == "__main__":
    main()
