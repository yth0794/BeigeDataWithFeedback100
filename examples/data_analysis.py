#!/usr/bin/env python3
"""
Data Analysis Example
数据分析示例
"""

from datasets import load_dataset
import pandas as pd
import numpy as np

def main():
    print("📊 数据分析示例...")
    
    # 加载数据集
    dataset = load_dataset("yth0794/BeigeDataWithFeedback100")
    df = dataset['train'].to_pandas()
    
    # 基本统计
    print("📈 基本统计信息:")
    print(df.describe())
    
    # 评分分析
    if 'Essay_score' in df.columns:
        scores = df['Essay_score'].tolist()
        print(f"\n📊 评分统计:")
        print(f"平均分: {np.mean(scores):.2f}")
        print(f"最高分: {max(scores)}")
        print(f"最低分: {min(scores)}")
        print(f"评分分布: {dict(pd.Series(scores).value_counts().sort_index())}")
    
    # 文本长度分析
    if 'Essay' in df.columns:
        essay_lengths = df['Essay'].str.len()
        print(f"\n📝 作文长度统计:")
        print(f"平均长度: {essay_lengths.mean():.0f} 字符")
        print(f"最长: {essay_lengths.max()} 字符")
        print(f"最短: {essay_lengths.min()} 字符")
    
    # 缺失值检查
    print(f"\n🔍 缺失值检查:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

if __name__ == "__main__":
    main()
