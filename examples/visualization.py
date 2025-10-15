#!/usr/bin/env python3
"""
Visualization Example
可视化示例
"""

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("📊 数据可视化示例...")
    
    # 加载数据集
    dataset = load_dataset("yth0794/BeigeDataWithFeedback100")
    df = dataset['train'].to_pandas()
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('数据集可视化分析', fontsize=16)
    
    # 评分分布
    if 'Essay_score' in df.columns:
        axes[0, 0].hist(df['Essay_score'], bins=10, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('评分分布')
        axes[0, 0].set_xlabel('评分')
        axes[0, 0].set_ylabel('频次')
    
    # 作文长度分布
    if 'Essay' in df.columns:
        essay_lengths = df['Essay'].str.len()
        axes[0, 1].hist(essay_lengths, bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('作文长度分布')
        axes[0, 1].set_xlabel('字符数')
        axes[0, 1].set_ylabel('频次')
    
    # 评分相关性
    score_cols = [col for col in df.columns if 'Score' in col]
    if len(score_cols) > 1:
        corr_matrix = df[score_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 0])
        axes[1, 0].set_title('评分相关性热力图')
    
    # 特征分布
    if 'Overall_score' in df.columns:
        axes[1, 1].boxplot(df['Overall_score'])
        axes[1, 1].set_title('总体评分箱线图')
        axes[1, 1].set_ylabel('评分')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ 可视化完成！")

if __name__ == "__main__":
    main()
