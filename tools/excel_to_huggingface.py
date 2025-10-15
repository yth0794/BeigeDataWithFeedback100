#!/usr/bin/env python3
"""
Excel to Hugging Face Dataset Converter
将Excel文件转换为Hugging Face数据集格式
"""

import pandas as pd
import json
import os
from datasets import Dataset, DatasetDict
from typing import Dict, Any, List
import argparse

class ExcelToHuggingFaceConverter:
    def __init__(self, excel_file: str, output_dir: str = "huggingface_dataset"):
        self.excel_file = excel_file
        self.output_dir = output_dir
        self.df = None
        
    def load_excel(self) -> pd.DataFrame:
        """加载Excel文件"""
        print(f"正在加载Excel文件: {self.excel_file}")
        self.df = pd.read_excel(self.excel_file)
        print(f"数据形状: {self.df.shape}")
        print(f"列名: {list(self.df.columns)}")
        return self.df
    
    def analyze_data(self) -> Dict[str, Any]:
        """分析数据结构"""
        if self.df is None:
            self.load_excel()
        
        analysis = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "data_types": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "sample_data": self.df.head(3).to_dict('records')
        }
        
        print("=== 数据分析结果 ===")
        print(f"总行数: {analysis['total_rows']}")
        print(f"总列数: {analysis['total_columns']}")
        print(f"缺失值统计: {analysis['missing_values']}")
        
        return analysis
    
    def create_dataset_config(self) -> Dict[str, Any]:
        """创建数据集配置文件"""
        config = {
            "dataset_name": "essay_feedback_dataset",
            "version": "1.0.0",
            "description": "英语作文评分和反馈数据集",
            "language": "zh",
            "task_categories": ["text-classification", "text-generation"],
            "features": {
                "Essay_id": {"dtype": "int64", "description": "作文ID"},
                "Essay_Prompt": {"dtype": "string", "description": "作文题目"},
                "Essay": {"dtype": "string", "description": "作文内容"},
                "Essay_score": {"dtype": "int64", "description": "作文总分"},
                "Overall_score": {"dtype": "int64", "description": "总体评分"},
                "Score_TR": {"dtype": "int64", "description": "任务完成度评分"},
                "Score_CC": {"dtype": "int64", "description": "连贯性评分"},
                "Score_LR": {"dtype": "int64", "description": "词汇丰富度评分"},
                "Score_GRA": {"dtype": "int64", "description": "语法准确性评分"},
                "Feedback_TR": {"dtype": "string", "description": "任务完成度反馈"},
                "Feedback_CC": {"dtype": "string", "description": "连贯性反馈"},
                "Feedback_LR": {"dtype": "string", "description": "词汇丰富度反馈"},
                "Feedback_GRA": {"dtype": "string", "description": "语法准确性反馈"},
                "Suggestion for improvement": {"dtype": "string", "description": "改进建议"}
            },
            "splits": {
                "train": {
                    "num_examples": len(self.df),
                    "description": "训练集"
                }
            }
        }
        return config
    
    def convert_to_huggingface(self) -> Dataset:
        """转换为Hugging Face Dataset格式"""
        if self.df is None:
            self.load_excel()
        
        print("正在转换为Hugging Face Dataset格式...")
        
        # 处理缺失值和数据类型
        df_clean = self.df.copy()
        
        # 对于数值列，用0填充缺失值
        numeric_columns = ['Essay_id', 'Essay_score', 'Overall_score', 'Score_TR', 'Score_CC', 'Score_LR', 'Score_GRA']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype('int64')
        
        # 对于文本列，用空字符串填充缺失值
        text_columns = ['Essay_Prompt', 'Essay', 'Feedback_TR', 'Feedback_CC', 'Feedback_LR', 'Feedback_GRA', 'Suggestion for improvement']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna("").astype('string')
        
        # 转换为字典列表
        data_dict = df_clean.to_dict('records')
        
        # 创建Dataset
        dataset = Dataset.from_list(data_dict)
        
        print(f"成功创建Dataset，包含 {len(dataset)} 个样本")
        return dataset
    
    def save_dataset(self, dataset: Dataset) -> str:
        """保存数据集到本地"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存为DatasetDict格式
        dataset_dict = DatasetDict({
            "train": dataset
        })
        
        # 保存数据集
        dataset_path = os.path.join(self.output_dir, "dataset")
        dataset_dict.save_to_disk(dataset_path)
        
        # 保存配置文件
        config = self.create_dataset_config()
        config_path = os.path.join(self.output_dir, "dataset_info.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # 保存README
        self.create_readme()
        
        print(f"数据集已保存到: {dataset_path}")
        print(f"配置文件已保存到: {config_path}")
        
        return dataset_path
    
    def create_readme(self):
        """创建README文件"""
        readme_content = f"""# Essay Feedback Dataset

## 数据集描述
这是一个英语作文评分和反馈数据集，包含100篇作文的详细评分和反馈信息。

## 数据集结构
- **总样本数**: {len(self.df)}
- **特征数**: {len(self.df.columns)}

## 特征说明
- `Essay_id`: 作文ID
- `Essay_Prompt`: 作文题目
- `Essay`: 作文内容
- `Essay_score`: 作文总分
- `Overall_score`: 总体评分
- `Score_TR`: 任务完成度评分
- `Score_CC`: 连贯性评分
- `Score_LR`: 词汇丰富度评分
- `Score_GRA`: 语法准确性评分
- `Feedback_TR`: 任务完成度反馈
- `Feedback_CC`: 连贯性反馈
- `Feedback_LR`: 词汇丰富度反馈
- `Feedback_GRA`: 语法准确性反馈
- `Suggestion for improvement`: 改进建议

## 使用方法

### 加载数据集
```python
from datasets import load_from_disk

# 加载数据集
dataset = load_from_disk("./dataset")
print(dataset)

# 查看训练集
train_data = dataset["train"]
print(f"训练集样本数: {{len(train_data)}}")

# 查看第一个样本
print(train_data[0])
```

### 数据探索
```python
import pandas as pd

# 转换为pandas DataFrame进行探索
df = train_data.to_pandas()
print(df.head())
print(df.describe())
```

## 许可证
请根据您的需求添加适当的许可证信息。

## 引用
如果您使用了这个数据集，请引用：
```
@dataset{{essay_feedback_dataset,
  title={{Essay Feedback Dataset}},
  year={{2024}},
  url={{https://github.com/your-repo/essay-feedback-dataset}}
}}
```
"""
        
        readme_path = os.path.join(self.output_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"README文件已保存到: {readme_path}")
    
    def run_conversion(self):
        """运行完整的转换流程"""
        print("开始Excel到Hugging Face数据集转换...")
        
        # 1. 加载数据
        self.load_excel()
        
        # 2. 分析数据
        analysis = self.analyze_data()
        
        # 3. 转换为Hugging Face格式
        dataset = self.convert_to_huggingface()
        
        # 4. 保存数据集
        dataset_path = self.save_dataset(dataset)
        
        print("转换完成！")
        return dataset_path, analysis

def main():
    parser = argparse.ArgumentParser(description="将Excel文件转换为Hugging Face数据集格式")
    parser.add_argument("--excel_file", "-f", required=True, help="Excel文件路径")
    parser.add_argument("--output_dir", "-o", default="huggingface_dataset", help="输出目录")
    
    args = parser.parse_args()
    
    converter = ExcelToHuggingFaceConverter(args.excel_file, args.output_dir)
    dataset_path, analysis = converter.run_conversion()
    
    print(f"\n数据集已成功转换并保存到: {dataset_path}")
    print("您现在可以:")
    print("1. 使用 datasets.load_from_disk() 加载数据集")
    print("2. 上传到Hugging Face Hub")
    print("3. 使用Web界面查看器查看数据")

if __name__ == "__main__":
    main()
