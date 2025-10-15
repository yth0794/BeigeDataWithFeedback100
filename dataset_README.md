# Essay Feedback Dataset

## 数据集描述
这是一个英语作文评分和反馈数据集，包含100篇作文的详细评分和反馈信息。

## 数据集结构
- **总样本数**: 100
- **特征数**: 14

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
print(f"训练集样本数: {len(train_data)}")

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
@dataset{essay_feedback_dataset,
  title={Essay Feedback Dataset},
  year={2024},
  url={https://github.com/your-repo/essay-feedback-dataset}
}
```
