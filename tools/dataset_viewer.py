#!/usr/bin/env python3
"""
Hugging Face Style Dataset Viewer
类似Hugging Face的数据集查看器
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_from_disk
import json
import os
from typing import Dict, Any, List
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="Dataset Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DatasetViewer:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = None
        self.df = None
        self.config = None
        
    def load_dataset(self):
        """加载数据集"""
        try:
            self.dataset = load_from_disk(self.dataset_path)
            self.df = self.dataset["train"].to_pandas()
            
            # 加载配置文件
            config_path = os.path.join(os.path.dirname(self.dataset_path), "dataset_info.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            
            return True
        except Exception as e:
            st.error(f"加载数据集失败: {str(e)}")
            return False
    
    def render_header(self):
        """渲染页面头部"""
        st.title("📊 Dataset Viewer")
        
        if self.config:
            st.markdown(f"**Dataset:** {self.config.get('dataset_name', 'Unknown')}")
            st.markdown(f"**Version:** {self.config.get('version', 'Unknown')}")
            st.markdown(f"**Description:** {self.config.get('description', 'No description')}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总样本数", len(self.df))
        with col2:
            st.metric("特征数", len(self.df.columns))
        with col3:
            st.metric("缺失值", self.df.isnull().sum().sum())
        with col4:
            st.metric("数据类型", len(self.df.dtypes.unique()))
    
    def render_search_bar(self):
        """渲染搜索栏"""
        st.markdown("### 🔍 Search this dataset")
        search_term = st.text_input("Search", placeholder="Search...", key="search_input", label_visibility="collapsed")
        return search_term
    
    def filter_data(self, search_term: str):
        """根据搜索词过滤数据"""
        if not search_term:
            return self.df
        
        # 在所有文本列中搜索
        text_columns = self.df.select_dtypes(include=['object', 'string']).columns
        mask = pd.Series([False] * len(self.df))
        
        for col in text_columns:
            mask |= self.df[col].str.contains(search_term, case=False, na=False)
        
        return self.df[mask]
    
    def render_column_info(self):
        """渲染列信息"""
        st.markdown("### 📋 Column Information")
        
        for col in self.df.columns:
            col_type = str(self.df[col].dtype)
            null_count = self.df[col].isnull().sum()
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{col}**")
            with col2:
                st.write(f"`{col_type}`")
            with col3:
                st.write(f"{null_count} nulls")
            
            # 显示数据分布
            if self.df[col].dtype in ['int64', 'float64']:
                # 数值列的直方图
                fig = px.histogram(
                    self.df, 
                    x=col, 
                    title=f"{col} Distribution",
                    nbins=20
                )
                fig.update_layout(height=200, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 文本列的长度分布
                lengths = self.df[col].astype(str).str.len()
                fig = px.histogram(
                    x=lengths,
                    title=f"{col} Length Distribution",
                    nbins=20
                )
                fig.update_layout(height=200, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
    
    def render_data_table(self, filtered_df: pd.DataFrame):
        """渲染数据表格"""
        st.markdown("### 📊 Data Table")
        
        # 分页设置
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
        
        # 计算总页数
        total_pages = (len(filtered_df) - 1) // page_size + 1
        
        # 页码选择
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        # 计算当前页的数据
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        current_data = filtered_df.iloc[start_idx:end_idx]
        
        # 显示数据
        st.dataframe(
            current_data,
            width='stretch',
            height=400
        )
        
        # 分页信息
        st.info(f"Showing rows {start_idx + 1} to {min(end_idx, len(filtered_df))} of {len(filtered_df)}")
        
        # 分页导航
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        with col1:
            if st.button("◀ Previous", disabled=(page == 1)):
                st.rerun()
        with col3:
            st.write(f"Page {page} of {total_pages}")
        with col5:
            if st.button("Next ▶", disabled=(page == total_pages)):
                st.rerun()
    
    def render_statistics(self):
        """渲染统计信息"""
        st.markdown("### 📈 Dataset Statistics")
        
        # 数值列统计
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.markdown("#### Numerical Statistics")
            st.dataframe(self.df[numeric_cols].describe(), width='stretch')
        
        # 文本列统计
        text_cols = self.df.select_dtypes(include=['object', 'string']).columns
        if len(text_cols) > 0:
            st.markdown("#### Text Statistics")
            text_stats = []
            for col in text_cols:
                lengths = self.df[col].astype(str).str.len()
                text_stats.append({
                    'Column': col,
                    'Min Length': lengths.min(),
                    'Max Length': lengths.max(),
                    'Avg Length': lengths.mean(),
                    'Unique Values': self.df[col].nunique()
                })
            
            text_df = pd.DataFrame(text_stats)
            st.dataframe(text_df, width='stretch')
    
    def render_visualizations(self):
        """渲染可视化图表"""
        st.markdown("### 📊 Visualizations")
        
        # 评分分布
        score_cols = [col for col in self.df.columns if 'Score' in col]
        if score_cols:
            st.markdown("#### Score Distributions")
            
            fig = go.Figure()
            for col in score_cols:
                fig.add_trace(go.Histogram(
                    x=self.df[col],
                    name=col,
                    opacity=0.7
                ))
            
            fig.update_layout(
                title="Score Distributions",
                xaxis_title="Score",
                yaxis_title="Count",
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 相关性热力图
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.markdown("#### Correlation Heatmap")
            corr_matrix = self.df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """运行查看器"""
        if not self.load_dataset():
            return
        
        # 渲染头部
        self.render_header()
        
        # 渲染搜索栏
        search_term = self.render_search_bar()
        
        # 过滤数据
        filtered_df = self.filter_data(search_term)
        
        # 创建标签页
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Table", "📋 Column Info", "📈 Statistics", "📊 Visualizations"])
        
        with tab1:
            self.render_data_table(filtered_df)
        
        with tab2:
            self.render_column_info()
        
        with tab3:
            self.render_statistics()
        
        with tab4:
            self.render_visualizations()

def main():
    st.sidebar.title("Dataset Viewer")
    
    # 数据集路径选择
    dataset_path = st.sidebar.text_input(
        "Dataset Path",
        value="huggingface_dataset/dataset",
        help="输入Hugging Face数据集的路径"
    )
    
    if st.sidebar.button("Load Dataset"):
        if os.path.exists(dataset_path):
            viewer = DatasetViewer(dataset_path)
            viewer.run()
        else:
            st.error(f"数据集路径不存在: {dataset_path}")
    
    # 侧边栏信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 使用说明")
    st.sidebar.markdown("""
    1. 在左侧输入数据集路径
    2. 点击"Load Dataset"加载数据
    3. 使用搜索功能过滤数据
    4. 在不同标签页中探索数据
    """)
    
    st.sidebar.markdown("### 功能特性")
    st.sidebar.markdown("""
    - 🔍 实时搜索过滤
    - 📊 数据表格浏览
    - 📋 列信息分析
    - 📈 统计信息展示
    - 📊 可视化图表
    - 📄 分页浏览
    """)

if __name__ == "__main__":
    main()
