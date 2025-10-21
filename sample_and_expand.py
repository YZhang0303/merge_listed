#!/usr/bin/env python3
"""
随机抽取merged数据中的10行，并根据custom_id列展开responsibilities和responsibilities_match列表
"""

import pandas as pd
import numpy as np

def main():
    print("正在加载数据文件...")
    
    # 读取merged数据文件
    try:
        # 尝试读取最可能的merged文件
        df = pd.read_parquet('matched_all_with_responsibility_2022.parquet')
        print(f"成功加载数据，形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
    except FileNotFoundError:
        print("文件 matched_all_with_responsibility_2022.parquet 不存在，尝试其他文件...")
        try:
            df = pd.read_parquet('matched_all_with_responsibility.parquet')
            print(f"成功加载数据，形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
        except FileNotFoundError:
            print("未找到合适的merged数据文件")
            return
    
    # 检查是否包含所需的列
    required_cols = ['responsibilities', 'responsibilities_match']
    available_cols = [col for col in required_cols if col in df.columns]
    print(f"可用的相关列: {available_cols}")
    
    # 随机抽取10行
    print("\n正在随机抽取10行数据...")
    sample_df = df.sample(n=10, random_state=42).reset_index(drop=True)
    
    # 显示基本信息
    print(f"抽样数据形状: {sample_df.shape}")
    
    # 寻找custom_id相关的列
    id_cols = [col for col in sample_df.columns if 'id' in col.lower() or 'custom' in col.lower()]
    print(f"可能的ID列: {id_cols}")
    
    # 检查responsibilities和responsibilities_match列的数据类型
    for col in available_cols:
        print(f"\n{col}列的数据类型和示例:")
        print(f"数据类型: {sample_df[col].dtype}")
        print(f"非空值数量: {sample_df[col].notna().sum()}")
        
        # 显示前几个非空值的示例
        non_null_values = sample_df[col].dropna().head(3)
        for i, val in enumerate(non_null_values):
            print(f"示例 {i+1}: {type(val)} - {str(val)[:200]}...")
    
    # 展开列表数据
    print("\n开始展开responsibilities和responsibilities_match列表...")
    
    expanded_rows = []
    
    for idx, row in sample_df.iterrows():
        # 获取ID信息（优先使用custom_id，如果没有则使用其他ID列）
        row_id = None
        if 'custom_id' in row:
            row_id = row['custom_id']
        elif id_cols:
            row_id = row[id_cols[0]]  # 使用第一个找到的ID列
        else:
            row_id = f"row_{idx}"  # 如果没有ID列，使用行索引
        
        # 处理responsibilities列
        responsibilities = row.get('responsibilities', [])
        responsibilities_match = row.get('responsibilities_match', [])
        
        # 确保都是列表类型
        if not isinstance(responsibilities, list):
            if pd.isna(responsibilities):
                responsibilities = []
            else:
                responsibilities = [responsibilities] if responsibilities else []
        
        if not isinstance(responsibilities_match, list):
            if pd.isna(responsibilities_match):
                responsibilities_match = []
            else:
                responsibilities_match = [responsibilities_match] if responsibilities_match else []
        
        # 展开列表，每个元素一行
        max_len = max(len(responsibilities), len(responsibilities_match), 1)
        
        for i in range(max_len):
            expanded_row = {
                'custom_id': row_id,
                'row_index': idx,
                'list_index': i,
                'responsibility': responsibilities[i] if i < len(responsibilities) else None,
                'responsibility_match': responsibilities_match[i] if i < len(responsibilities_match) else None
            }
            
            # 添加其他重要列
            for col in row.index:
                if col not in ['responsibilities', 'responsibilities_match']:
                    expanded_row[f'original_{col}'] = row[col]
            
            expanded_rows.append(expanded_row)
    
    # 创建展开后的DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    
    print(f"\n展开后的数据形状: {expanded_df.shape}")
    print("\n展开后的数据预览:")
    print(expanded_df[['custom_id', 'row_index', 'list_index', 'responsibility', 'responsibility_match']].to_string())
    
    # 保存结果
    output_file = 'expanded_sample_10_rows.csv'
    expanded_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_file}")
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"原始10行数据展开为 {len(expanded_df)} 行")
    print(f"有效responsibilities数量: {expanded_df['responsibility'].notna().sum()}")
    print(f"有效responsibilities_match数量: {expanded_df['responsibility_match'].notna().sum()}")
    
    # 按custom_id分组统计
    print(f"\n按custom_id分组的展开行数:")
    group_counts = expanded_df.groupby('custom_id').size()
    print(group_counts.to_string())

if __name__ == "__main__":
    main()







