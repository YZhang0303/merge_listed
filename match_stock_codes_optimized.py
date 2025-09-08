import polars as pl
import json
import re
from pathlib import Path

def clean_company_name(name):
    """清理公司名称，去除中英文括号内容"""
    if not name:
        return ""
    
    # 去除中文括号及其内容
    name = re.sub(r'[\(（][^)）]*[\)）]', '', name)
    # 去除英文括号及其内容
    name = re.sub(r'\([^)]*\)', '', name)
    # 去除多余空格
    name = re.sub(r'\s+', '', name)
    
    return name.strip()

def load_reference_data():
    """加载参考数据：上市公司信息、子公司信息和天眼查桥梁数据"""
    print("正在加载参考数据...")
    
    # 加载上市公司信息（JSONL格式）
    print("加载上市公司信息...")
    listed_companies = []
    with open('STK_LISTEDCOINFOANL.json', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                listed_companies.append(json.loads(line.strip()))
            if i % 10000 == 0:
                print(f"已加载上市公司记录: {i}")
    
    # 加载子公司信息（JSONL格式）
    print("加载子公司信息...")
    subsidiaries = []
    with open('FN_Fn061.json', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                subsidiaries.append(json.loads(line.strip()))
            if i % 10000 == 0:
                print(f"已加载子公司记录: {i}")
                
    with open('FN_Fn0611.json', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                subsidiaries.append(json.loads(line.strip()))
            if i % 10000 == 0:
                print(f"已加载更多子公司记录: {i}")
    
    # 加载天眼查桥梁数据
    print("加载天眼查桥梁数据...")
    tianyancha = pl.read_parquet('tianyancha.parquet')
    
    print(f"上市公司数量: {len(listed_companies)}")
    print(f"子公司数量: {len(subsidiaries)}")
    print(f"天眼查桥梁数据行数: {len(tianyancha)}")
    
    return listed_companies, subsidiaries, tianyancha

def create_company_stock_mapping(listed_companies, subsidiaries, tianyancha):
    """创建公司名称到股票代码的映射"""
    print("正在创建公司名称到股票代码的映射...")
    
    # 创建映射字典
    company_to_stock = {}
    
    # 1. 直接映射上市公司（取最新的记录）
    print("处理上市公司映射...")
    latest_companies = {}
    for company in listed_companies:
        stock_code = company.get('Symbol', '')
        full_name = company.get('FullName', '')
        short_name = company.get('ShortName', '')
        end_date = company.get('EndDate', '')
        
        if stock_code and full_name:
            # 保留最新日期的记录
            if stock_code not in latest_companies or end_date > latest_companies[stock_code].get('EndDate', ''):
                latest_companies[stock_code] = company
    
    # 使用最新的上市公司信息建立映射
    for company in latest_companies.values():
        stock_code = company.get('Symbol', '')
        full_name = company.get('FullName', '')
        short_name = company.get('ShortName', '')
        
        if stock_code and full_name:
            # 原始名称
            company_to_stock[full_name] = stock_code
            # 清理后的名称
            clean_name = clean_company_name(full_name)
            if clean_name and clean_name != full_name:
                company_to_stock[clean_name] = stock_code
        
        if stock_code and short_name:
            company_to_stock[short_name] = stock_code
            clean_short = clean_company_name(short_name)
            if clean_short and clean_short != short_name:
                company_to_stock[clean_short] = stock_code
    
    # 2. 映射子公司到母公司股票代码
    print("处理子公司映射...")
    parent_stock_codes = set(latest_companies.keys())
    
    for sub in subsidiaries:
        parent_code = sub.get('Stkcd', '')
        sub_name = sub.get('FN_Fn06101', '')
        
        if parent_code in parent_stock_codes and sub_name:
            company_to_stock[sub_name] = parent_code
            clean_sub_name = clean_company_name(sub_name)
            if clean_sub_name and clean_sub_name != sub_name:
                company_to_stock[clean_sub_name] = parent_code
    
    # 3. 使用天眼查数据建立映射关系
    print("处理天眼查映射...")
    tianyancha_mapping = {}
    for row in tianyancha.iter_rows(named=True):
        original_name = row.get('原文件导入名称', '')
        matched_name = row.get('系统匹配企业名称', '')
        
        if original_name and matched_name:
            tianyancha_mapping[original_name] = matched_name
            # 也对清理后的名称建立映射
            clean_original = clean_company_name(original_name)
            clean_matched = clean_company_name(matched_name)
            if clean_original and clean_matched:
                tianyancha_mapping[clean_original] = clean_matched
    
    print(f"直接映射的公司数量: {len(company_to_stock)}")
    print(f"天眼查映射关系数量: {len(tianyancha_mapping)}")
    
    return company_to_stock, tianyancha_mapping

def match_stock_code(company_name, company_to_stock, tianyancha_mapping):
    """为公司名称匹配股票代码"""
    if not company_name:
        return None
    
    # 1. 直接匹配原始名称
    if company_name in company_to_stock:
        return company_to_stock[company_name]
    
    # 2. 匹配清理后的名称
    clean_name = clean_company_name(company_name)
    if clean_name in company_to_stock:
        return company_to_stock[clean_name]
    
    # 3. 通过天眼查映射查找
    if company_name in tianyancha_mapping:
        matched_name = tianyancha_mapping[company_name]
        if matched_name in company_to_stock:
            return company_to_stock[matched_name]
        clean_matched = clean_company_name(matched_name)
        if clean_matched in company_to_stock:
            return company_to_stock[clean_matched]
    
    # 4. 清理后的名称通过天眼查映射查找
    if clean_name in tianyancha_mapping:
        matched_name = tianyancha_mapping[clean_name]
        if matched_name in company_to_stock:
            return company_to_stock[matched_name]
        clean_matched = clean_company_name(matched_name)
        if clean_matched in company_to_stock:
            return company_to_stock[clean_matched]
    
    return None

def process_batch_file(batch_file, company_to_stock, tianyancha_mapping, chunk_size=100000):
    """分块处理单个batch文件"""
    print(f"\n正在处理 {batch_file}...")
    
    # 读取batch文件的schema信息
    df_schema = pl.scan_parquet(batch_file)
    columns = df_schema.collect_schema().names()
    total_rows = pl.read_parquet(batch_file).height
    
    print(f"原始数据行数: {total_rows}")
    print(f"列名: {columns}")
    
    # 使用已知的公司名称字段
    company_column = '公司名称'
    
    if company_column not in columns:
        print(f"错误：未找到'{company_column}'字段")
        print(f"可用字段: {columns}")
        return None
    
    print(f"使用字段作为公司名称: {company_column}")
    
    # 分块处理
    all_results = []
    total_matched = 0
    
    for start_row in range(0, total_rows, chunk_size):
        end_row = min(start_row + chunk_size, total_rows)
        print(f"处理第 {start_row//chunk_size + 1} 块 (行 {start_row}-{end_row})...")
        
        # 使用lazy读取和切片
        chunk_df = pl.scan_parquet(batch_file).slice(start_row, end_row - start_row).collect()
        
        # 应用股票代码匹配
        stock_codes = []
        for company_name in chunk_df[company_column].to_list():
            stock_code = match_stock_code(company_name, company_to_stock, tianyancha_mapping)
            stock_codes.append(stock_code)
        
        # 添加股票代码列
        chunk_with_stock = chunk_df.with_columns([
            pl.Series(name='股票代码', values=stock_codes)
        ])
        
        # 统计匹配情况
        chunk_matched = chunk_with_stock.filter(pl.col('股票代码').is_not_null()).height
        total_matched += chunk_matched
        chunk_rate = chunk_matched / len(chunk_with_stock) * 100
        
        print(f"块匹配情况: {chunk_matched}/{len(chunk_with_stock)} ({chunk_rate:.2f}%)")
        
        all_results.append(chunk_with_stock)
    
    # 合并所有结果
    print("合并结果...")
    final_result = pl.concat(all_results)
    
    # 总体统计
    match_rate = total_matched / total_rows * 100
    print(f"总匹配结果: {total_matched}/{total_rows} ({match_rate:.2f}%)")
    
    # 保存结果
    output_file = f"{batch_file.stem}_with_stock_codes.parquet"
    final_result.write_parquet(output_file)
    print(f"结果已保存到: {output_file}")
    
    return final_result

def main():
    """主函数"""
    print("开始批量处理企业股票代码匹配任务...")
    
    # 加载参考数据
    listed_companies, subsidiaries, tianyancha = load_reference_data()
    
    # 创建映射
    company_to_stock, tianyancha_mapping = create_company_stock_mapping(
        listed_companies, subsidiaries, tianyancha
    )
    
    # 先处理最小的batch文件进行测试
    batch_files = [
        Path("batch_5.parquet"),  # 最小的文件
        Path("batch_1.parquet"),
        Path("batch_2.parquet"), 
        Path("batch_3.parquet"),
        Path("batch_4.parquet")
    ]
    
    results = {}
    
    for batch_file in batch_files:
        if batch_file.exists():
            try:
                result_df = process_batch_file(batch_file, company_to_stock, tianyancha_mapping)
                if result_df is not None:
                    results[batch_file.name] = result_df
            except Exception as e:
                print(f"处理 {batch_file} 时出错: {e}")
        else:
            print(f"文件不存在: {batch_file}")
    
    print("\n所有batch文件处理完成！")
    
    # 汇总统计
    total_processed = 0
    total_matched = 0
    for file_name, df in results.items():
        file_total = len(df)
        file_matched = len(df.filter(pl.col('股票代码').is_not_null()))
        total_processed += file_total
        total_matched += file_matched
        print(f"{file_name}: {file_matched}/{file_total} ({file_matched/file_total*100:.2f}%)")
    
    overall_rate = total_matched / total_processed * 100 if total_processed > 0 else 0
    print(f"\n总体匹配率: {total_matched}/{total_processed} ({overall_rate:.2f}%)")

if __name__ == "__main__":
    main()