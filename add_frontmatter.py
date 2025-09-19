import os
import re
from datetime import datetime

def main():
    # 设置文章目录路径
    
    posts_dir = './source/_posts/E_Learning/Conference'
    
    print("开始批量添加 Front-matter...")
    
    # 检查目录是否存在
    if not os.path.exists(posts_dir):
        print(f"错误：目录 {posts_dir} 不存在！")
        return
    
    # 获取所有.md文件
    md_files = [f for f in os.listdir(posts_dir) if f.endswith('.md')]
    
    if not md_files:
        print("未找到.md文件！")
        return
    
    processed_count = 0
    
    for filename in md_files:
        try:
            filepath = os.path.join(posts_dir, filename)
            
            # 检查文件是否已经有Front-matter（避免重复添加）
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 如果文件已经包含Front-matter（以---开头），则跳过
            if content.startswith('---'):
                print(f"跳过 {filename}（已包含Front-matter）")
                continue
            
            # 从文件名提取日期信息
            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
            
            if date_match:
                year, month, day = date_match.groups()
                # 生成标题
                title = f"{year}年{month}月{day}日conference_note"
                
                # 检查是否是"latter"或"later"版本
                if 'latter' in filename.lower() or 'later' in filename.lower():
                    title += "（续）"
                
                # 生成日期字符串
                date_str = f"{year}-{month}-{day} 21:00:00"
            else:
                # 如果无法从文件名提取日期，使用文件名作为标题
                title = filename.replace('.md', '').replace('-', ' ')
                date_str = "2025-01-20 12:00:00"
            
            # 构造 Front-matter
            frontmatter = f"""---
title: {title}
date: {date_str}
tags: [E_L,Conference,conference_note]
---

"""
            # 写入文件（添加Front-matter到原内容前面）
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(frontmatter + content)
            
            print(f"✓ 已处理: {filename}")
            processed_count += 1
            
        except Exception as e:
            print(f"✗ 处理 {filename} 时出错: {str(e)}")
    
    print(f"\n完成！共处理 {processed_count} 个文件。")
    print("接下来请运行: hexo clean && hexo generate")

if __name__ == "__main__":
    main()