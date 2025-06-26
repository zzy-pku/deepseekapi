import pandas as pd
import glob
import os
import sqlite3
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import traceback

# 初始化 embedding 模型（强制用GPU）
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device='cuda')

def embed_text(text: str) -> np.ndarray:
    embedding = EMBEDDING_MODEL.encode(text)
    return embedding.astype(np.float32)

def store_knowledge(content: str, metadata: dict, cur):
    embedding = embed_text(content)
    cur.execute(
        "INSERT INTO documents (content, metadata, embedding) VALUES (?, ?, ?)",
        (content, json.dumps(metadata, ensure_ascii=False), embedding.tobytes())
    )

# 连接数据库
conn = sqlite3.connect("knowledge_base.db")
cur = conn.cursor()

# 先建表（如果不存在）
cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT,
    metadata TEXT,
    embedding BLOB
)
""")
conn.commit()

# 假设所有csv都在当前目录下
csv_files = glob.glob("ds/data/*.csv")  # 或指定文件夹路径，如 "data/*.csv"

print("找到的CSV文件：", csv_files)

batch_size = 100
count = 0
for csv_file in csv_files:
    print(f"正在导入：{csv_file}")
    df = pd.read_csv(csv_file, encoding="utf-8")
    print("字段名：", df.columns)
    for idx, row in df.iterrows():
        try:
            
            # 获取文件名（不含扩展名）作为省份
            province_from_file = os.path.splitext(os.path.basename(csv_file))[0]
            
            # 合并选课要求信息
            sginfo = str(row['Sg_info']) if pd.notna(row['Sg_info']) else ''
            sgxuanke = str(row['Sg_xuanke']) if pd.notna(row['Sg_xuanke']) else ''
            if sginfo and sgxuanke:
                xk_info = f"{sginfo}，{sgxuanke}"
            else:
                xk_info = sginfo or sgxuanke or ''
            # 合并类别信息
            level1 = str(row['Level1_name']) if pd.notna(row['Level1_name']) else ''
            level2 = str(row['Level2_name']) if pd.notna(row['Level2_name']) else ''
            level3 = str(row['Level3_name']) if pd.notna(row['Level3_name']) else ''
            category = '，'.join([x for x in [level1, level2, level3] if x and x != '无'])
            # 组织知识内容，增加省份信息
            content = (
                f"{row['Year']}年{row['School_name']}在省份代码为{row['Province']}（省份：{province_from_file}）的高考{row['Type']}招生，专业为{row['Spname']}，"
                f"最高分{row['Max']}，最低分{row['Min']}，平均分{row['Average']}，最低分排位{row['Min_section']}。"
                f"批次：{row['Local_batch_name']}，招生类型：{row['Zslx_name']}。"
                f"选课要求信息：{xk_info}"
                f"类别：{category}"
            )
            # 组织元数据，增加省份信息
            meta = {
                "学校": row['School_name'],
                "学校代码": row['School'],
                "省份代码": row['Province'],
                "省份": province_from_file,
                "年份": int(row['Year']),
                "科类": row['Type'],
                "专业": row['Spname'],
                "最高分": row['Max'],
                "最低分": row['Min'],
                "平均分": row['Average'],
                "最低分排位": row['Min_section'],
                "批次": row['Local_batch_name'],
                "招生类型": row['Zslx_name'],
                "附加信息": row['Info'],
                "类别": category,
                "选课要求信息": xk_info,
            }
            store_knowledge(content, meta, cur)
            count += 1
            if count % batch_size == 0:
                conn.commit()
        except Exception as e:
            print(f"第{idx}行出错：{e}")
            traceback.print_exc()
            break

conn.commit()
cur.close()
conn.close()
print("全部导入完成！") 