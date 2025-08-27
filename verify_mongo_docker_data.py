#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证MongoDB Docker容器中的数据是否正确迁移"""

import sys
from pymongo import MongoClient

# 设置中文字符支持
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def verify_mongo_docker_data():
    """验证Docker中的MongoDB容器数据"""
    print("\n===== MongoDB Docker容器数据验证工具 =====\n")
    
    try:
        # 连接到Docker中的MongoDB（无认证连接）
        print("正在连接到Docker中的MongoDB容器...")
        client = MongoClient("mongodb://localhost:27017/health_db", serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ 成功连接到Docker中的MongoDB容器")
        
        # 选择数据库
        db = client.health_db
        
        # 列出所有集合
        collections = db.list_collection_names()
        print(f"\n目标数据库中的集合列表: {collections}")
        
        # 检查每个集合的记录数
        print("\n各集合记录数量:")
        for collection_name in collections:
            collection = db[collection_name]
            count = collection.count_documents({})
            print(f"  - {collection_name}: {count} 条记录")
            
            # 如果集合有记录，显示前2条文档的前几个字段
            if count > 0:
                print("    部分示例文档:")
                for doc in collection.find().limit(2):
                    # 只显示前5个字段
                    preview = {k: v for i, (k, v) in enumerate(doc.items()) if i < 5}
                    print(f"    {preview}")
        
        print("\n✅ 数据验证完成！")
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ 连接MongoDB失败或验证出错: {str(e)}")
        return False

if __name__ == "__main__":
    success = verify_mongo_docker_data()
    sys.exit(0 if success else 1)