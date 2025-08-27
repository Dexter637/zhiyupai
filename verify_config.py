#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证MongoDB配置修改后的连接是否正常"""

import sys
import pymongo

# 设置中文字符支持
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 导入配置
from config import MONGO_URI, DATABASE_NAME, COLLECTIONS

def verify_config():
    """验证MongoDB配置是否正确"""
    print("\n===== MongoDB配置验证工具 =====\n")
    
    # 显示当前配置
    print(f"当前MongoDB配置:")
    print(f"  MONGO_URI: {MONGO_URI}")
    print(f"  DATABASE_NAME: {DATABASE_NAME}")
    print(f"  集合列表: {list(COLLECTIONS.values())}")
    
    try:
        # 尝试连接MongoDB
        print("\n正在尝试连接到MongoDB...")
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # 执行ping命令验证连接
        client.admin.command('ping')
        print("✅ 成功连接到MongoDB!")
        
        # 获取服务器信息
        server_info = client.server_info()
        print(f"  服务器版本: {server_info['version']}")
        
        # 选择数据库
        db = client[DATABASE_NAME]
        print(f"✅ 成功访问数据库: {DATABASE_NAME}")
        
        # 列出数据库中的所有集合
        collections = db.list_collection_names()
        print(f"✅ 数据库中的集合: {collections}")
        
        # 检查配置中定义的集合是否存在
        print("\n检查配置中定义的集合:")
        for col_name, col_value in COLLECTIONS.items():
            if col_value in collections:
                count = db[col_value].count_documents({})
                print(f"  ✅ {col_value}: 存在，{count} 条记录")
            else:
                print(f"  ⚠️ {col_value}: 不存在")
        
        # 尝试插入并查询一条测试数据
        print("\n测试数据操作:")
        test_collection = db["test_collection"]
        
        # 插入测试文档
        test_doc = {"test": True, "message": "配置验证测试"}
        result = test_collection.insert_one(test_doc)
        print(f"  ✅ 成功插入测试文档，ID: {result.inserted_id}")
        
        # 查询测试文档
        found_doc = test_collection.find_one({"_id": result.inserted_id})
        print(f"  ✅ 成功查询测试文档: {found_doc}")
        
        # 删除测试文档
        test_collection.delete_one({"_id": result.inserted_id})
        print("  ✅ 成功删除测试文档")
        
        # 关闭连接
        client.close()
        print("\n✅ 配置验证成功！应用程序现在可以正确连接到Docker中的MongoDB容器。")
        return True
        
    except Exception as e:
        print(f"❌ 配置验证失败: {str(e)}")
        print("\n可能的解决方案:")
        print("1. 确保MongoDB容器正在运行: docker ps")
        print("2. 检查config.py中的连接配置是否正确")
        print("3. 尝试重启MongoDB容器: docker restart mongodb-container")
        return False

if __name__ == "__main__":
    success = verify_config()
    sys.exit(0 if success else 1)