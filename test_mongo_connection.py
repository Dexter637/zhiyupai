import pymongo
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MongoDB连接测试脚本
用于测试不同用户的MongoDB连接凭证
"""

import sys
import pymongo

# 测试不同的连接配置
test_configs = [
    {
        "name": "管理员用户 (admin) with authSource=admin",
        "uri": "mongodb://admin:admin123@localhost:27017/health_db?authSource=admin",
        "db_name": "health_db"
    },
    {
        "name": "普通用户 (health_user)",
        "uri": "mongodb://health_user:health_password@localhost:27017/health_db",
        "db_name": "health_db"
    },
    {
        "name": "管理员用户 (admin) without authSource",
        "uri": "mongodb://admin:admin123@localhost:27017/health_db",
        "db_name": "health_db"
    },
    {
        "name": "无认证连接",
        "uri": "mongodb://localhost:27017/health_db",
        "db_name": "health_db"
    }
]

def test_connection(config):
    """测试MongoDB连接"""
    print(f"\n测试连接: {config['name']}")
    print(f"连接URI: {config['uri']}")
    print(f"数据库名称: {config['db_name']}")
    
    try:
        # 尝试连接MongoDB
        client = pymongo.MongoClient(config['uri'], serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[config['db_name']]
        
        # 测试数据库操作
        collections = db.list_collection_names()
        print(f"✅ 连接成功!")
        print(f"  可用集合: {collections}")
        
        # 关闭连接
        client.close()
        return True
    except pymongo.errors.OperationFailure as e:
        if 'Authentication failed' in str(e):
            print(f"❌ 认证失败: {str(e)}")
            print("  请检查用户名和密码是否正确")
        else:
            print(f"❌ 操作失败: {str(e)}")
        return False
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print(f"❌ 服务器选择超时: {str(e)}")
        print("  请检查MongoDB服务是否正在运行")
        return False
    except Exception as e:
        print(f"❌ 连接失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("MongoDB连接测试工具")
    print("="*60)
    
    # 测试所有连接配置
    success_count = 0
    for config in test_configs:
        if test_connection(config):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"测试结果: 成功 {success_count}/{len(test_configs)} 个连接配置")
    
    # 提供一些常见的解决方案
    if success_count == 0:
        print("\n可能的解决方案:")
        print("1. 确保MongoDB容器正在运行: docker ps")
        print("2. 检查MongoDB容器日志: docker logs mongodb-container")
        print("3. 确认认证信息是否正确")
        print("4. 尝试重启MongoDB容器: docker restart mongodb-container")
    
    print("="*60)
    
    # 根据测试结果设置退出码
    sys.exit(0 if success_count > 0 else 1)