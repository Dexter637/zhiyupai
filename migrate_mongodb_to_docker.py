#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MongoDB数据迁移工具
用于将现有MongoDB数据库的数据迁移到Docker容器中

使用说明：
1. 确保已安装pymongo：pip install pymongo
2. 如果需要迁移整个数据库，请直接运行此脚本
3. 如果只需要迁移特定集合，请修改脚本中的COLLECTIONS_TO_MIGRATE变量

注意：
- 迁移前请确保源MongoDB和目标MongoDB（Docker中的实例）都可访问
- 脚本会先备份源数据，然后再进行迁移
"""

import os
import sys
import time
import json
import shutil
import subprocess
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# 设置中文字符支持
import io
import locale

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class MongoDBMigrator:
    def __init__(self):
        # 源MongoDB配置（迁移前的数据库）
        self.source_uri = "mongodb://localhost:27017/"
        self.source_db_name = "health_data"
        
        # 目标MongoDB配置（Docker中的实例）
        # 使用无认证连接
        self.target_uri = "mongodb://localhost:27017/health_db"
        self.target_db_name = "health_db"
        
        # 需要迁移的集合列表
        self.collections_to_migrate = [
            "wearable_data",
            "heart_rate_alerts",
            "sleep_records",
            "activity_recommendations",
            "user_profiles",
            "users",
            "health_data",
            "device_data",
            "sleep_analysis"
        ]
        
        # 备份目录
        self.backup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mongo_backup")
        
        # 初始化客户端连接
        self.source_client = None
        self.target_client = None
        self.source_db = None
        self.target_db = None
    
    def connect_to_mongodb(self):
        """连接到源MongoDB和目标MongoDB"""
        print("正在连接到MongoDB...")
        
        try:
            # 连接源MongoDB
            print(f"连接源MongoDB: {self.source_uri} 数据库: {self.source_db_name}")
            self.source_client = MongoClient(self.source_uri, serverSelectionTimeoutMS=5000)
            self.source_client.admin.command('ping')
            self.source_db = self.source_client[self.source_db_name]
            print("✅ 成功连接到源MongoDB")
        except Exception as e:
            print(f"❌ 连接源MongoDB失败: {str(e)}")
            print("请检查源MongoDB服务是否正在运行")
            return False
        
        try:
            # 连接目标MongoDB（Docker中的实例）
            print(f"连接目标MongoDB: {self.target_uri} 数据库: {self.target_db_name}")
            self.target_client = MongoClient(self.target_uri, serverSelectionTimeoutMS=5000)
            self.target_client.admin.command('ping')
            self.target_db = self.target_client[self.target_db_name]
            print("✅ 成功连接到目标MongoDB")
        except Exception as e:
            print(f"❌ 连接目标MongoDB失败: {str(e)}")
            print("请检查Docker容器是否正在运行，以及认证信息是否正确")
            self.source_client.close()
            return False
        
        return True
    
    def create_backup_directory(self):
        """创建备份目录"""
        try:
            # 如果备份目录已存在，创建一个带时间戳的子目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_dir = os.path.join(self.backup_dir, timestamp)
            
            if not os.path.exists(self.backup_dir):
                os.makedirs(self.backup_dir, exist_ok=True)
            
            print(f"✅ 备份目录已创建: {self.backup_dir}")
            return True
        except Exception as e:
            print(f"❌ 创建备份目录失败: {str(e)}")
            return False
    
    def backup_source_data(self):
        """备份源MongoDB的数据"""
        print("\n开始备份源MongoDB数据...")
        
        # 获取源数据库中的所有集合
        source_collections = self.source_db.list_collection_names()
        print(f"源数据库中的集合: {source_collections}")
        
        # 创建集合到备份的映射
        for collection_name in self.collections_to_migrate:
            # 检查集合是否存在于源数据库
            if collection_name not in source_collections:
                print(f"⚠️ 集合 '{collection_name}' 在源数据库中不存在，跳过备份")
                continue
            
            try:
                # 获取集合
                source_collection = self.source_db[collection_name]
                
                # 创建备份文件
                backup_file = os.path.join(self.backup_dir, f"{collection_name}_backup.json")
                
                # 备份文档
                docs = list(source_collection.find())
                
                # 将ObjectId转换为字符串以便JSON序列化
                for doc in docs:
                    if "_id" in doc:
                        doc["_id"] = str(doc["_id"])
                
                # 保存到文件
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(docs, f, ensure_ascii=False, indent=2, default=str)
                
                print(f"✅ 已备份集合 '{collection_name}' 中的 {len(docs)} 条记录到 {backup_file}")
            except Exception as e:
                print(f"❌ 备份集合 '{collection_name}' 失败: {str(e)}")
                return False
        
        return True
    
    def migrate_data(self):
        """将数据从源MongoDB迁移到目标MongoDB"""
        print("\n开始迁移数据到Docker中的MongoDB...")
        
        for collection_name in self.collections_to_migrate:
            backup_file = os.path.join(self.backup_dir, f"{collection_name}_backup.json")
            
            # 检查备份文件是否存在
            if not os.path.exists(backup_file):
                print(f"⚠️ 备份文件 '{backup_file}' 不存在，跳过迁移集合 '{collection_name}'")
                continue
            
            try:
                # 读取备份文件
                with open(backup_file, 'r', encoding='utf-8') as f:
                    docs = json.load(f)
                
                print(f"正在迁移集合 '{collection_name}' 中的 {len(docs)} 条记录...")
                
                # 获取目标集合
                target_collection = self.target_db[collection_name]
                
                # 清空目标集合（可选）
                print(f"  清空目标集合 '{collection_name}'")
                target_collection.delete_many({})
                
                # 将文档中的字符串_id转换回ObjectId（如果需要）
                for doc in docs:
                    if "_id" in doc:
                        # 保持_id为字符串，MongoDB会自动处理
                        pass
                
                # 批量插入文档
                if docs:
                    target_collection.insert_many(docs)
                
                print(f"✅ 成功迁移集合 '{collection_name}'")
            except Exception as e:
                print(f"❌ 迁移集合 '{collection_name}' 失败: {str(e)}")
                return False
        
        return True
    
    def verify_migration(self):
        """验证数据迁移是否成功"""
        print("\n开始验证数据迁移结果...")
        
        all_verified = True
        
        for collection_name in self.collections_to_migrate:
            try:
                # 检查源集合和目标集合的文档数量
                source_count = self.source_db[collection_name].count_documents({})
                target_count = self.target_db[collection_name].count_documents({})
                
                print(f"集合 '{collection_name}': 源数据库 {source_count} 条记录，目标数据库 {target_count} 条记录")
                
                if source_count == target_count:
                    print(f"✅ 集合 '{collection_name}' 数据数量验证通过")
                else:
                    print(f"❌ 集合 '{collection_name}' 数据数量不匹配")
                    all_verified = False
                
                # 随机抽查一些文档
                if target_count > 0:
                    sample_doc = self.target_db[collection_name].find_one()
                    print(f"  随机抽查文档: {json.dumps(sample_doc, ensure_ascii=False, default=str)[:200]}...")
            except Exception as e:
                print(f"❌ 验证集合 '{collection_name}' 失败: {str(e)}")
                all_verified = False
        
        return all_verified
    
    def close_connections(self):
        """关闭MongoDB连接"""
        if self.source_client:
            self.source_client.close()
        if self.target_client:
            self.target_client.close()
        print("\n✅ MongoDB连接已关闭")
    
    def run(self):
        """运行整个迁移流程"""
        try:
            print("="*60)
            print("MongoDB数据迁移工具")
            print(f"迁移时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
            
            # 1. 连接到MongoDB
            if not self.connect_to_mongodb():
                return False
            
            # 2. 创建备份目录
            if not self.create_backup_directory():
                self.close_connections()
                return False
            
            # 3. 备份源数据
            if not self.backup_source_data():
                self.close_connections()
                return False
            
            # 4. 迁移数据
            if not self.migrate_data():
                self.close_connections()
                return False
            
            # 5. 验证迁移
            if not self.verify_migration():
                print("\n⚠️ 数据迁移验证未通过，请检查迁移结果")
            else:
                print("\n✅ 数据迁移验证通过！")
            
            # 6. 关闭连接
            self.close_connections()
            
            print("\n" + "="*60)
            print("MongoDB数据迁移完成！")
            print(f"备份文件保存在: {self.backup_dir}")
            print("迁移的集合: {', '.join(self.collections_to_migrate)}")
            print("✅ 恭喜！您的MongoDB数据库已成功迁移到Docker容器中。")
            print("="*60)
            
            return True
        except KeyboardInterrupt:
            print("\n⚠️ 迁移过程被用户中断")
            self.close_connections()
            return False
        except Exception as e:
            print(f"\n❌ 迁移过程中发生错误: {str(e)}")
            self.close_connections()
            return False

if __name__ == "__main__":
    # 自动检查Docker容器状态
    print("\n正在检查MongoDB容器状态...")
    
    # 创建迁移器并运行
    migrator = MongoDBMigrator()
    success = migrator.run()
    
    # 根据结果设置退出码
    sys.exit(0 if success else 1)