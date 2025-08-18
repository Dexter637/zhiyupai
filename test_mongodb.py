# MongoDB连接测试脚本
import pymongo
import sys
import os
import time
from datetime import datetime

# 导入配置
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MONGO_URI, DATABASE_NAME, WEARABLE_DATA_COLLECTION, HEART_RATE_ALERTS_COLLECTION, \
    SLEEP_RECORDS_COLLECTION, ACTIVITY_RECOMMENDATIONS_COLLECTION, USER_PROFILES_COLLECTION, \
    CONNECTION_TIMEOUT_MS, MAX_POOL_SIZE, MIN_POOL_SIZE

def test_mongodb_connection():
    """测试MongoDB连接是否正常"""
    try:
        # 尝试连接MongoDB
        client = pymongo.MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=CONNECTION_TIMEOUT_MS,
            maxPoolSize=MAX_POOL_SIZE,
            minPoolSize=MIN_POOL_SIZE
        )
        
        # 检查连接是否成功
        client.admin.command('ping')
        print("✅ MongoDB连接成功!")
        print(f"服务器信息: {client.server_info()['version']}")
        
        # 获取数据库
        db = client[DATABASE_NAME]
        print(f"✅ 成功连接到数据库: {DATABASE_NAME}")
        
        # 测试集合操作
        collection_name = WEARABLE_DATA_COLLECTION
        collection = db[collection_name]
        
        # 插入测试文档
        test_doc = {
            "test": True,
            "timestamp": datetime.now(),
            "message": "这是一个测试文档",
            "test_id": "test_" + str(int(time.time()))
        }
        
        result = collection.insert_one(test_doc)
        print(f"✅ 成功插入测试文档，ID: {result.inserted_id}")
        
        # 查询测试文档
        found_doc = collection.find_one({"_id": result.inserted_id})
        print(f"✅ 成功查询测试文档: {found_doc}")
        
        # 删除测试文档
        collection.delete_one({"_id": result.inserted_id})
        print("✅ 成功删除测试文档")
        
        # 列出所有集合
        print("\n当前数据库中的集合:")
        collections = db.list_collection_names()
        if collections:
            for coll in collections:
                print(f"  - {coll}")
        else:
            print("  (数据库中没有集合)")
        
        return client
    
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print(f"❌ MongoDB连接超时: {e}")
        print("请检查MongoDB服务是否正在运行，以及连接URI是否正确")
        return None
    
    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB连接失败: {e}")
        return None
    
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return None

def create_indexes(client=None):
    """为集合创建索引以提高查询性能"""
    try:
        # 如果没有提供客户端，创建一个新的连接
        if client is None:
            client = pymongo.MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=CONNECTION_TIMEOUT_MS,
                maxPoolSize=MAX_POOL_SIZE,
                minPoolSize=MIN_POOL_SIZE
            )
        
        db = client[DATABASE_NAME]
        
        # 为可穿戴设备数据创建索引
        wearable_collection = db[WEARABLE_DATA_COLLECTION]
        wearable_collection.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
        wearable_collection.create_index([("device_id", pymongo.ASCENDING)])
        wearable_collection.create_index([("data_type", pymongo.ASCENDING)])
        print(f"✅ 为 {WEARABLE_DATA_COLLECTION} 创建索引成功")
        
        # 为心率警报创建索引
        alerts_collection = db[HEART_RATE_ALERTS_COLLECTION]
        alerts_collection.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
        alerts_collection.create_index([("severity", pymongo.ASCENDING)])
        print(f"✅ 为 {HEART_RATE_ALERTS_COLLECTION} 创建索引成功")
        
        # 为睡眠记录创建索引
        sleep_collection = db[SLEEP_RECORDS_COLLECTION]
        sleep_collection.create_index([("user_id", pymongo.ASCENDING), ("date", pymongo.DESCENDING)])
        sleep_collection.create_index([("sleep_quality", pymongo.ASCENDING)])
        print(f"✅ 为 {SLEEP_RECORDS_COLLECTION} 创建索引成功")
        
        # 为活动推荐创建索引
        activity_collection = db[ACTIVITY_RECOMMENDATIONS_COLLECTION]
        activity_collection.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
        activity_collection.create_index([("activity_type", pymongo.ASCENDING)])
        print(f"✅ 为 {ACTIVITY_RECOMMENDATIONS_COLLECTION} 创建索引成功")
        
        # 为用户档案创建索引
        user_collection = db[USER_PROFILES_COLLECTION]
        user_collection.create_index([("user_id", pymongo.ASCENDING)], unique=True)
        # 创建phone字段的唯一索引，使用sparse=True允许null值
        user_collection.create_index([("phone", pymongo.ASCENDING)], unique=True, sparse=True)
        print(f"✅ 为 {USER_PROFILES_COLLECTION} 创建索引成功")
        
        print("✅ 成功创建所有索引")
        return True
    
    except Exception as e:
        print(f"❌ 创建索引时出错: {e}")
        return False
    
    finally:
        # 如果我们创建了新的客户端连接，则关闭它
        if client is not None and 'client' not in locals():
            client.close()

def test_collections_exist(client):
    """测试项目所需的集合是否存在"""
    if not client:
        return False
    
    try:
        # 获取数据库
        db = client[DATABASE_NAME]
        
        # 检查所需集合
        required_collections = [
            WEARABLE_DATA_COLLECTION,
            HEART_RATE_ALERTS_COLLECTION,
            SLEEP_RECORDS_COLLECTION,
            ACTIVITY_RECOMMENDATIONS_COLLECTION,
            USER_PROFILES_COLLECTION
        ]
        
        existing_collections = db.list_collection_names()
        
        print("\n🔍 检查项目所需集合:")
        all_exist = True
        
        for collection in required_collections:
            if collection in existing_collections:
                print(f"  ✅ {collection} - 已存在")
            else:
                print(f"  ❌ {collection} - 不存在")
                all_exist = False
        
        if not all_exist:
            print("\n💡 提示: 不存在的集合将在首次插入数据时自动创建")
        
        return all_exist
    except Exception as e:
        print(f"❌ 集合检查失败: {str(e)}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n🔄 开始MongoDB数据库测试...\n")
    
    # 测试连接
    client = test_mongodb_connection()
    if not client:
        print("\n❌ MongoDB连接测试失败，无法继续后续测试")
        return False
    
    # 测试集合是否存在
    collections_exist = test_collections_exist(client)
    
    # 创建索引
    print("\n=== 创建数据库索引 ===")
    indexes_created = create_indexes(client)
    
    # 关闭连接
    client.close()
    print("MongoDB连接已关闭")
    
    # 总结
    print("\n📝 测试总结:")
    print(f"  MongoDB连接: {'✅ 成功' if client else '❌ 失败'}")
    print(f"  集合检查: {'✅ 所有集合已存在' if collections_exist else '⚠️ 部分集合不存在'}")
    print(f"  索引创建: {'✅ 成功' if indexes_created else '❌ 失败'}")
    
    if client:
        print("\n✅ MongoDB数据库已准备就绪!")
        return True
    else:
        print("\n⚠️ MongoDB数据库测试发现问题，请检查配置和连接")
        return False


if __name__ == "__main__":
    run_all_tests()