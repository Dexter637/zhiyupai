# 创建MongoDB索引脚本
import pymongo
from config import MONGO_URI, DATABASE_NAME, COLLECTIONS

def create_indexes():
    """为MongoDB集合创建索引以提高查询性能"""
    try:
        # 连接到MongoDB
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        
        print(f"正在为数据库 {DATABASE_NAME} 创建索引...")
        
        # 为可穿戴设备数据创建索引
        wearable_collection = db[COLLECTIONS['wearable_data']]
        wearable_collection.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
        wearable_collection.create_index([("device_id", pymongo.ASCENDING)])
        wearable_collection.create_index([("data_type", pymongo.ASCENDING)])
        print(f"✅ 已为 {COLLECTIONS['wearable_data']} 集合创建索引")
        
        # 为心率警报创建索引
        alerts_collection = db[COLLECTIONS['heart_rate_alerts']]
        alerts_collection.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
        alerts_collection.create_index([("severity", pymongo.ASCENDING)])
        alerts_collection.create_index([("is_read", pymongo.ASCENDING)])
        print(f"✅ 已为 {COLLECTIONS['heart_rate_alerts']} 集合创建索引")
        
        # 为睡眠记录创建索引
        sleep_collection = db[COLLECTIONS['sleep_records']]
        sleep_collection.create_index([("user_id", pymongo.ASCENDING), ("date", pymongo.DESCENDING)])
        print(f"✅ 已为 {COLLECTIONS['sleep_records']} 集合创建索引")
        
        # 为活动推荐创建索引
        recommendations_collection = db[COLLECTIONS['activity_recommendations']]
        recommendations_collection.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
        recommendations_collection.create_index([("is_completed", pymongo.ASCENDING)])
        print(f"✅ 已为 {COLLECTIONS['activity_recommendations']} 集合创建索引")
        
        # 为用户档案创建索引
        user_profiles_collection = db[COLLECTIONS['user_profiles']]
        user_profiles_collection.create_index("user_id", unique=True)
        print(f"✅ 已为 {COLLECTIONS['user_profiles']} 集合创建索引")
        
        print("\n✅ 所有索引创建完成!")
        
    except Exception as e:
        print(f"❌ 创建索引时出错: {e}")
    
    finally:
        if 'client' in locals():
            client.close()
            print("MongoDB连接已关闭")

if __name__ == "__main__":
    create_indexes()