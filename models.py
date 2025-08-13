# MongoDB数据模型
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId

# 导入配置
from config import MONGO_URI, DATABASE_NAME, COLLECTIONS

# 创建MongoDB客户端连接
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

class BaseModel:
    """基础模型类，提供通用的CRUD操作"""
    collection_name = None
    
    @classmethod
    def get_collection(cls):
        """获取集合对象"""
        if not cls.collection_name:
            raise ValueError(f"集合名称未定义: {cls.__name__}")
        return db[cls.collection_name]
    
    @classmethod
    def create(cls, data):
        """创建文档"""
        if '_id' not in data:
            data['created_at'] = datetime.now()
            data['updated_at'] = datetime.now()
        result = cls.get_collection().insert_one(data)
        return result.inserted_id
    
    @classmethod
    def find_by_id(cls, id):
        """通过ID查找文档"""
        if isinstance(id, str):
            id = ObjectId(id)
        return cls.get_collection().find_one({"_id": id})
    
    @classmethod
    def find(cls, query=None, sort=None, limit=0, skip=0):
        """查找多个文档"""
        query = query or {}
        cursor = cls.get_collection().find(query)
        
        if sort:
            cursor = cursor.sort(sort)
        
        if skip:
            cursor = cursor.skip(skip)
        
        if limit:
            cursor = cursor.limit(limit)
        
        return list(cursor)
    
    @classmethod
    def update(cls, id, data):
        """更新文档"""
        if isinstance(id, str):
            id = ObjectId(id)
        
        data['updated_at'] = datetime.now()
        return cls.get_collection().update_one(
            {"_id": id},
            {"$set": data}
        )
    
    @classmethod
    def delete(cls, id):
        """删除文档"""
        if isinstance(id, str):
            id = ObjectId(id)
        return cls.get_collection().delete_one({"_id": id})

class WearableData(BaseModel):
    """可穿戴设备数据模型"""
    collection_name = COLLECTIONS['wearable_data']
    
    @classmethod
    def find_by_user(cls, user_id, data_type=None, start_time=None, end_time=None, limit=100):
        """查找用户的可穿戴设备数据"""
        query = {"user_id": user_id}
        
        if data_type:
            query["data_type"] = data_type
        
        if start_time or end_time:
            query["timestamp"] = {}
            
            if start_time:
                query["timestamp"]["$gte"] = start_time
            
            if end_time:
                query["timestamp"]["$lte"] = end_time
        
        return cls.find(
            query=query,
            sort=[("timestamp", -1)],
            limit=limit
        )

class HeartRateAlert(BaseModel):
    """心率异常警报模型"""
    collection_name = COLLECTIONS['heart_rate_alerts']
    
    @classmethod
    def create_alert(cls, user_id, heart_rate, severity, message, timestamp=None):
        """创建心率警报"""
        alert_data = {
            "user_id": user_id,
            "heart_rate": heart_rate,
            "severity": severity,  # 'low', 'medium', 'high'
            "message": message,
            "timestamp": timestamp or datetime.now(),
            "is_read": False
        }
        return cls.create(alert_data)
    
    @classmethod
    def get_unread_alerts(cls, user_id):
        """获取用户未读警报"""
        return cls.find(
            query={"user_id": user_id, "is_read": False},
            sort=[("timestamp", -1)]
        )

class ActivityRecommendation(BaseModel):
    """活动推荐模型"""
    collection_name = COLLECTIONS['activity_recommendations']
    
    @classmethod
    def create_recommendation(cls, user_id, activity_type, duration, intensity, reason):
        """创建活动推荐"""
        recommendation_data = {
            "user_id": user_id,
            "activity_type": activity_type,
            "duration": duration,  # 分钟
            "intensity": intensity,  # 'low', 'medium', 'high'
            "reason": reason,
            "timestamp": datetime.now(),
            "is_completed": False
        }
        return cls.create(recommendation_data)
    
    @classmethod
    def get_recent_recommendations(cls, user_id, limit=5):
        """获取用户最近的活动推荐"""
        return cls.find(
            query={"user_id": user_id},
            sort=[("timestamp", -1)],
            limit=limit
        )

class UserProfile(BaseModel):
    """用户档案模型"""
    collection_name = COLLECTIONS['user_profiles']
    
    @classmethod
    def find_by_user_id(cls, user_id):
        """通过用户ID查找用户档案"""
        return cls.get_collection().find_one({"user_id": user_id})