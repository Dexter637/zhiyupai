# MongoDB配置文件

# MongoDB连接信息
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "health_data"

# 集合名称
COLLECTIONS = {
    "wearable_data": "wearable_data",
    "heart_rate_alerts": "heart_rate_alerts",
    "sleep_records": "sleep_records",
    "activity_recommendations": "activity_recommendations",
    "user_profiles": "user_profiles"
}

# 单独的集合名称变量（用于直接导入）
WEARABLE_DATA_COLLECTION = COLLECTIONS["wearable_data"]
HEART_RATE_ALERTS_COLLECTION = COLLECTIONS["heart_rate_alerts"]
SLEEP_RECORDS_COLLECTION = COLLECTIONS["sleep_records"]
ACTIVITY_RECOMMENDATIONS_COLLECTION = COLLECTIONS["activity_recommendations"]
USER_PROFILES_COLLECTION = COLLECTIONS["user_profiles"]

# 数据库连接超时设置（毫秒）
CONNECTION_TIMEOUT_MS = 5000

# 连接池设置
MAX_POOL_SIZE = 50
MIN_POOL_SIZE = 10