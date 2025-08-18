import pymongo
from datetime import datetime

# 从配置文件导入MongoDB连接信息
try:
    from config import MONGO_URI, DATABASE_NAME, COLLECTIONS
except ImportError:
    print("无法导入配置文件，使用默认值")
    MONGO_URI = "mongodb://localhost:27017/"
    DATABASE_NAME = "health_data"
    COLLECTIONS = {
        "user_profiles": "user_profiles",
        "wearable_data": "wearable_data"
    }

# 创建MongoDB客户端
client = pymongo.MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# 集合引用
user_profiles_collection = db[COLLECTIONS['user_profiles']]
wearable_data_collection = db[COLLECTIONS['wearable_data']]

# 通过用户名获取用户ID
def get_user_id_by_username(username):
    """通过用户名查找对应的用户ID"""
    profile = user_profiles_collection.find_one({"username": username})
    if profile:
        return profile.get("user_id")
    return None

# 通过用户名存储健康数据
def save_health_data_by_username(username, health_data):
    """通过用户名存储健康数据"""
    # 获取用户ID
    user_id = get_user_id_by_username(username)
    if not user_id:
        print(f"未找到用户: {username}")
        return False

    # 添加用户ID和时间戳到健康数据
    health_data['user_id'] = user_id
    health_data['timestamp'] = datetime.now()

    # 存储健康数据
    try:
        result = wearable_data_collection.insert_one(health_data)
        print(f"健康数据已保存，ID: {result.inserted_id}")
        return True
    except Exception as e:
        print(f"保存健康数据时出错: {str(e)}")
        return False

# 通过用户名查询健康数据
def get_health_data_by_username(username, data_type=None, limit=10):
    """通过用户名查询健康数据"""
    # 获取用户ID
    user_id = get_user_id_by_username(username)
    if not user_id:
        print(f"未找到用户: {username}")
        return []

    # 构建查询条件
    query = {"user_id": user_id}
    if data_type:
        query["data_type"] = data_type

    # 查询健康数据
    try:
        health_data = list(wearable_data_collection.find(query).sort("timestamp", -1).limit(limit))
        print(f"找到 {len(health_data)} 条健康数据")
        return health_data
    except Exception as e:
        print(f"查询健康数据时出错: {str(e)}")
        return []

# 测试函数
def test_username_health_data():
    # 测试用户名
    test_username = "testuser123"

    # 测试1: 通过用户名获取用户ID
    user_id = get_user_id_by_username(test_username)
    print(f"用户名 {test_username} 对应的用户ID: {user_id}")

    # 如果找到用户ID，继续测试
    if user_id:
        # 测试2: 通过用户名存储健康数据
        test_health_data = {
            "data_type": "heart_rate",
            "value": 72,
            "unit": "bpm"
        }
        save_result = save_health_data_by_username(test_username, test_health_data)
        print(f"通过用户名存储健康数据: {'成功' if save_result else '失败'}")

        # 测试3: 通过用户名查询健康数据
        health_data = get_health_data_by_username(test_username, "heart_rate")
        print(f"查询到的健康数据数量: {len(health_data)}")
        if health_data:
            print("第一条健康数据:")
            for key, value in health_data[0].items():
                print(f"  {key}: {value}")

# 运行测试
if __name__ == "__main__":
    print("开始测试通过用户名存储和查询健康数据...")
    test_username_health_data()
    client.close()