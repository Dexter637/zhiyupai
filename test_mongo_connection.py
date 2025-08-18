import pymongo
import time

# 从配置文件导入MongoDB连接信息
try:
    from config import MONGO_URI, DATABASE_NAME
    print(f"使用配置文件中的MongoDB连接信息: {MONGO_URI}, 数据库: {DATABASE_NAME}")
except ImportError:
    # 如果无法导入配置文件，使用默认值
    MONGO_URI = "mongodb://localhost:27017/"
    DATABASE_NAME = "health_data"
    print(f"使用默认MongoDB连接信息: {MONGO_URI}, 数据库: {DATABASE_NAME}")

# 测试MongoDB连接
def test_mongo_connection():
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 尝试连接MongoDB
        client = pymongo.MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        
        # 尝试获取服务器信息（这会实际触发连接）
        server_info = client.server_info()
        
        # 记录结束时间
        end_time = time.time()
        
        print(f"成功连接到MongoDB服务器！")
        print(f"服务器版本: {server_info.get('version', '未知')}")
        print(f"连接耗时: {end_time - start_time:.2f} 秒")
        
        # 尝试访问数据库
        db = client[DATABASE_NAME]
        collections = db.list_collection_names()
        print(f"数据库 {DATABASE_NAME} 中的集合: {collections}")
        
        return True
        
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print(f"MongoDB连接超时: {str(e)}")
        print("请确认MongoDB服务是否正在运行，以及连接地址是否正确。")
    except pymongo.errors.ConnectionFailure as e:
        print(f"MongoDB连接失败: {str(e)}")
    except Exception as e:
        print(f"连接MongoDB时发生未知错误: {str(e)}")
    
    return False

# 运行测试
if __name__ == "__main__":
    print("开始测试MongoDB连接...")
    success = test_mongo_connection()
    print(f"连接测试{'成功' if success else '失败'}")
    
    # 如果连接失败，提供一些常见的解决方案
    if not success:
        print("\n可能的解决方案:")
        print("1. 确保MongoDB服务已启动")
        print("2. 检查MongoDB端口号是否正确（默认是27017）")
        print("3. 检查防火墙设置是否阻止了连接")
        print("4. 确认MongoDB服务是否允许远程连接（如果不是本地连接）")