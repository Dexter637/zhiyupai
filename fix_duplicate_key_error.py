# 修复MongoDB重复键错误的脚本
import pymongo
from config import MONGO_URI, DATABASE_NAME, COLLECTIONS

def fix_duplicate_key_error():
    """检查并删除user_profiles集合中的email索引以解决重复键错误"""
    try:
        # 连接到MongoDB
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTIONS['user_profiles']]
        
        print(f"正在检查 {COLLECTIONS['user_profiles']} 集合的索引...")
        
        # 获取所有索引
        indexes = collection.index_information()
        
        # 打印当前索引
        print("当前索引:")
        for index_name, index_info in indexes.items():
            print(f"- {index_name}: {index_info}")
        
        # 检查是否存在email索引
        email_index_found = False
        for index_name, index_info in indexes.items():
            if 'email' in index_info['key'][0][0]:
                email_index_found = True
                print(f"找到email索引: {index_name}")
                
                # 删除索引
                collection.drop_index(index_name)
                print(f"已删除email索引: {index_name}")
                break
        
        if not email_index_found:
            print("未找到email索引")
        
        print("\n✅ 操作完成!")
        
    except Exception as e:
        print(f"❌ 操作时出错: {e}")
    
    finally:
        if 'client' in locals():
            client.close()
            print("MongoDB连接已关闭")

if __name__ == "__main__":
    fix_duplicate_key_error()