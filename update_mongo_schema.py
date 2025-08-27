#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新MongoDB数据库结构脚本
功能：
1. 为用户健康数据增加血糖、血压、血脂、食谱、身高体重等字段
2. 增加手机号作为登录凭证并确保唯一性
3. 修改数据库索引和集合结构以支持这些变更

注意：此脚本针对Docker环境中的MongoDB数据库
"""

import sys
import os
import json
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置（如果不需要修改配置文件，可以注释掉此行）
# from config import MONGO_URI, DATABASE_NAME, COLLECTIONS

# 数据库连接信息
MONGO_URI = "mongodb://localhost:27017/health_db"
DATABASE_NAME = "health_db"

# 集合名称映射
COLLECTIONS = {
    "wearable_data": "wearable_data",
    "heart_rate_alerts": "heart_rate_alerts",
    "sleep_records": "sleep_records",
    "activity_recommendations": "activity_recommendations",
    "user_profiles": "user_profiles"
}

class MongoSchemaUpdater:
    """MongoDB数据库结构更新工具类"""
    
    def __init__(self):
        """初始化数据库连接"""
        try:
            # 连接MongoDB数据库
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DATABASE_NAME]
            print(f"成功连接到MongoDB数据库: {DATABASE_NAME}")
        except Exception as e:
            print(f"连接MongoDB数据库失败: {str(e)}")
            sys.exit(1)
    
    def add_phone_unique_index(self):
        """为用户档案集合添加手机号唯一索引"""
        try:
            user_profiles_collection = self.db[COLLECTIONS["user_profiles"]]
            
            # 检查是否已存在phone索引
            indexes = user_profiles_collection.index_information()
            if "phone_1" not in indexes:
                # 创建phone唯一索引
                user_profiles_collection.create_index("phone", unique=True)
                print("成功创建用户档案集合的phone唯一索引")
            else:
                print("用户档案集合的phone索引已存在，跳过创建")
                
            return True
        except Exception as e:
            print(f"添加phone唯一索引失败: {str(e)}")
            return False
    
    def update_health_data_structure(self):
        """更新健康数据集合结构，支持新的数据类型"""
        try:
            # 不需要修改集合结构，MongoDB是文档型数据库，可以直接插入新字段
            # 但我们可以添加一些索引来优化查询性能
            wearable_data_collection = self.db[COLLECTIONS["wearable_data"]]
            
            # 检查并创建适合新健康数据类型的索引
            indexes = wearable_data_collection.index_information()
            
            # 为user_id和data_type创建复合索引，优化按用户和数据类型查询
            if "user_id_data_type_timestamp_-1" not in indexes:
                wearable_data_collection.create_index([
                    ("user_id", 1),
                    ("data_type", 1),
                    ("timestamp", -1)
                ])
                print("成功创建user_id+data_type+timestamp复合索引")
            else:
                print("user_id+data_type+timestamp索引已存在，跳过创建")
            
            print("健康数据集合结构已更新，现在支持血糖、血压、血脂、食谱、身高体重等数据类型")
            return True
        except Exception as e:
            print(f"更新健康数据集合结构失败: {str(e)}")
            return False
    
    def create_health_data_demo(self):
        """创建一些演示健康数据记录（可选）"""
        try:
            wearable_data_collection = self.db[COLLECTIONS["wearable_data"]]
            user_profiles_collection = self.db[COLLECTIONS["user_profiles"]]
            
            # 获取所有用户ID
            users = list(user_profiles_collection.find({}, {"user_id": 1}))
            if not users:
                print("没有找到用户记录，无法创建演示健康数据")
                return False
            
            # 为第一个用户创建一些演示数据
            demo_user_id = users[0]["user_id"]
            now = datetime.now()
            
            # 定义新的健康数据类型示例
            health_data_types = [
                {
                    "data_type": "blood_glucose",
                    "value": 5.2,
                    "unit": "mmol/L",
                    "measurement_time": now,
                    "note": "空腹血糖"
                },
                {
                    "data_type": "blood_pressure",
                    "systolic": 120,
                    "diastolic": 80,
                    "unit": "mmHg",
                    "measurement_time": now,
                    "note": "正常血压"
                },
                {
                    "data_type": "blood_lipids",
                    "total_cholesterol": 4.5,
                    "triglycerides": 1.8,
                    "hdl": 1.5,
                    "ldl": 2.5,
                    "unit": "mmol/L",
                    "measurement_time": now,
                    "note": "血脂检查"
                },
                {
                    "data_type": "diet",
                    "meals": [
                        {
                            "type": "早餐",
                            "foods": ["鸡蛋", "牛奶", "面包"],
                            "calories": 450
                        },
                        {
                            "type": "午餐",
                            "foods": ["米饭", "鸡肉", "蔬菜"],
                            "calories": 650
                        }
                    ],
                    "date": now.date(),
                    "total_calories": 1100
                },
                {
                    "data_type": "height_weight",
                    "height": 175,
                    "weight": 70,
                    "bmi": 22.9,
                    "unit": "cm/kg",
                    "measurement_time": now,
                    "note": "身高体重记录"
                }
            ]
            
            # 插入演示数据
            for data_type in health_data_types:
                # 构建完整的数据记录
                record = {
                    "user_id": demo_user_id,
                    "timestamp": now,
                    **data_type
                }
                
                try:
                    wearable_data_collection.insert_one(record)
                    print(f"已插入{data_type['data_type']}演示数据")
                except DuplicateKeyError:
                    print(f"{data_type['data_type']}演示数据已存在，跳过插入")
                except Exception as e:
                    print(f"插入{data_type['data_type']}演示数据失败: {str(e)}")
            
            return True
        except Exception as e:
            print(f"创建演示健康数据失败: {str(e)}")
            return False
    
    def update_login_api_support(self):
        """此函数用于提示需要更新登录API以支持手机号登录"""
        print("\n请手动更新user_app/views.py文件中的login_user函数，以支持手机号登录功能：")
        print("以下是建议的修改代码示例：")
        print('''
@api_view(['POST'])
@permission_classes([AllowAny])
def login_user(request):
    # 用户登录 - 支持用户名或手机号登录
    try:
        data = json.loads(request.body)
        
        # 验证必填字段
        required_fields = ['password']
        for field in required_fields:
            if field not in data:
                return JsonResponse(
                {"error": f"缺少必填字段: {field}"},
                status=400
            )
        
        # 检查是使用用户名还是手机号登录
        username = data.get('username')
        phone = data.get('phone')
        
        if not username and not phone:
            return JsonResponse(
                {"error": "必须提供用户名或手机号"},
                status=400
            )
        
        # 先尝试使用Django的authenticate（基于用户名）
        user = None
        if username:
            user = authenticate(username=username, password=data['password'])
        
        # 如果用户名验证失败，尝试使用手机号查找用户
        if not user and phone:
            # 从MongoDB查找手机号对应的用户档案
            from models import UserProfile
            profile = UserProfile.find_one({"phone": phone})
            if profile and "user_id" in profile:
                try:
                    from django.contrib.auth.models import User
                    user_obj = User.objects.get(id=profile["user_id"])
                    # 验证密码
                    if user_obj.check_password(data['password']):
                        user = user_obj
                except User.DoesNotExist:
                    pass
        
        if user is None:
            return JsonResponse(
                {"error": "用户名/手机号或密码错误"},
                status=401
            )
        
        # 生成JWT令牌
        from rest_framework_simplejwt.tokens import RefreshToken
        refresh = RefreshToken.for_user(user)
        
        # 获取用户档案
        from models import UserProfile
        profile = UserProfile.find_by_user_id(str(user.id))
        
        return JsonResponse({
            "status": "success",
            "message": "用户登录成功",
            "user_id": user.id,
            "profile_id": str(profile['_id']) if profile else None,
            "tokens": {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            }
        }, status=200)
        
    except Exception as e:
        return JsonResponse(
            {"error": f"登录用户时出错: {str(e)}"},
            status=500
        )
        ''')
        
    def update_health_data_api(self):
        """此函数用于提示需要更新健康数据API以支持按当前用户返回数据"""
        print("\n请手动更新health_app/views.py文件中的API函数，以支持按当前登录用户返回健康数据：")
        print("以下是建议的修改代码示例：")
        print('''
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_health_data(request, data_type=None):
    # 获取当前登录用户的健康数据
    try:
        # 获取当前登录用户ID
        user_id = str(request.user.id)
        
        # 获取查询参数
        start_time = request.query_params.get('start_time')
        end_time = request.query_params.get('end_time')
        limit = int(request.query_params.get('limit', 100))
        
        # 构建查询条件
        query = {"user_id": user_id}
        if data_type:
            query["data_type"] = data_type
        
        # 检索数据
        from models import WearableData
        results = WearableData.find_by_user(
            user_id=user_id,
            data_type=data_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return Response({
            "status": "success",
            "count": len(results),
            "data": results
        })
        
    except Exception as e:
        return Response(
            {"error": f"获取健康数据时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        ''')
    
    def run_all_updates(self):
        """运行所有更新操作"""
        print("开始更新MongoDB数据库结构...")
        
        # 添加phone唯一索引
        self.add_phone_unique_index()
        
        # 更新健康数据集合结构
        self.update_health_data_structure()
        
        # 可选：创建演示健康数据
        create_demo = input("是否创建演示健康数据？(y/n): ").lower() == 'y'
        if create_demo:
            self.create_health_data_demo()
        
        # 提示更新登录API
        self.update_login_api_support()
        
        # 提示更新健康数据API
        self.update_health_data_api()
        
        print("\nMongoDB数据库结构更新完成！")
        print("请根据提示手动更新相关API代码以完成所有功能修改。")

if __name__ == "__main__":
    """主函数入口"""
    try:
        updater = MongoSchemaUpdater()
        updater.run_all_updates()
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        print(f"执行更新过程中发生错误: {str(e)}")
    finally:
        # 关闭数据库连接
        if 'updater' in locals() and hasattr(updater, 'client'):
            updater.client.close()
            print("已关闭MongoDB数据库连接")