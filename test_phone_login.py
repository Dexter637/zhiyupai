#!/usr/bin/env python3
"""
测试手机号登录功能
"""

import os
import sys
import django

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'zhiyupai_project.settings')
django.setup()

from django.contrib.auth import get_user_model, authenticate
from models import UserProfile

def test_phone_login():
    """测试手机号登录功能"""
    User = get_user_model()
    
    print("=== 测试手机号登录功能 ===")
    
    # 创建一个测试用户
    try:
        import random
        username = f"testuser{random.randint(1000, 9999)}"
        phone_number = f"138{random.randint(10000000, 99999999)}"
        user = User.objects.create_user(
            username=username,
            password="testpassword123",
            email=f"{username}@example.com"
        )
        print(f"✓ 创建测试用户: {user.username}")
        
        # 创建用户档案（包含手机号）
        profile_data = {
            "user_id": str(user.id),
            "phone": phone_number,
            "first_name": "Test",
            "last_name": "User"
        }
        
        # 使用UserProfile的create方法
        profile = UserProfile.create(profile_data)
        print(f"✓ 创建用户档案，手机号: {profile_data['phone']}")
        
        # 测试用户名登录
        print("\n1. 测试用户名登录:")
        auth_user = authenticate(username="testuser", password="testpassword123")
        if auth_user:
            print(f"✓ 用户名登录成功: {auth_user.username}")
        else:
            print("✗ 用户名登录失败")
        
        # 测试手机号登录
        print("\n2. 测试手机号登录:")
        auth_user = authenticate(phone=phone_number, password="testpassword123")
        if auth_user:
            print(f"✓ 手机号登录成功: {auth_user.username}")
        else:
            print("✗ 手机号登录失败")
        
        # 测试错误密码
        print("\n3. 测试错误密码:")
        auth_user = authenticate(phone=phone_number, password="wrongpassword")
        if auth_user:
            print("✗ 错误密码竟然登录成功（不应该发生）")
        else:
            print("✓ 错误密码登录失败（正常）")
        
        # 测试不存在的手机号
        print("\n4. 测试不存在的手机号:")
        auth_user = authenticate(phone="13999999999", password="testpassword123")
        if auth_user:
            print("✗ 不存在的手机号竟然登录成功（不应该发生）")
        else:
            print("✓ 不存在的手机号登录失败（正常）")
        
        # 清理测试数据
        user.delete()
        # 手动删除用户档案
        profile_collection = UserProfile.get_collection()
        profile_collection.delete_one({"user_id": str(user.id)})
        print("\n✓ 清理测试数据完成")
        
    except Exception as e:
        print(f"✗ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_phone_login()