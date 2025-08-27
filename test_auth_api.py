#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试注册和登录API是否能够正确连接到Docker中的MongoDB容器"""

import sys
import requests
import json
import uuid
import random

# 设置中文字符支持
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# API基本URL（根据URL配置，用户应用的API路径应该是/api/users）
BASE_URL = "http://localhost:8000/api/users"

# 随机生成用户名，避免冲突，只包含英文和数字
def generate_username():
    suffix = uuid.uuid4().hex[:6]
    return f"testuser{suffix}"

TEST_USERNAME = generate_username()
TEST_PASSWORD = "Test123."
TEST_PHONE = f"138{random.randint(10000000, 99999999)}"

class AuthAPITester:
    """认证API测试工具"""
    
    def __init__(self, base_url):
        self.base_url = base_url
        self.access_token = None
        self.refresh_token = None
        self.user_id = None
        self.profile_id = None
        
    def register_user(self, username, password, phone):
        """测试用户注册API"""
        print(f"\n===== 测试用户注册 =====")
        print(f"注册用户: {username}")
        
        url = f"{self.base_url}/register/"
        data = {
            "username": username,
            "password": password,
            "phone": phone,
            "age": 30,
            "gender": "male"
        }
        
        try:
            response = requests.post(url, json=data, timeout=10)
            response_data = response.json()
            
            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
            
            if response.status_code == 201:
                # 保存令牌信息
                self.access_token = response_data.get("tokens", {}).get("access")
                self.refresh_token = response_data.get("tokens", {}).get("refresh")
                self.user_id = response_data.get("user_id")
                self.profile_id = response_data.get("profile_id")
                print("✅ 用户注册成功！")
                return True
            else:
                print(f"❌ 用户注册失败: {response_data.get('error', '未知错误')}")
                return False
        except Exception as e:
            print(f"❌ 注册请求失败: {str(e)}")
            return False
    
    def login_user(self, username, password):
        """测试用户登录API"""
        print(f"\n===== 测试用户登录 =====")
        print(f"登录用户: {username}")
        
        url = f"{self.base_url}/login/"
        data = {
            "username": username,
            "password": password
        }
        
        try:
            response = requests.post(url, json=data, timeout=10)
            response_data = response.json()
            
            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
            
            if response.status_code == 200:
                # 保存令牌信息
                self.access_token = response_data.get("tokens", {}).get("access")
                self.refresh_token = response_data.get("tokens", {}).get("refresh")
                self.user_id = response_data.get("user_id")
                self.profile_id = response_data.get("profile_id")
                print("✅ 用户登录成功！")
                return True
            else:
                print(f"❌ 用户登录失败: {response_data.get('error', '未知错误')}")
                return False
        except Exception as e:
            print(f"❌ 登录请求失败: {str(e)}")
            return False
    
    def get_user_profile(self):
        """测试获取用户档案API"""
        print(f"\n===== 测试用户档案获取 =====")
        
        if not self.access_token or not self.user_id:
            print("❌ 未登录或用户ID未知，无法获取用户档案")
            return False
        
        # 打印调试信息
        print(f"用户ID: {self.user_id}")
        print(f"Access Token长度: {len(self.access_token) if self.access_token else 0}")
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        print(f"请求头(Authorization前100个字符): {headers['Authorization'][:100] if self.access_token else '无'}")
        
        # 发送GET请求获取用户档案 - 尝试两个可能的URL格式
        urls_to_try = [
            f"{self.base_url}/profile/",  # 不带用户ID的版本
            f"{self.base_url}/profile/{self.user_id}/"  # 带用户ID的版本
        ]
        
        for url in urls_to_try:
            print(f"尝试请求URL: {url}")
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                print(f"响应状态码: {response.status_code}")
                print(f"响应头: {response.headers}")
                print(f"响应内容长度: {len(response.text)}")
                print(f"原始响应内容: {response.text[:500]}")  # 限制输出长度
                
                # 如果状态码是200，则尝试解析JSON响应
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        print(f"响应内容(JSON): {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                        print("✅ 获取用户档案成功！")
                        return True
                    except json.JSONDecodeError:
                        print(f"❌ 响应不是有效的JSON格式: {response.text}")
                        # 继续尝试下一个URL
                        continue
                elif response.status_code == 401 or response.status_code == 403:
                    print(f"❌ 认证失败: {response.status_code}")
                    print(f"原始响应内容: {response.text}")
                    # 继续尝试下一个URL
                    continue
                elif response.status_code == 404:
                    print(f"❌ URL不存在: {response.status_code}")
                    # 继续尝试下一个URL
                    continue
                else:
                    print(f"❌ 请求失败，状态码: {response.status_code}")
                    print(f"原始响应内容: {response.text}")
                    # 继续尝试下一个URL
                    continue
            except Exception as e:
                print(f"❌ 获取用户档案请求失败: {str(e)}")
                # 继续尝试下一个URL
                continue
        
        # 如果所有URL都尝试失败
        print("❌ 所有URL尝试均失败")
        return False
    
    def run_full_test(self, username, password, phone):
        """运行完整测试流程"""
        print("\n===== 开始认证API完整测试 =====")
        print(f"测试环境: {self.base_url}")
        
        # 测试注册
        register_success = self.register_user(username, password, phone)
        if not register_success:
            print("\n❌ 测试失败：用户注册失败")
            return False
        
        # 测试登录（使用刚注册的用户）
        login_success = self.login_user(username, password)
        if not login_success:
            print("\n❌ 测试失败：用户登录失败")
            return False
        
        # 测试获取用户档案
        profile_success = self.get_user_profile()
        if not profile_success:
            print("\n❌ 测试失败：获取用户档案失败")
            return False
        
        print("\n✅ 所有测试通过！认证API能够正确连接到Docker中的MongoDB容器。")
        print(f"\n测试摘要：")
        print(f"- 用户名: {username}")
        print(f"- 用户ID: {self.user_id}")
        print(f"- 档案ID: {self.profile_id}")
        print(f"- 访问令牌: {'已获取' if self.access_token else '未获取'}")
        return True

if __name__ == "__main__":
    # 随机生成手机号
    import random
    TEST_PHONE = f"138{random.randint(10000000, 99999999)}"
    
    # 创建测试器
    tester = AuthAPITester(BASE_URL)
    
    # 运行测试
    success = tester.run_full_test(TEST_USERNAME, TEST_PASSWORD, TEST_PHONE)
    
    # 根据测试结果设置退出码
    sys.exit(0 if success else 1)