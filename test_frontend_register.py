import requests
import json

# 测试数据
test_data = {
    "username": "testuser123",
    "phone": "13812345678",
    "password": "Test123."
}

# 模拟前端fetch请求
try:
    # 设置请求头，与前端保持一致
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # 发送POST请求到注册API
    response = requests.post(
        "http://127.0.0.1:8000/api/users/register/",
        headers=headers,
        data=json.dumps(test_data)
    )
    
    # 打印响应状态码
    print(f"响应状态码: {response.status_code}")
    
    # 尝试解析JSON响应
    try:
        response_json = response.json()
        print(f"响应内容: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
    except json.JSONDecodeError:
        print(f"无法解析响应为JSON: {response.text}")
        
    # 检查是否成功
    if response.status_code == 201:
        print("测试成功！注册API正常工作")
    else:
        print(f"测试失败！状态码: {response.status_code}")
        
except Exception as e:
    print(f"请求发生异常: {str(e)}")

# 尝试使用不同参数组合进行测试
try:
    # 测试参数不完整的情况
    incomplete_data = {
        "username": "testuser123",
        # 缺少phone和password
    }
    
    response = requests.post(
        "http://127.0.0.1:8000/api/users/register/",
        headers=headers,
        data=json.dumps(incomplete_data)
    )
    
    print(f"\n测试不完整参数 - 状态码: {response.status_code}")
    try:
        print(f"不完整参数响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except json.JSONDecodeError:
        print(f"无法解析不完整参数响应为JSON: {response.text}")
        
    # 测试手机号格式不正确的情况
    invalid_phone_data = test_data.copy()
    invalid_phone_data["phone"] = "12345678"
    
    response = requests.post(
        "http://127.0.0.1:8000/api/users/register/",
        headers=headers,
        data=json.dumps(invalid_phone_data)
    )
    
    print(f"\n测试无效手机号 - 状态码: {response.status_code}")
    try:
        print(f"无效手机号响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except json.JSONDecodeError:
        print(f"无法解析无效手机号响应为JSON: {response.text}")
        
except Exception as e:
    print(f"测试其他情况时发生异常: {str(e)}")