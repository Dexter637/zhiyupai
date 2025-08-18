import requests
import time

# 测试API端点
def test_api_endpoint():
    url = "http://127.0.0.1:8001/api/users/register/"
    print(f"测试API端点: {url}")
    
    try:
        # 发送一个简单的GET请求来检查端点是否可达
        start_time = time.time()
        response = requests.get(url, timeout=5)
        end_time = time.time()
        
        print(f"请求状态码: {response.status_code}")
        print(f"响应内容长度: {len(response.text)} 字节")
        print(f"请求耗时: {end_time - start_time:.2f} 秒")
        
        # 对于POST端点，GET请求可能会返回405 Method Not Allowed，这是正常的
        if response.status_code == 405:
            print("API端点存在，但不允许GET请求（这是正常的，因为这是一个POST端点）")
        elif response.status_code == 200:
            print("API端点存在且响应正常")
        else:
            print(f"API端点返回了非预期的状态码: {response.status_code}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("无法连接到API端点，可能是服务器未启动或端口被占用")
    except requests.exceptions.Timeout:
        print("API请求超时")
    except Exception as e:
        print(f"测试API端点时发生错误: {str(e)}")
    
    return False

# 运行测试
if __name__ == "__main__":
    print("开始测试API连接...")
    success = test_api_endpoint()
    print(f"API连接测试{'成功' if success else '失败'}")
    
    # 如果测试失败，提供一些常见的解决方案
    if not success:
        print("\n可能的解决方案:")
        print("1. 确保Django开发服务器正在运行")
        print("2. 检查服务器端口是否正确（当前测试的是8000端口）")
        print("3. 尝试使用不同的端口启动服务器（例如：python manage.py runserver 8001）")