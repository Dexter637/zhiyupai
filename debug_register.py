import requests
import json

# 详细调试注册API
def debug_register():
    url = 'http://127.0.0.1:8000/api/users/register/'
    headers = {'Content-Type': 'application/json'}
    
    # 准备测试数据
    test_data = {
        'username': 'testuser456',
        'phone': '13900139000',
        'password': 'password456'
    }
    
    print(f'发送注册请求: {test_data}')
    try:
        response = requests.post(url, headers=headers, data=json.dumps(test_data))
        
        print(f'响应状态码: {response.status_code}')
        print(f'响应头: {response.headers}')
        print(f'响应内容: {response.text}')
        
        # 尝试解析JSON响应
        try:
            data = response.json()
            print(f'解析后的JSON响应: {json.dumps(data, ensure_ascii=False, indent=2)}')
        except json.JSONDecodeError:
            print('无法解析响应为JSON格式')
            
        if response.status_code == 201:
            print('注册成功！')
        elif response.status_code == 400:
            print('注册失败，可能是验证错误')
        else:
            print(f'注册失败，状态码: {response.status_code}')
            
    except Exception as e:
        print(f'请求出错: {str(e)}')

if __name__ == '__main__':
    debug_register()