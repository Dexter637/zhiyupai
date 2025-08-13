# 数据库检测脚本
import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入测试模块
from test_mongodb import run_all_tests

if __name__ == "__main__":
    print("="*50)
    print("智愈派 - 数据库准备情况检测")
    print("="*50)
    
    # 运行MongoDB测试
    print("\n[1/1] 检测MongoDB数据库连接和配置...")
    db_ready = run_all_tests()
    
    # 总结
    print("\n"+"="*50)
    if db_ready:
        print("✅ 数据库检测完成: MongoDB数据库已准备就绪!")
        print("您可以开始使用智愈派系统了。")
    else:
        print("⚠️ 数据库检测完成: 发现一些问题需要解决")
        print("请检查上述错误信息，确保MongoDB服务正在运行，并且配置正确。")
    
    print("="*50)
    
    # 等待用户按键退出
    input("\n按回车键退出...")