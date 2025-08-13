# 智愈派健康系统

## 项目概述
智愈派是一个综合性健康管理系统，集成了动态推荐算法、健康预警系统和完整的后端API架构。系统基于Django和MongoDB构建，提供了用户认证、健康数据管理、异常检测和个性化推荐等功能。该系统特别关注用户的心脏健康，能够分析MIT-BIH心电图数据并提供健康风险评估。

## 核心功能
1. **动态推荐算法**：基于用户健康数据，提供个性化的健康建议和运动推荐
2. **健康预警系统**：监测用户健康数据，及时发现异常并提供预警，特别是心脏健康问题
3. **系统架构与API**：提供稳定、安全的数据接口，支持可穿戴设备数据接入
4. **用户认证与权限管理**：基于JWT的用户认证和细粒度权限控制
5. **跨域请求处理**：支持前端跨域访问API
6. **心电图分析**：集成MIT-BIH数据集，提供心脏健康分析功能

## 技术栈
- **后端框架**：Django, Django REST Framework
- **数据库**：MongoDB (主数据存储), SQLite (Django内部使用)
- **认证系统**：JWT (JSON Web Tokens)
- **数据处理**：NumPy, Pandas, wfdb (用于心电图数据处理)
- **机器学习**：PyOD (异常检测), LightGBM (分类模型)
- **Web服务**：Gunicorn (WSGI服务器), Whitenoise (静态文件服务)

## 项目结构
```
zhiyupai/
├── .gitignore           # Git忽略文件
├── README.md            # 项目说明
├── api/                 # API接口模块
│   └── wearable_device_api.py  # 可穿戴设备API
├── check_autoencoder_params.py  # 自动编码器参数检查
├── check_database.py    # 数据库检查
├── config.py            # MongoDB配置
├── create_indexes.py    # 创建MongoDB索引
├── data/                # 数据目录
│   └── mit-bih/         # MIT-BIH心电图数据集
├── demo_model_loading.py  # 模型加载演示
├── docs/                # 文档目录
├── fix_duplicate_key_error.py  # 修复重复键错误
├── health_app/          # 健康数据管理应用
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── health_monitoring/   # 健康监测与预警模块
├── import wfdb.py       # 导入wfdb库示例
├── manage.py            # Django命令行工具
├── models.py            # MongoDB数据模型
├── models/              # 模型存储目录
│   └── best_threshold.txt  # 最佳阈值配置
├── predict_mit_data.py  # MIT数据预测
├── recommendation/      # 推荐算法模块
├── requirements.txt     # 项目依赖
├── templates/           # 模板目录
│   └── index.html
├── test1.py             # 测试文件
├── test_mongodb.py      # MongoDB测试
├── train_pyod_model.py  # 训练PyOD模型
├── train_pyod_model_backup.py  # 训练PyOD模型备份
├── user_app/            # 用户管理应用
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   │   └── __init__.py
│   ├── models.py
│   ├── permissions.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
└── zhiyupai_project/    # Django项目配置
    ├── __init__.py
    ├── asgi.py
    ├── middleware.py
    ├── settings.py
    ├── urls.py
    └── wsgi.py
```

## 安装与运行

### 环境要求
- Python 3.8+
- MongoDB 4.4+
- Git

### 安装步骤
1. 克隆项目代码
   ```
   git clone https://github.com/Dexter637/zhiyupai.git
   cd zhiyupai
   ```
2. 创建并激活虚拟环境（可选但推荐）
   ```
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. 安装依赖包：
   ```
   pip install -r requirements.txt
   ```
4. 配置MongoDB连接（在config.py中）
   ```python
   # 示例配置
   MONGODB_URI = 'mongodb://localhost:27017/zhiyupai'
   ```
5. 创建MongoDB索引：
   ```
   python create_indexes.py
   ```
6. 运行Django开发服务器：
   ```
   python manage.py runserver
   ```

### 测试数据加载
系统包含MIT-BIH心电图数据集，可通过以下方式加载和测试：
```
python import_wfdb.py
python predict_mit_data.py
```

## API文档

### 认证API
- `POST /api/auth/token/` - 获取JWT访问令牌
- `POST /api/auth/token/refresh/` - 刷新JWT令牌
- `POST /api/auth/token/verify/` - 验证JWT令牌
- `POST /api/user/register/` - 用户注册

### 用户API
- `GET /api/user/profile/` - 获取当前用户档案
- `PUT /api/user/profile/` - 更新用户档案
- `GET /api/user/health-overview/` - 获取用户健康数据概览

### 健康数据API
- `GET /api/health/heart-rate/` - 获取心率数据列表
- `GET /api/health/heart-rate/<user_id>/` - 获取特定用户的心率数据
- `GET /api/health/sleep/` - 获取睡眠数据列表
- `GET /api/health/sleep/<user_id>/` - 获取特定用户的睡眠数据

### 可穿戴设备API
- `POST /api/wearable/upload/` - 上传可穿戴设备数据
- `GET /api/wearable/data/<user_id>/` - 获取用户的可穿戴设备数据

### 活动推荐API
- `GET /api/health/recommendations/<user_id>/` - 获取用户的活动推荐
- `POST /api/health/recommendations/<recommendation_id>/complete/` - 标记活动推荐为已完成

## 开发路线图
1. **近期目标（1-2个月）**
   - 完成心电图异常检测算法优化
   - 开发用户健康报告生成功能
   - 实现基于用户历史数据的个性化推荐

2. **中期目标（3-6个月）**
   - 开发React前端界面
   - 开发"睡眠质量-运动建议"关联模型
   - 集成更多可穿戴设备数据接口

3. **长期目标（6个月以上）**
   - 实现AI辅助诊断功能
   - 开发移动端应用
   - 建立健康数据共享平台

## 贡献指南
1. Fork本项目
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 提交Pull Request

## 问题反馈
如遇任何问题，请在GitHub Issues中提交反馈：
https://github.com/Dexter637/zhiyupai/issues

## 许可证
本项目采用MIT许可证 - 详情见LICENSE文件
   - 实现数据可视化展示
3. 功能扩展
   - 添加社交功能，支持用户互动
   - 集成更多类型的健康数据
4. 部署优化
   - 容器化部署
   - 性能优化与监控