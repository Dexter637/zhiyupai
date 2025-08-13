# 智愈派健康系统

## 项目概述
智愈派是一个综合性健康管理系统，集成了动态推荐算法、健康预警系统和完整的后端API架构。系统基于Django和MongoDB构建，提供了用户认证、健康数据管理、异常检测和个性化推荐等功能。

## 核心功能
1. **动态推荐算法**：基于用户健康数据，提供个性化的健康建议和运动推荐
2. **健康预警系统**：监测用户健康数据，及时发现异常并提供预警
3. **系统架构与API**：提供稳定、安全的数据接口，支持可穿戴设备数据接入
4. **用户认证与权限管理**：基于JWT的用户认证和细粒度权限控制
5. **跨域请求处理**：支持前端跨域访问API

## 技术栈
- **后端框架**：Django, Django REST Framework
- **数据库**：MongoDB (主数据存储), SQLite (Django内部使用)
- **认证系统**：JWT (JSON Web Tokens)
- **推荐算法**：Python, NumPy, Pandas
- **异常检测**：自定义算法模型

## 项目结构
```
智愈派/
├── zhiyupai_project/    # Django项目配置
│   ├── settings.py      # 项目设置
│   ├── urls.py          # URL路由配置
│   ├── middleware.py    # 自定义中间件
│   ├── wsgi.py          # WSGI配置
│   └── asgi.py          # ASGI配置
├── health_app/          # 健康数据管理应用
│   ├── views.py         # API视图
│   └── urls.py          # URL配置
├── user_app/            # 用户管理应用
│   ├── views.py         # API视图
│   ├── urls.py          # URL配置
│   └── permissions.py   # 权限类
├── recommendation/      # 推荐算法模块
├── health_monitoring/   # 健康监测与预警模块
├── api/                 # API接口模块
├── models.py            # MongoDB数据模型
├── config.py            # MongoDB配置
├── manage.py            # Django命令行工具
├── requirements.txt     # 项目依赖
└── README.md            # 项目说明
```

## 安装与运行

### 环境要求
- Python 3.8+
- MongoDB 4.4+

### 安装步骤
1. 克隆项目代码
2. 安装依赖包：
   ```
   pip install -r requirements.txt
   ```
3. 配置MongoDB连接（在config.py中）
4. 创建MongoDB索引：
   ```
   python create_indexes.py
   ```
5. 运行Django开发服务器：
   ```
   python manage.py runserver
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
1. 推荐算法深化
   - 实现基于用户历史数据的个性化推荐
   - 开发"睡眠质量-运动建议"关联模型
2. 前端开发
   - 开发React/Vue前端界面
   - 实现数据可视化展示
3. 功能扩展
   - 添加社交功能，支持用户互动
   - 集成更多类型的健康数据
4. 部署优化
   - 容器化部署
   - 性能优化与监控