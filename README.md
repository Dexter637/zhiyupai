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
- **后端框架**：Django 4.2+, Django REST Framework 3.14+
- **数据库**：MongoDB 5.0 (主数据存储), SQLite (Django内部使用)
- **认证系统**：JWT (JSON Web Tokens) with django-rest-framework-simplejwt
- **数据处理**：NumPy 1.24+, Pandas 2.0+, wfdb 4.1+ (用于心电图数据处理)
- **机器学习**：PyOD 1.1+ (异常检测), LightGBM 3.3+ (分类模型), scikit-learn 1.2+
- **Web服务**：Gunicorn 20.1+ (WSGI服务器), Whitenoise 6.4+ (静态文件服务)
- **数据可视化**：Matplotlib 3.7+, Seaborn 0.12+
- **开发工具**：Docker Desktop, MongoDB 5.0, Mongo-Express

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

#### 1. 克隆项目代码
```
git clone https://github.com/Dexter637/zhiyupai.git
cd zhiyupai
```

#### 2. Docker容器化MongoDB环境配置
本项目提供了完整的Docker化MongoDB数据库环境，包含MongoDB服务器和Mongo-Express管理界面，适合开发和测试环境使用。

##### 2.1 环境要求
- Windows 10/11 Pro/Enterprise版本（支持WSL2）
- 已安装Docker Desktop（建议最新版本）
- 已启用Hyper-V和WSL2功能
- 至少4GB内存
- 至少10GB可用磁盘空间

##### 2.2 Docker Compose配置说明
项目使用 `docker-compose.yml` 文件配置以下服务：
- **mongodb**: MongoDB 5.0数据库服务
  - 端口映射：27017:27017
  - 数据持久化：使用Docker卷 `mongo-data` 和 `mongo-config`
  - 自动初始化：通过 `mongo-init.js` 脚本创建数据库和用户
  - 管理员账户：admin/admin123

- **mongo-express**: MongoDB Web管理界面
  - 端口映射：8081:8081
  - 可选的图形化管理工具

##### 2.2.1 Docker镜像加速配置（可选）
为加速Docker镜像下载，项目提供了配置脚本。以**管理员身份**运行PowerShell：

1. 进入项目目录：
   ```powershell
   cd d:\zhiyupai
   ```
2. 运行镜像源配置脚本：
   ```powershell
   .\setup_docker_mirror.ps1
   ```
3. 重启Docker服务：
   ```powershell
   Restart-Service docker
   ```

配置的镜像源包括：
- https://mirrors.tuna.tsinghua.edu.cn/ (清华大学镜像)
- https://docker.m.daocloud.io (道客云镜像)
- https://dockerproxy.com (Docker代理)
- https://mirror.baidubce.com (百度云镜像)
- https://docker.nju.edu.cn (南京大学镜像)
- https://mirror.iscas.ac.cn (中科院镜像)
- https://mirrors.cn99.com (备用镜像)
- https://docker.mirrors.ustc.edu.cn (中科大镜像)
- https://registry.docker-cn.com (Docker中国官方镜像)
- https://hub-mirror.c.163.com (网易镜像)

##### 2.3 启动MongoDB容器

1. 打开PowerShell或命令提示符
2. 进入项目根目录：
   ```powershell
   cd d:\zhiyupai
   ```
3. 启动Docker容器（后台模式）：
   ```powershell
   docker-compose up -d
   ```

首次启动会自动下载MongoDB 5.0和Mongo-Express镜像，并执行数据库初始化脚本。

##### 2.4 验证容器运行状态

```powershell
# 查看容器运行状态
docker-compose ps

# 查看实时日志
docker-compose logs -f mongodb

# 检查MongoDB服务健康状态
docker exec mongodb-container mongosh --eval "db.adminCommand('ping')"
```

##### 2.5 访问数据库

**命令行连接（推荐）**:
```powershell
# 使用管理员账户连接
docker exec -it mongodb-container mongosh -u admin -p admin123 --authenticationDatabase admin

# 使用应用账户连接
docker exec -it mongodb-container mongosh -u health_user -p health_password --authenticationDatabase health_db
```

**Web界面访问（可选）**:
打开浏览器访问：http://localhost:8081
- 用户名：admin
- 密码：admin123
- 连接字符串：mongodb://admin:admin123@mongodb:27017/

##### 2.6 数据库连接配置
在项目的 `config.py` 文件中配置连接信息：

```python
# MongoDB连接信息 - Docker环境配置
MONGO_URI = "mongodb://localhost:27017/health_db"
DATABASE_NAME = "health_db"

# 注意：当前Docker环境使用无认证模式，如需启用认证请修改配置
# MONGO_URI = "mongodb://health_user:health_password@localhost:27017/health_db?authSource=health_db"
```

**重要提示**：当前Docker配置默认使用无认证模式。如需启用认证，请：
1. 修改 `docker-compose.yml` 中的MongoDB配置
2. 更新 `config.py` 中的连接字符串
3. 重启Docker容器

#### 3. 创建并激活Python虚拟环境（可选但推荐）
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
   # Docker环境下的连接配置（当前使用无认证模式）
   MONGO_URI = "mongodb://localhost:27017/health_db"
   DATABASE_NAME = "health_db"
   ```
   
   **注意**：Docker环境下的MongoDB默认使用以下配置：
   - 数据库名称：health_db
   - 连接模式：无认证（开发环境）
   - 管理员用户：admin/admin123（用于Mongo-Express）
   - 应用用户：health_user/health_password（已创建但未启用）
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
- `GET /api/health/sleep-data/?user_id=<user_id>` - 获取用户的详细睡眠分析数据（包含总睡眠时间、深睡时间、浅睡时间等指标）

### 可穿戴设备API
- `POST /api/wearable/upload/` - 上传可穿戴设备数据
- `GET /api/wearable/data/<user_id>/` - 获取用户的可穿戴设备数据

### 活动推荐API
- `GET /api/health/recommendations/<user_id>/` - 获取用户的活动推荐
- `POST /api/health/recommendations/<recommendation_id>/complete/` - 标记活动推荐为已完成

## 重要更新说明

### 睡眠数据API路由修复
近期修复了睡眠数据API的路由配置问题：

**主要变更**:
- 前端请求URL从 `/api/sleep-data/` 更新为 `/api/health/sleep-data/`
- 需要传递 `user_id` 查询参数来获取特定用户的睡眠数据
- API返回包含详细睡眠指标的数据，包括总睡眠时间、深睡时间、浅睡时间等

**前端修改**:
模板文件 `templates/sleep_analysis.html` 中的fetch请求URL已更新：
```javascript
// 修改前
fetch(`/api/sleep-data/?user_id=${userId}`)

// 修改后  
fetch(`/api/health/sleep-data/?user_id=${userId}`)
```

**API响应格式**:
睡眠数据API现在返回结构化的睡眠分析数据，包含详细的睡眠阶段和质量评分。

### Docker迁移完成
项目已完成MongoDB数据库的Docker化迁移：
- 使用 `docker-compose.yml` 配置MongoDB容器服务
- 数据持久化存储在Docker卷中，确保数据安全
- 支持多人协作开发环境，团队成员可远程访问数据库

## 测试与验证

### 可用测试脚本
项目提供了多个测试脚本用于验证系统功能：

- `test_mongodb.py` - 测试MongoDB连接和基本操作
- `test_mongo_connection.py` - 测试不同用户的MongoDB连接配置
- `verify_config.py` - 验证Docker环境下的MongoDB配置
- `check_database.py` - 完整的数据库准备情况检测
- `test_api_connection.py` - 测试API端点连接
- `test_auth_api.py` - 测试用户认证API

### 测试账号
系统包含预生成的测试账号（存储在 `generated_test_accounts.txt`）：

```
用户名: testuser001, 密码: I0XEbpMG, 手机号: 13825261680
用户名: testuser002, 密码: CeWi5u5t, 手机号: 13804533909
用户名: testuser003, 密码: l7G1fZ84, 手机号: 13854481373
用户名: testuser004, 密码: ZBPKmg2j, 手机号: 13837229371
用户名: testuser005, 密码: aQn4GY4u, 手机号: 13806452407
```

### 运行测试
```powershell
# 测试MongoDB连接
python test_mongodb.py

# 验证Docker配置
python verify_config.py

# 完整数据库检测
python check_database.py
```

## 常用Docker命令

在Docker环境下，您可以使用以下命令管理MongoDB容器：

```powershell
# 停止容器
docker-compose down

# 重启容器
docker-compose restart

# 查看Docker卷（数据持久化）
docker volume ls

# 查看容器详细信息
docker inspect mongodb-container

# 使用MongoDB客户端连接容器
docker exec -it mongodb-container mongo -u health_user -p health_password --authenticationDatabase health_db

# 删除容器和所有数据（谨慎使用）
docker-compose down -v
```

## 多人协作访问配置

如需团队成员远程访问MongoDB数据库：
1. 确保主机防火墙已开放27017端口
2. 团队成员使用主机IP地址和上述凭证连接数据库
3. 建议在生产环境中修改默认密码，提高安全性

## 数据库结构说明

Docker环境下的MongoDB初始化脚本自动创建了以下集合：
- `users` - 用户信息（已创建email唯一索引）
- `health_data` - 健康数据（已创建user_id和timestamp索引）
- `device_data` - 设备数据
- `sleep_analysis` - 睡眠分析数据（包含睡眠阶段、时长、质量评分等详细指标）

### 睡眠分析数据结构
睡眠数据API返回的JSON格式示例：
```json
{
  "user_id": 25,
  "sleep_records": [
    {
      "date": "2024-01-15",
      "total_sleep_mins": 480,
      "deep_sleep_mins": 120,
      "light_sleep_mins": 240,
      "rem_sleep_mins": 120,
      "sleep_quality_score": 85
    }
  ]
}
```

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