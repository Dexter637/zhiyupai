# MongoDB数据库结构更新指南

本文档提供了如何使用`update_mongo_schema.py`脚本来更新Docker中的MongoDB数据库结构，以支持以下功能：

1. 为用户健康数据增加血糖、血压、血脂、食谱、身高体重等字段
2. 增加手机号作为登录凭证并确保唯一性
3. 修改数据库索引以优化查询性能

## 脚本功能概述

`update_mongo_schema.py`脚本主要完成以下工作：

- 为用户档案集合(`user_profiles`)添加`phone`字段的唯一索引
- 更新健康数据集合(`wearable_data`)的索引结构，以支持新的数据类型查询
- 可选：为现有用户创建演示健康数据记录
- 提供登录API和健康数据API的代码修改建议

## 前提条件

在运行此脚本之前，请确保：

1. Docker中的MongoDB容器已启动并正常运行
2. Python环境已安装pymongo库 (`pip install pymongo`)
3. 已备份数据库（重要数据请先备份，以防万一）

## 使用方法

1. 打开命令行终端
2. 导航到项目根目录 (`d:\zhiyupai`)
3. 运行以下命令：

```bash
python update_mongo_schema.py
```

4. 按照脚本提示完成操作

## 脚本执行流程

执行脚本后，将按照以下流程进行操作：

1. 连接到MongoDB数据库
2. 检查并为用户档案集合添加`phone`唯一索引
3. 更新健康数据集合的索引结构
4. 询问是否创建演示健康数据
5. 显示登录API的修改建议（需要手动更新）
6. 显示健康数据API的修改建议（需要手动更新）

## 手动更新API代码

脚本执行完成后，还需要手动更新以下文件：

### 1. 更新登录API以支持手机号登录

打开`d:\zhiyupai\user_app\views.py`文件，找到`login_user`函数，使用脚本提供的建议代码替换现有代码。新代码将支持用户使用用户名或手机号进行登录。

### 2. 添加获取当前用户健康数据的API

在`d:\zhiyupai\health_app\views.py`文件中，添加脚本提供的`my_health_data`函数，该函数将根据当前登录的用户返回其健康数据，而不是针对特定用户ID。

### 3. 更新URL配置

在`d:\zhiyupai\health_app\urls.py`文件中，添加新API的URL路由配置：

```python
urlpatterns = [
    # 现有的URL配置...
    path('my-health-data/', views.my_health_data, name='my_health_data'),
    path('my-health-data/<str:data_type>/', views.my_health_data, name='my_health_data_by_type'),
]
```

## 前端修改建议

为了支持新的功能，前端也需要进行相应的修改：

1. 在登录表单中添加使用手机号登录的选项
2. 添加健康数据输入表单，允许用户输入血糖、血压、血脂、食谱、身高体重等数据
3. 修改健康数据展示页面，使用新的`my-health-data`API获取当前登录用户的数据

## 数据结构说明

### 支持的健康数据类型

更新后的数据库支持以下健康数据类型：

| 数据类型 | 字段 | 说明 |
|---------|------|------|
| blood_glucose | value, unit, measurement_time, note | 血糖数据 |
| blood_pressure | systolic, diastolic, unit, measurement_time, note | 血压数据 |
| blood_lipids | total_cholesterol, triglycerides, hdl, ldl, unit, measurement_time, note | 血脂数据 |
| diet | meals, date, total_calories | 食谱数据 |
| height_weight | height, weight, bmi, unit, measurement_time, note | 身高体重数据 |

### 示例数据结构

```json
// 血糖数据示例
{
  "user_id": "1",
  "timestamp": "2023-05-10T10:30:00",
  "data_type": "blood_glucose",
  "value": 5.2,
  "unit": "mmol/L",
  "measurement_time": "2023-05-10T10:30:00",
  "note": "空腹血糖"
}

// 血压数据示例
{
  "user_id": "1",
  "timestamp": "2023-05-10T10:35:00",
  "data_type": "blood_pressure",
  "systolic": 120,
  "diastolic": 80,
  "unit": "mmHg",
  "measurement_time": "2023-05-10T10:35:00",
  "note": "正常血压"
}
```

## 注意事项

1. 此脚本针对Docker环境中的MongoDB数据库，确保连接配置正确
2. 在生产环境中使用前，请先在测试环境验证
3. 修改API代码后，建议进行全面测试以确保功能正常
4. 如果遇到索引创建失败的情况，可能是因为现有数据中存在重复的手机号值

## 常见问题解决

### 索引创建失败

如果创建`phone`唯一索引失败，可能是因为数据库中已存在重复的手机号。可以使用以下命令查找重复的手机号：

```python
# 在Python shell中执行
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/health_db")
db = client.health_db
pipeline = [
    {"$group": {"_id": "$phone", "count": {"$sum": 1}}},
    {"$match": {"count": {"$gt": 1}}}
]
duplicate_phones = list(db.user_profiles.aggregate(pipeline))
print("重复的手机号:", duplicate_phones)
```

找到重复的手机号后，手动修改或删除重复记录，然后重新运行脚本。

### 连接数据库失败

如果连接MongoDB数据库失败，请检查：

1. Docker容器是否正在运行
2. MongoDB服务的端口映射是否正确（默认27017）
3. 连接字符串是否正确

## 支持

如果在使用过程中遇到任何问题，请联系系统管理员或开发团队获取支持。