// MongoDB初始化脚本
// 创建数据库和用户

// 连接到admin数据库
conn = new Mongo();
db = conn.getDB('admin');

// 验证管理员身份
db.auth('admin', 'admin123');

// 创建health_db数据库
db = conn.getDB('health_db');

// 创建具有读写权限的用户
db.createUser({
  user: 'health_user',
  pwd: 'health_password',
  roles: [
    {
      role: 'readWrite',
      db: 'health_db'
    }
  ]
});

// 创建集合（可选，根据项目需求）
db.createCollection('users');
db.createCollection('health_data');
db.createCollection('device_data');
db.createCollection('sleep_analysis');

db.getCollection('users').createIndex({ 'email': 1 }, { unique: true });
db.getCollection('health_data').createIndex({ 'user_id': 1, 'timestamp': -1 });

// 输出初始化完成信息
print('MongoDB初始化完成！已创建：');
print('1. 数据库: health_db');
print('2. 用户: health_user (具有读写权限)');
print('3. 集合: users, health_data, device_data, sleep_analysis');
print('4. 索引: users.email (唯一), health_data.user_id+timestamp');