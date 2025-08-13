# 心率异常检测模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from sklearn.preprocessing import StandardScaler
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MongoDB模型
from models import HeartRateAlert

class HeartRateMonitor:
    """
    基于PyOD库的心率异常检测系统
    可以检测用户心率数据中的异常值，并提供健康预警
    """
    
    def __init__(self, model_type='iforest', contamination=0.05):
        """
        初始化心率监测器
        
        参数:
            model_type (str): 异常检测算法类型，可选'knn', 'iforest', 'lof'
            contamination (float): 预期的异常值比例，范围(0, 0.5)
        """
        self.model_type = model_type
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.user_baseline = {}
        self.is_fitted = False
    
    def _create_model(self):
        """
        根据指定的模型类型创建异常检测模型
        """
        if self.model_type == 'knn':
            return KNN(contamination=self.contamination)
        elif self.model_type == 'iforest':
            return IForest(contamination=self.contamination, random_state=42)
        elif self.model_type == 'lof':
            return LOF(contamination=self.contamination)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}，请选择'knn', 'iforest'或'lof'")
    
    def fit(self, heart_rate_data, user_info=None):
        """
        使用历史心率数据训练异常检测模型
        
        参数:
            heart_rate_data (DataFrame): 包含心率数据的DataFrame
                应包含字段: timestamp, heart_rate, activity_level(可选)
            user_info (dict, optional): 用户基本信息，如年龄、性别等
        """
        # 提取特征
        features = self._extract_features(heart_rate_data)
        
        # 标准化特征
        X = self.scaler.fit_transform(features)
        
        # 创建并训练模型
        self.model = self._create_model()
        self.model.fit(X)
        
        # 计算用户基线数据
        self._calculate_baseline(heart_rate_data, user_info)
        
        self.is_fitted = True
        
        return self
    
    def _extract_features(self, heart_rate_data):
        """
        从心率数据中提取特征
        
        参数:
            heart_rate_data (DataFrame): 心率数据
            
        返回:
            DataFrame: 提取的特征
        """
        # 确保数据按时间排序
        if 'timestamp' in heart_rate_data.columns:
            heart_rate_data = heart_rate_data.sort_values('timestamp')
        
        # 提取基本特征
        features = pd.DataFrame()
        features['heart_rate'] = heart_rate_data['heart_rate']
        
        # 计算滑动窗口统计量
        window_sizes = [5, 10, 20]  # 不同窗口大小
        for window in window_sizes:
            if len(heart_rate_data) >= window:
                features[f'hr_rolling_mean_{window}'] = heart_rate_data['heart_rate'].rolling(window=window, min_periods=1).mean()
                features[f'hr_rolling_std_{window}'] = heart_rate_data['heart_rate'].rolling(window=window, min_periods=1).std()
                features[f'hr_rolling_max_{window}'] = heart_rate_data['heart_rate'].rolling(window=window, min_periods=1).max()
                features[f'hr_rolling_min_{window}'] = heart_rate_data['heart_rate'].rolling(window=window, min_periods=1).min()
        
        # 添加活动水平（如果有）
        if 'activity_level' in heart_rate_data.columns:
            features['activity_level'] = heart_rate_data['activity_level']
        
        # 计算心率变化率
        features['hr_change'] = features['heart_rate'].diff().fillna(0)
        
        # 填充缺失值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        return features
    
    def _calculate_baseline(self, heart_rate_data, user_info=None):
        """
        计算用户的基线心率数据
        
        参数:
            heart_rate_data (DataFrame): 心率数据
            user_info (dict, optional): 用户信息
        """
        # 计算基本统计量
        self.user_baseline['mean_hr'] = heart_rate_data['heart_rate'].mean()
        self.user_baseline['std_hr'] = heart_rate_data['heart_rate'].std()
        self.user_baseline['min_hr'] = heart_rate_data['heart_rate'].min()
        self.user_baseline['max_hr'] = heart_rate_data['heart_rate'].max()
        
        # 如果有活动水平信息，计算不同活动水平下的心率
        if 'activity_level' in heart_rate_data.columns:
            activity_stats = heart_rate_data.groupby('activity_level')['heart_rate'].agg(['mean', 'std', 'min', 'max'])
            self.user_baseline['activity_hr'] = activity_stats.to_dict()
        
        # 考虑用户信息（如果有）
        if user_info:
            self.user_baseline['user_info'] = user_info
            
            # 根据年龄计算理论最大心率
            if 'age' in user_info:
                self.user_baseline['theoretical_max_hr'] = 220 - user_info['age']
    
    def detect_anomalies(self, new_data, user_id=None, generate_alert=False, save_to_db=True):
        """
        检测新数据中的异常
        
        参数:
            new_data (DataFrame): 新的心率数据
            user_id (str, optional): 用户ID，用于保存警报
            generate_alert (bool): 是否生成健康警报
            save_to_db (bool): 是否将警报保存到MongoDB
            
        返回:
            tuple 或 DataFrame: 如果generate_alert为True，返回(带有异常标记的数据, 健康警报信息)；否则仅返回带有异常标记的数据
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 提取特征
        features = self._extract_features(new_data)
        
        # 标准化特征
        X = self.scaler.transform(features)
        
        # 预测异常分数和标签
        scores = self.model.decision_function(X)
        labels = self.model.predict(X)  # 二进制标签: 0正常, 1异常
        
        # 将结果添加到原始数据
        result = new_data.copy()
        result['anomaly_score'] = scores
        result['is_anomaly'] = labels
        
        # 生成健康警报
        if generate_alert:
            if user_id is None and save_to_db:
                raise ValueError("保存警报到数据库时必须提供user_id")
            
            alert = self.generate_health_alert(
                anomaly_data=result,
                user_id=user_id,
                save_to_db=save_to_db
            )
            return result, alert
        
        return result
    
    def generate_health_alert(self, anomaly_data, user_id, threshold=0.8, save_to_db=True):
        """
        根据异常检测结果生成健康预警
        
        参数:
            anomaly_data (DataFrame): 带有异常标记的数据
            user_id (str): 用户ID
            threshold (float): 异常分数阈值，超过此值将生成高优先级警报
            save_to_db (bool): 是否将警报保存到MongoDB
            
        返回:
            dict: 健康预警信息
        """
        # 筛选出异常点
        anomalies = anomaly_data[anomaly_data['is_anomaly'] == 1]
        
        if len(anomalies) == 0:
            alert = {
                'status': 'normal',
                'message': '未检测到心率异常',
                'alert_level': 'info',
                'anomaly_count': 0
            }
            return alert
        
        # 计算异常程度
        max_score = anomalies['anomaly_score'].max()
        avg_score = anomalies['anomaly_score'].mean()
        anomaly_ratio = len(anomalies) / len(anomaly_data)
        
        # 确定警报级别
        if max_score > threshold and anomaly_ratio > 0.1:
            alert_level = 'high'
            status = 'danger'
            message = '检测到严重心率异常，请立即关注'
            severity = 'high'
        elif max_score > threshold:
            alert_level = 'medium'
            status = 'warning'
            message = '检测到心率异常，建议密切关注'
            severity = 'medium'
        else:
            alert_level = 'low'
            status = 'caution'
            message = '检测到轻微心率异常，建议注意观察'
            severity = 'low'
        
        # 生成详细信息
        details = {
            'anomaly_timestamps': anomalies['timestamp'].tolist() if 'timestamp' in anomalies.columns else [],
            'anomaly_heart_rates': anomalies['heart_rate'].tolist(),
            'max_anomaly_score': float(max_score),  # 确保可序列化为JSON
            'avg_anomaly_score': float(avg_score),
            'anomaly_ratio': float(anomaly_ratio),
            'baseline_comparison': {
                'mean_difference': float(anomalies['heart_rate'].mean() - self.user_baseline['mean_hr']),
                'max_difference': float(anomalies['heart_rate'].max() - self.user_baseline['max_hr'])
            }
        }
        
        alert = {
            'status': status,
            'message': message,
            'alert_level': alert_level,
            'anomaly_count': len(anomalies),
            'details': details
        }
        
        # 保存到MongoDB
        if save_to_db:
            # 获取最高的心率值作为警报的心率值
            max_heart_rate = float(anomalies['heart_rate'].max())
            
            # 创建警报记录
            alert_id = HeartRateAlert.create_alert(
                user_id=user_id,
                heart_rate=max_heart_rate,
                severity=severity,
                message=message,
                timestamp=datetime.now()
            )
            
            # 将MongoDB ID添加到返回结果
            alert['alert_id'] = str(alert_id)
        
        return alert
    
    def visualize_anomalies(self, data, figsize=(12, 6)):
        """
        可视化心率数据和检测到的异常
        
        参数:
            data (DataFrame): 带有异常标记的数据
            figsize (tuple): 图形大小
            
        返回:
            matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制心率数据
        x = data.index if 'timestamp' not in data.columns else data['timestamp']
        ax.plot(x, data['heart_rate'], 'b-', label='心率')
        
        # 标记异常点
        anomalies = data[data['is_anomaly'] == 1]
        ax.scatter(anomalies.index if 'timestamp' not in anomalies.columns else anomalies['timestamp'], 
                  anomalies['heart_rate'], color='red', s=50, label='异常')
        
        # 添加基线参考
        if hasattr(self, 'user_baseline') and 'mean_hr' in self.user_baseline:
            ax.axhline(y=self.user_baseline['mean_hr'], color='g', linestyle='--', label='平均心率')
            ax.axhline(y=self.user_baseline['mean_hr'] + 2*self.user_baseline['std_hr'], 
                      color='y', linestyle='--', label='平均值+2σ')
            ax.axhline(y=self.user_baseline['mean_hr'] - 2*self.user_baseline['std_hr'], 
                      color='y', linestyle='--', label='平均值-2σ')
        
        # 设置图表
        ax.set_title('心率异常检测')
        ax.set_ylabel('心率 (BPM)')
        ax.set_xlabel('时间' if 'timestamp' in data.columns else '样本索引')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig


# 示例用法
def generate_sample_heart_data(n_samples=1000, anomaly_ratio=0.05):
    """
    生成示例心率数据用于测试
    
    参数:
        n_samples (int): 样本数量
        anomaly_ratio (float): 异常样本比例
        
    返回:
        DataFrame: 生成的心率数据
    """
    np.random.seed(42)
    
    # 生成时间戳
    base_time = pd.Timestamp('2023-01-01')
    timestamps = [base_time + pd.Timedelta(minutes=i) for i in range(n_samples)]
    
    # 生成正常心率数据 (均值75，标准差8)
    normal_hr = np.random.normal(75, 8, n_samples)
    
    # 生成活动水平 (0-静息, 1-轻度活动, 2-中度活动, 3-剧烈活动)
    activity_level = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # 根据活动水平调整心率
    for i, level in enumerate(activity_level):
        if level == 1:  # 轻度活动
            normal_hr[i] += np.random.normal(20, 5)
        elif level == 2:  # 中度活动
            normal_hr[i] += np.random.normal(40, 8)
        elif level == 3:  # 剧烈活动
            normal_hr[i] += np.random.normal(60, 10)
    
    # 生成异常数据
    anomaly_indices = np.random.choice(
        range(n_samples), 
        size=int(n_samples * anomaly_ratio), 
        replace=False
    )
    
    heart_rate = normal_hr.copy()
    
    # 插入异常值
    for idx in anomaly_indices:
        # 异常类型1: 突然极高或极低的心率
        if np.random.random() < 0.5:
            heart_rate[idx] = np.random.choice([
                np.random.uniform(120, 180),  # 异常高
                np.random.uniform(30, 45)     # 异常低
            ])
        # 异常类型2: 与活动水平不符的心率
        else:
            if activity_level[idx] <= 1:  # 静息或轻度活动
                heart_rate[idx] = np.random.uniform(120, 160)  # 异常高
            else:  # 中度或剧烈活动
                heart_rate[idx] = np.random.uniform(40, 60)    # 异常低
    
    # 创建DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': heart_rate,
        'activity_level': activity_level
    })
    
    return data


if __name__ == "__main__":
    try:
        # 生成示例数据
        print("生成测试心率数据...")
        heart_data = generate_sample_heart_data(1000, anomaly_ratio=0.05)
        
        # 添加用户ID字段用于MongoDB集成测试
        heart_data['user_id'] = "test_user_123"
        
        # 分割训练集和测试集
        train_data = heart_data.iloc[:800]
        test_data = heart_data.iloc[800:]
        
        # 用户信息
        user_info = {
            'age': 35,
            'gender': 'male',
            'weight': 70,
            'height': 175,  # cm
            'fitness_level': 'moderate'
        }
        
        # 创建并训练模型
        print("初始化心率监测器并训练模型...")
        monitor = HeartRateMonitor(model_type='iforest', contamination=0.05)
        monitor.fit(train_data, user_info=user_info)
        
        # 检测异常并生成警报，测试MongoDB集成
        print("检测异常并生成健康警报，保存到MongoDB...")
        results, alert = monitor.detect_anomalies(
            test_data, 
            user_id="test_user_123",
            generate_alert=True,
            save_to_db=True
        )
        
        # 打印警报信息
        print("\n健康预警:")
        print(f"警报ID: {alert.get('alert_id', 'N/A')}")
        print(f"状态: {alert['status']}")
        print(f"消息: {alert['message']}")
        print(f"警报级别: {alert['alert_level']}")
        print(f"异常点数量: {alert['anomaly_count']}")
        
        # 可视化结果
        print("\n可视化异常检测结果...")
        fig = monitor.visualize_anomalies(results)
        plt.show()
        
        print("\nMongoDB集成测试完成!")
        
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()