# 睡眠质量-运动建议关联模型
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MongoDB模型
from models import ActivityRecommendation

class SleepActivityRecommender:
    """
    基于LightGBM的睡眠质量-运动建议关联模型
    通过分析用户的睡眠数据，推荐合适的运动类型和强度
    """
    
    def __init__(self):
        self.model = None
        self.activity_types = [
            '轻度散步', '中等强度步行', '慢跑', '游泳', 
            '瑜伽', '拉伸运动', '力量训练', '有氧运动',
            '太极', '冥想放松'
        ]
    
    def preprocess_data(self, sleep_data):
        """
        预处理睡眠数据
        
        参数:
            sleep_data (DataFrame): 包含睡眠指标的数据框
                应包含字段: 深睡时长、浅睡时长、睡眠效率、睡眠中断次数等
        
        返回:
            处理后的特征数据
        """
        features = pd.DataFrame()
        
        # 提取关键睡眠指标
        if 'deep_sleep_mins' in sleep_data.columns:
            features['deep_sleep_ratio'] = sleep_data['deep_sleep_mins'] / sleep_data['total_sleep_mins']
        
        if 'rem_sleep_mins' in sleep_data.columns:
            features['rem_sleep_ratio'] = sleep_data['rem_sleep_mins'] / sleep_data['total_sleep_mins']
        
        # 睡眠质量指标
        if 'sleep_efficiency' in sleep_data.columns:
            features['sleep_efficiency'] = sleep_data['sleep_efficiency']
        
        if 'interruptions' in sleep_data.columns:
            features['interruption_density'] = sleep_data['interruptions'] / (sleep_data['total_sleep_mins'] / 60)
        
        # 添加用户基本信息（如果有）
        for col in ['age', 'weight', 'height', 'gender', 'fitness_level']:
            if col in sleep_data.columns:
                features[col] = sleep_data[col]
        
        return features
    
    def train(self, sleep_data, activity_outcomes):
        """
        训练模型
        
        参数:
            sleep_data (DataFrame): 睡眠数据
            activity_outcomes (DataFrame): 运动结果数据，包含运动类型和效果评分
        """
        # 预处理数据
        X = self.preprocess_data(sleep_data)
        y = activity_outcomes['effectiveness_score']
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 设置LightGBM参数
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 训练模型
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data]
        )
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f'模型RMSE: {rmse:.4f}')
    
    def recommend_activity(self, sleep_data, user_id=None, user_profile=None, save_to_db=True):
        """
        基于睡眠数据推荐运动活动
        
        参数:
            sleep_data (dict): 用户的睡眠数据
            user_id (str, optional): 用户ID，用于保存到MongoDB
            user_profile (dict, optional): 用户的基本信息
            save_to_db (bool): 是否将推荐结果保存到MongoDB
        
        返回:
            dict: 推荐的运动类型、强度和持续时间
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 将输入数据转换为DataFrame
        input_data = pd.DataFrame([sleep_data])
        if user_profile:
            for key, value in user_profile.items():
                input_data[key] = value
        
        # 预处理数据
        features = self.preprocess_data(input_data)
        
        # 预测每种活动类型的效果
        activity_scores = {}
        for activity in self.activity_types:
            # 这里简化处理，实际应用中可能需要更复杂的逻辑
            # 例如为每种活动类型训练单独的模型
            activity_features = features.copy()
            activity_features['activity_type'] = self.activity_types.index(activity)
            
            # 预测效果分数
            score = self.model.predict(activity_features)[0]
            activity_scores[activity] = float(score)  # 确保可序列化为JSON
        
        # 选择得分最高的活动
        best_activity = max(activity_scores.items(), key=lambda x: x[1])
        
        # 确定推荐的运动强度和时长
        intensity, duration = self._determine_intensity_duration(sleep_data, best_activity[0])
        
        # 创建推荐结果
        recommendation = {
            'activity_type': best_activity[0],
            'expected_effectiveness': float(best_activity[1]),  # 确保可序列化为JSON
            'recommended_intensity': intensity,
            'recommended_duration': duration,
            'all_scores': activity_scores,
            'sleep_data': sleep_data,
            'timestamp': datetime.now()
        }
        
        # 保存到MongoDB
        if save_to_db and user_id:
            # 创建活动推荐记录
            recommendation_id = ActivityRecommendation.create_recommendation(
                user_id=user_id,
                activity_type=best_activity[0],
                intensity=intensity,
                duration=duration,
                effectiveness_score=float(best_activity[1]),
                sleep_data=sleep_data,
                timestamp=datetime.now()
            )
            
            # 将MongoDB ID添加到返回结果
            recommendation['recommendation_id'] = str(recommendation_id)
        elif save_to_db and not user_id:
            raise ValueError("保存推荐结果到数据库时必须提供user_id")
        
        return recommendation
    
    def _determine_intensity_duration(self, sleep_data, activity_type):
        """
        根据睡眠质量和活动类型确定推荐的运动强度和持续时间
        
        参数:
            sleep_data (dict): 用户的睡眠数据
            activity_type (str): 活动类型
        
        返回:
            tuple: (强度, 持续时间)
        """
        # 提取睡眠质量指标
        sleep_efficiency = sleep_data.get('sleep_efficiency', 0.8)
        deep_sleep_ratio = sleep_data.get('deep_sleep_mins', 0) / sleep_data.get('total_sleep_mins', 480)
        
        # 基于睡眠质量调整强度
        if sleep_efficiency < 0.7 or deep_sleep_ratio < 0.15:
            # 睡眠质量差，推荐低强度
            intensity = '低'
            base_duration = 20  # 分钟
        elif sleep_efficiency > 0.85 and deep_sleep_ratio > 0.25:
            # 睡眠质量好，可以推荐高强度
            intensity = '高'
            base_duration = 45  # 分钟
        else:
            # 睡眠质量一般，推荐中等强度
            intensity = '中'
            base_duration = 30  # 分钟
        
        # 根据活动类型调整持续时间
        if activity_type in ['轻度散步', '瑜伽', '拉伸运动', '太极', '冥想放松']:
            # 低强度活动可以持续更长时间
            duration = base_duration * 1.2
        elif activity_type in ['中等强度步行', '游泳']:
            duration = base_duration
        else:  # 高强度活动
            duration = base_duration * 0.8
        
        return intensity, int(duration)


# 示例用法
def generate_sample_data(n_samples=100):
    """
    生成示例数据用于测试
    """
    np.random.seed(42)
    
    # 生成睡眠数据
    sleep_data = pd.DataFrame({
        'total_sleep_mins': np.random.normal(480, 60, n_samples),  # 平均8小时
        'deep_sleep_mins': np.random.normal(120, 30, n_samples),   # 平均2小时深睡
        'rem_sleep_mins': np.random.normal(90, 20, n_samples),     # 平均1.5小时REM
        'sleep_efficiency': np.random.normal(0.85, 0.1, n_samples).clip(0.5, 1.0),
        'interruptions': np.random.poisson(2, n_samples),
        'age': np.random.randint(18, 65, n_samples),
        'fitness_level': np.random.randint(1, 6, n_samples)  # 1-5的健身水平
    })
    
    # 生成运动效果数据（模拟真实关系）
    effectiveness = (
        0.3 * sleep_data['deep_sleep_mins'] / sleep_data['total_sleep_mins'] +
        0.3 * sleep_data['sleep_efficiency'] -
        0.2 * sleep_data['interruptions'] / (sleep_data['total_sleep_mins'] / 60) +
        0.1 * sleep_data['fitness_level'] +
        0.1 * np.random.normal(0, 1, n_samples)  # 随机噪声
    )
    
    # 归一化到1-10分
    min_score, max_score = effectiveness.min(), effectiveness.max()
    normalized_scores = 1 + 9 * (effectiveness - min_score) / (max_score - min_score)
    
    activity_outcomes = pd.DataFrame({
        'effectiveness_score': normalized_scores
    })
    
    return sleep_data, activity_outcomes


if __name__ == "__main__":
    try:
        # 生成示例数据
        print("生成测试睡眠数据...")
        sleep_data, activity_outcomes = generate_sample_data(500)
        
        # 创建并训练模型
        print("初始化活动推荐器并训练模型...")
        recommender = SleepActivityRecommender()
        recommender.train(sleep_data, activity_outcomes)
        
        # 测试推荐
        test_sleep_data = {
            'total_sleep_mins': 450,
            'deep_sleep_mins': 100,
            'rem_sleep_mins': 80,
            'sleep_efficiency': 0.82,
            'interruptions': 3
        }
        
        test_user_profile = {
            'age': 35,
            'fitness_level': 3
        }
        
        # 测试MongoDB集成
        print("\n生成活动推荐并保存到MongoDB...")
        recommendation = recommender.recommend_activity(
            sleep_data=test_sleep_data, 
            user_id="test_user_123", 
            user_profile=test_user_profile,
            save_to_db=True
        )
        
        print("\n推荐结果:")
        print(f"推荐ID: {recommendation.get('recommendation_id', 'N/A')}")
        print(f"推荐活动: {recommendation['activity_type']}")
        print(f"预期效果评分: {recommendation['expected_effectiveness']:.2f}/10")
        print(f"推荐强度: {recommendation['recommended_intensity']}")
        print(f"推荐时长: {recommendation['recommended_duration']}分钟")
        
        # 测试不保存到数据库的情况
        print("\n生成活动推荐但不保存到MongoDB...")
        recommendation_no_save = recommender.recommend_activity(
            sleep_data=test_sleep_data, 
            user_profile=test_user_profile,
            save_to_db=False
        )
        
        print("\n推荐结果 (不保存):")
        print(f"推荐活动: {recommendation_no_save['activity_type']}")
        print(f"预期效果评分: {recommendation_no_save['expected_effectiveness']:.2f}/10")
        
        print("\nMongoDB集成测试完成!")
        
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()