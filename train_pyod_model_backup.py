import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
import wfdb
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class HeartRateAnomalyDetector:
    def __init__(self, random_state=42, window_size=30, contamination=0.1):
        self.random_state = random_state
        self.window_size = window_size  # 滑动窗口大小
        self.contamination = contamination  # 异常样本比例
        self.scaler = RobustScaler()    # 使用IQR缩放的鲁棒归一化
        self.model = None
        self.feature_names = None

    def load_mit_bih_data(self, data_path):
        """加载MIT-BIH数据集"""
        records = []
        annotations = []
        record_ids = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
        
        for record_id in tqdm(record_ids, desc="加载数据"):
            record_path = os.path.join(data_path, f"{record_id}")
            
            # 读取信号数据
            signal, fields = wfdb.rdsamp(record_path)
            df_signal = pd.DataFrame(signal, columns=[f'channel_{i}' for i in range(signal.shape[1])])
            df_signal['record_id'] = record_id
            df_signal['timestamp'] = range(len(df_signal))
            records.append(df_signal)
            
            # 读取标注数据
            ann = wfdb.rdann(record_path, 'atr')
            df_ann = pd.DataFrame({
                'timestamp': ann.sample,
                'annotation': ann.symbol,
                'record_id': record_id
            })
            annotations.append(df_ann)
        
        # 合并所有记录和标注
        df_signals = pd.concat(records, ignore_index=True)
        df_annotations = pd.concat(annotations, ignore_index=True)
        
        return df_signals, df_annotations

    def load_annotation_mapping(self, annotation_path):
        """加载标注映射表"""
        return pd.read_csv(annotation_path)

    def create_labels(self, df_signals, df_annotations, annotation_mapping):
        """创建正常/异常标签用于评估"""
        # 定义异常标注符号集合
        abnormal_symbols = {'V', 'E', '!', '[', ']', 'r', 'e', 'n', 'f'}
        normal_symbols = {'N', 'L', 'R', 'j', 'A', 'a', 'S', 'J', 'F', 'Q', '/'}
        artifacts_symbols = {'~','|','x','(',')','u','?'}
        
        # 创建标签
        y = np.zeros(len(df_annotations))
        for idx, symbol in enumerate(df_annotations['annotation']):
            if symbol in abnormal_symbols:
                y[idx] = 1
            elif symbol in artifacts_symbols:
                y[idx] = -1  # 表示伪差
            
        # 将标注插值到每个时间点
        df_merged = pd.merge_asof(
                df_signals.sort_values('timestamp'),
                df_annotations[['record_id', 'timestamp', 'annotation']].sort_values('timestamp'),
                on='timestamp',
                by='record_id',
                direction='backward'
            )
        
        # 根据最近的标注创建标签
        y_labels = np.zeros(len(df_merged))
        for i, symbol in enumerate(df_merged['annotation']):
            if symbol in abnormal_symbols:
                y_labels[i] = 1
            elif symbol in artifacts_symbols:
                y_labels[i] = -1
        
        return y_labels

    def extract_features(self, signals):
        """从信号中提取特征"""
        features = []
        
        # 基础统计特征
        features.append(signals.mean().values)
        features.append(signals.std().values)
        features.append(signals.min().values)
        features.append(signals.max().values)
        features.append(signals.median().values)
        features.append(signals.skew().values)
        features.append(signals.kurtosis().values)
        
        # 心率变异性特征
        for channel in signals.columns:
            if channel not in ['record_id', 'timestamp']:
                # 计算R峰间距 (假设已经检测到R峰)
                # 这里简化处理，实际应用中需要R峰检测算法
                pass
        
        # 合并所有特征
        features = np.concatenate(features)
        return features

    def create_sliding_window_features(self, signals, y, window_size=1800):
        """创建滑动窗口特征和对应的标签"""
        X = []
        y_windowed = []
        
        # 修改窗口参数
        step_size = window_size // 2  # 50%重叠率
        max_windows = 10000  # 设置最大窗口上限
        
        # 按记录ID分组
        grouped_signals = signals.groupby('record_id')
        grouped_y = pd.Series(y, index=signals.index).groupby(signals['record_id'])
        total_windows_per_record = max_windows // grouped_signals.ngroups
        
        for record_id in tqdm(grouped_signals.groups.keys(), desc="处理患者记录"):
            group_signals = grouped_signals.get_group(record_id)
            group_y = grouped_y.get_group(record_id)
            
            record_length = len(group_signals)
            record_possible_windows = max(0, (record_length - window_size) // step_size + 1)
            sample_size = min(record_possible_windows, total_windows_per_record)
            
            if sample_size <= 0:
                continue
            
            # 均匀抽样窗口起始索引
            indices = np.linspace(window_size, record_length, sample_size, endpoint=False, dtype=int)
            
            for i in indices:
                if len(X) >= max_windows:
                    break
                window_signals = group_signals.iloc[i-window_size:i]
                window_y = group_y.iloc[i-window_size:i]
                
                # 提取窗口内的特征
                window_features = self.extract_features(window_signals)
                X.append(window_features)
                
                # 确定窗口标签（如果窗口内有异常，则标记为异常）
                window_label = 1 if np.any(window_y == 1) else 0
                y_windowed.append(window_label)
        
        # 创建特征名称
        if not self.feature_names and X:
            self.feature_names = [f'feat_{i}' for i in range(X[0].shape[0])]
        
        return np.array(X), np.array(y_windowed)

    def train(self, X):
        """使用PyOD训练异常检测模型"""
        # 鲁棒归一化
        X_scaled = self.scaler.fit_transform(X)
        
        # 可以选择不同的PyOD模型
        # 1. 孤立森林
        self.model = IForest(contamination=self.contamination, random_state=self.random_state)
        
        # 2. 单类SVM
        # self.model = OCSVM(contamination=self.contamination)
        
        # 3. 自动编码器
        # self.model = AutoEncoder(epochs=100, batch_size=32, contamination=self.contamination)
        
        self.model.fit(X_scaled)
        print(f"模型 {self.model.__class__.__name__} 训练完成")
        
        return self.model

    def evaluate(self, X, y):
        """评估模型"""
        if not self.model:
            raise ValueError("模型尚未训练")
            
        # 归一化
        X_scaled = self.scaler.transform(X)
        
        # 预测
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # 计算指标
        auc = roc_auc_score(y, y_pred_proba)
        
        print(f"测试集结果 - AUC: {auc:.4f}")
        print("\n分类报告:")
        print(classification_report(y, y_pred))
        
        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig('confusion_matrix_pyod.png')
        
        return {'auc': auc, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba}

    def run_pipeline(self, data_path, annotation_path):
        """运行完整流程"""
        # 加载数据
        print("加载数据...")
        df_signals, df_annotations = self.load_mit_bih_data(data_path)
        annotation_mapping = self.load_annotation_mapping(annotation_path)
        
        # 创建标签（用于评估）
        print("创建评估标签...")
        y = self.create_labels(df_signals, df_annotations, annotation_mapping)
        
        # 过滤伪差
        valid_indices = y != -1
        df_signals = df_signals[valid_indices]
        y = y[valid_indices]
        
        # 检查标签分布
        unique, counts = np.unique(y, return_counts=True)
        print(f"标签分布 - 正常: {counts[0]}, 异常: {counts[1] if len(counts)>1 else 0} ({(counts[1]/len(y)*100 if len(counts)>1 else 0):.1f}%)")
        
        # 创建特征和标签
        print("提取特征...")
        X, y_windowed = self.create_sliding_window_features(df_signals, y, self.window_size)
        print(f"特征矩阵形状: {X.shape}")
        print(f"标签形状: {y_windowed.shape}")
        
        # 检查特征和标签形状是否匹配
        if len(X) != len(y_windowed):
            raise ValueError("特征和标签的样本数量不匹配")
        
        # 划分训练和测试集（时序分割）
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y_windowed[:train_size], y_windowed[train_size:]
        
        # 训练模型
        print("训练模型...")
        self.train(X_train)
        
        # 评估模型
        print("评估模型...")
        evaluation_results = self.evaluate(X_test, y_test)
        
        # 为可视化准备数据（使用PCA降维到2维）
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        
        # 对训练集和测试集进行降维
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # 获取训练集和测试集预测结果
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # 可视化结果
        visualize('异常心率检测结果', X_train_pca, y_train, X_test_pca, y_test, y_train_pred, y_test_pred)
        
        print("流程完成")
        return evaluation_results

if __name__ == "__main__":
    # 数据集路径（保持与原脚本相同）
    DATA_PATH = "D:\智愈派\data\mit-bih"
    ANNOTATION_PATH = "D:\智愈派\data\mit_bih_annotations.csv"
    
    # 创建并运行检测器
    detector = HeartRateAnomalyDetector(window_size=1800, contamination=0.1)
    results = detector.run_pipeline(DATA_PATH, ANNOTATION_PATH)
    
    # 保存结果
    pd.DataFrame({'metric': ['auc'], 'value': [results['auc']]}).to_csv('evaluation_results_pyod.csv', index=False)
    print(f"最终结果 - AUC: {results['auc']:.4f}")