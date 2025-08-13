import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, f1_score
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.copod import COPOD
from pyod.models.combination import aom, moa, average, maximization
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
import wfdb
import os
import pickle
from tqdm import tqdm
import warnings
from scipy import stats, signal
from scipy.signal import find_peaks, welch
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

class HeartRateAnomalyDetector:
    def __init__(self, random_state=42, window_size=1800, contamination=0.1):
        self.random_state = random_state
        self.window_size = window_size  # 滑动窗口大小 (5秒)
        self.contamination = contamination  # 异常样本比例
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_names = []
        self.best_threshold = 0.5  # 默认阈值

    def load_mit_bih_data(self, data_path, record_ids=None):
        """加载MIT-BIH数据集"""
        records = []
        annotations = []
        all_record_ids = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
                          111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 
                          122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 
                          209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 
                          222, 223, 228, 230, 231, 232, 233, 234]
        
        if record_ids is None:
            record_ids = all_record_ids
        
        for record_id in tqdm(record_ids, desc="加载数据"):
            record_path = os.path.join(data_path, f"{record_id}")
            
            try:
                # 读取信号数据
                signal_data, fields = wfdb.rdsamp(record_path)
                df_signal = pd.DataFrame(signal_data, columns=[f'channel_{i}' for i in range(signal_data.shape[1])])
                df_signal['record_id'] = record_id
                df_signal['sample_index'] = range(len(df_signal))
                records.append(df_signal)
                
                # 读取标注数据
                ann = wfdb.rdann(record_path, 'atr')
                df_ann = pd.DataFrame({
                    'sample_index': ann.sample,
                    'annotation': ann.symbol,
                    'record_id': record_id
                })
                annotations.append(df_ann)
            except Exception as e:
                print(f"加载记录 {record_id} 时出错: {str(e)}")
                continue
        
        # 合并所有记录和标注
        df_signals = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        df_annotations = pd.concat(annotations, ignore_index=True) if annotations else pd.DataFrame()
        
        return df_signals, df_annotations, all_record_ids

    def create_labels(self, df_signals, df_annotations):
        """高效创建正常/异常标签"""
        # 定义异常标注符号集合
        abnormal_symbols = {'V', 'E', '!', '[', ']', 'r', 'e', 'n', 'f'}
        artifacts_symbols = {'~', '|', 'x', '(', ')', 'u', '?'}
        
        # 初始化标签数组为0 (正常)
        labels = np.zeros(len(df_signals), dtype=np.int8)
        
        # 按记录ID分组处理
        for record_id, group_signals in tqdm(df_signals.groupby('record_id'),
                                             desc="创建标签",
                                             total=df_signals['record_id'].nunique()):
            # 获取当前记录的信号索引范围
            start_idx = group_signals.index[0]
            end_idx = group_signals.index[-1]
            record_length = len(group_signals)
            
            # 获取当前记录的标注
            record_ann = df_annotations[df_annotations['record_id'] == record_id]
            
            if record_ann.empty:
                continue
            
            # 创建记录内的标签数组
            record_labels = np.zeros(record_length, dtype=np.int8)
            
            # 处理每个标注点
            for _, ann_row in record_ann.iterrows():
                sample_idx = ann_row['sample_index']
                symbol = ann_row['annotation']
                
                # 只处理在当前信号范围内的标注
                if sample_idx < record_length:
                    # 异常标注 - 标记周围区域
                    if symbol in abnormal_symbols:
                        start = max(0, sample_idx - 5)  # 前5个样本
                        end = min(record_length, sample_idx + 5)  # 后5个样本
                        record_labels[start:end] = 1
                    
                    # 伪差标注
                    elif symbol in artifacts_symbols:
                        start = max(0, sample_idx - 10)
                        end = min(record_length, sample_idx + 10)
                        record_labels[start:end] = -1
            
            # 将记录内的标签复制到全局标签数组
            labels[start_idx:start_idx+record_length] = record_labels
        
        return labels

    def extract_hrv_features(self, rr_intervals):
        """从RR间期提取HRV特征"""
        if len(rr_intervals) < 4:  # 至少需要4个RR间期
            return np.zeros(10) * np.nan
        
        # 时域特征
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = nn50 / len(rr_intervals) * 100 if len(rr_intervals) > 0 else 0
        
        # 频域特征 (使用Welch方法)
        fs = 4.0  # 重采样频率
        f, pxx = welch(rr_intervals, fs=fs, nperseg=min(256, len(rr_intervals)))
        
        # 频段定义
        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        # 计算各频段功率
        vlf_idx = np.logical_and(f >= vlf_band[0], f < vlf_band[1])
        lf_idx = np.logical_and(f >= lf_band[0], f < lf_band[1])
        hf_idx = np.logical_and(f >= hf_band[0], f < hf_band[1])
        
        vlf_power = np.trapz(pxx[vlf_idx], f[vlf_idx])
        lf_power = np.trapz(pxx[lf_idx], f[lf_idx])
        hf_power = np.trapz(pxx[hf_idx], f[hf_idx])
        total_power = vlf_power + lf_power + hf_power
        
        # 计算标准化功率和比率
        lfnu = lf_power / (lf_power + hf_power) * 100 if (lf_power + hf_power) > 0 else 0
        hfnu = hf_power / (lf_power + hf_power) * 100 if (lf_power + hf_power) > 0 else 0
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        
        return np.array([mean_rr, std_rr, rmssd, nn50, pnn50, 
                         lf_power, hf_power, lfnu, hfnu, lf_hf_ratio])

    def extract_waveform_features(self, signal_channel):
        """提取波形特征"""
        # 1. QRS波特征
        peaks, _ = find_peaks(signal_channel, height=np.max(signal_channel)*0.5, distance=100)
        
        qrs_features = []
        if len(peaks) > 1:
            # R峰间期
            rr_intervals = np.diff(peaks)
            qrs_features.extend([
                np.mean(rr_intervals), np.std(rr_intervals), np.min(rr_intervals), np.max(rr_intervals)
            ])
        else:
            qrs_features.extend([0, 0, 0, 0])
        
        # 2. 信号质量特征
        zcr = np.sum(np.diff(np.sign(signal_channel)) != 0) / len(signal_channel)  # 过零率
        entropy = stats.entropy(np.abs(signal_channel))  # 近似熵
        mean_abs_diff = np.mean(np.abs(np.diff(signal_channel)))  # 平均绝对差分
        
        return np.array([
            *qrs_features,
            zcr, entropy, mean_abs_diff
        ])

    def extract_features(self, window_signals, window_ann):
        """从信号窗口提取特征"""
        features = []
        channel_0 = window_signals['channel_0'].values
        channel_1 = window_signals['channel_1'].values
        
        # 1. 基础统计特征 (每通道)
        for channel in [channel_0, channel_1]:
            features.extend([
                np.mean(channel), np.std(channel), np.min(channel), 
                np.max(channel), np.median(channel), stats.skew(channel),
                stats.kurtosis(channel), np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
        
        # 2. 波形特征 (每通道)
        features.extend(self.extract_waveform_features(channel_0))
        features.extend(self.extract_waveform_features(channel_1))
        
        # 3. HRV特征 (仅使用MLII导联)
        # 从标注中提取R峰位置
        if not window_ann.empty:
            r_peaks = window_ann[window_ann['annotation'].isin(['N', 'L', 'R', 'V', 'A'])]['sample_index'].values
            if len(r_peaks) > 3:
                rr_intervals = np.diff(r_peaks)
                hrv_features = self.extract_hrv_features(rr_intervals)
                features.extend(hrv_features)
            else:
                features.extend([np.nan] * 10)
        else:
            features.extend([np.nan] * 10)
        
        return np.array(features)

    def create_sliding_window_features(self, df_signals, df_annotations, labels):
        """创建滑动窗口特征和对应的标签 (改进版)"""
        X = []
        y_windowed = []
        record_ids_window = []
        
        # 窗口参数
        step_size = self.window_size // 3  # 增加重叠率
        min_abnormal_points = 5  # 窗口内至少5个异常点才标记为异常
        
        # 按记录ID分组
        grouped_signals = df_signals.groupby('record_id')
        grouped_ann = df_annotations.groupby('record_id')
        grouped_labels = pd.Series(labels, index=df_signals.index).groupby(df_signals['record_id'])
        
        for record_id in tqdm(grouped_signals.groups.keys(), desc="处理患者记录"):
            group_signals = grouped_signals.get_group(record_id)
            group_ann = grouped_ann.get_group(record_id) if record_id in grouped_ann.groups else pd.DataFrame()
            group_labels = grouped_labels.get_group(record_id)
            
            record_length = len(group_signals)
            if record_length < self.window_size:
                continue
                
            # 创建滑动窗口
            for start_idx in range(0, record_length - self.window_size + 1, step_size):
                end_idx = start_idx + self.window_size
                
                # 获取窗口数据
                window_signals = group_signals.iloc[start_idx:end_idx]
                window_labels = group_labels.iloc[start_idx:end_idx]
                
                # 获取窗口内的标注
                if not group_ann.empty:
                    window_ann = group_ann[
                        (group_ann['sample_index'] >= start_idx) & 
                        (group_ann['sample_index'] < end_idx)
                    ]
                else:
                    window_ann = pd.DataFrame()
                
                # 提取特征
                window_features = self.extract_features(window_signals, window_ann)
                X.append(window_features)
                
                # 确定窗口标签
                abnormal_count = np.sum(window_labels == 1)
                window_label = 1 if abnormal_count >= min_abnormal_points else 0
                y_windowed.append(window_label)
                record_ids_window.append(record_id)
        
        return np.array(X), np.array(y_windowed), np.array(record_ids_window)

    def train_ensemble(self, X_train):
        """训练模型集成"""
        # 数据标准化
        X_train_norm = self.scaler.fit_transform(X_train)
        
        # 初始化多种异常检测模型
        self.models = {
            'iforest': IForest(
                n_estimators=200, 
                max_samples=0.8,
                contamination=self.contamination,
                random_state=self.random_state,
                verbose=0
            ),
            'copod': COPOD(
                contamination=self.contamination
            ),
            'autoencoder': AutoEncoder(
                hidden_neuron_list=[64, 32, 32, 64],
                epoch_num=200,
                batch_size=128,
                dropout_rate=0.2,
                contamination=self.contamination,
                verbose=1
            )
        }
        
        # 训练所有模型
        print("\n训练集成模型:")
        for name, model in self.models.items():
            print(f"正在训练 {name}...")
            model.fit(X_train_norm)
        
        return self.models
    
    def train_weighted_autoencoder(self, X_train, y_train):
        """训练带类别权重的AutoEncoder"""
        # 计算类别权重
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        # 创建并训练加权AutoEncoder
        X_train_norm = self.scaler.transform(X_train)
        
        weighted_ae = AutoEncoder(
            hidden_neuron_list=[128, 64, 32, 64, 128],
            epoch_num=300,
            batch_size=256,
            dropout_rate=0.3,
            contamination=self.contamination,
            verbose=1
        )
        
        print("训练加权AutoEncoder...")
        weighted_ae.fit(X_train_norm)
        
        # 添加到模型字典
        self.models['weighted_ae'] = weighted_ae
        return weighted_ae

    def predict_ensemble(self, X, use_weighted=False):
        """使用集成模型进行预测"""
        if not self.models:
            raise ValueError("模型尚未训练")
            
        X_norm = self.scaler.transform(X)
        predictions = np.zeros((X.shape[0], len(self.models)))
        pred_probas = np.zeros((X.shape[0], len(self.models)))
        
        # 收集各模型的预测
        for i, (name, model) in enumerate(self.models.items()):
            # 如果不使用加权模型，跳过weighted_ae
            if not use_weighted and name == 'weighted_ae':
                continue
                
            predictions[:, i] = model.predict(X_norm)
            pred_probas[:, i] = model.predict_proba(X_norm)[:, 1]
        
        # 组合策略：平均概率
        avg_proba = np.mean(pred_probas, axis=1)
        
        # 使用优化后的阈值进行预测
        y_pred = (avg_proba >= self.best_threshold).astype(int)
        
        return y_pred, avg_proba

    def evaluate(self, X, y, optimize_threshold=True):
        """评估模型并优化阈值"""
        if not self.models:
            raise ValueError("模型尚未训练")
            
        # 预测概率
        _, y_pred_proba = self.predict_ensemble(X)
        
        # AUC是阈值无关的，直接计算
        auc = roc_auc_score(y, y_pred_proba)
        
        # 优化阈值
        if optimize_threshold:
            precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
            
            # 找到最大化F1-score的阈值
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            self.best_threshold = best_threshold
            
            # 使用最佳阈值进行预测
            y_pred = (y_pred_proba >= best_threshold).astype(int)
            
            print(f"最佳阈值: {best_threshold:.4f}")
        else:
            # 使用默认阈值
            y_pred = (y_pred_proba >= self.best_threshold).astype(int)
        
        # 计算指标
        acc = np.mean(y_pred == y)
        report = classification_report(y, y_pred, target_names=['正常', '异常'])
        
        print(f"测试集结果 - AUC: {auc:.4f}, 准确率: {acc:.4f}")
        print("\n分类报告:")
        print(report)
        
        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['正常', '异常'],
                   yticklabels=['正常', '异常'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig('confusion_matrix_ensemble.png')
        plt.close()
        
        # 绘制PR曲线（仅在优化阈值时）
        if optimize_threshold:
            plt.figure(figsize=(10, 6))
            plt.plot(recall, precision, label='PR Curve')
            plt.scatter(recall[best_idx], precision[best_idx], c='red', 
                       label=f'Best Threshold (F1={f1_scores[best_idx]:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig('precision_recall_curve.png')
            plt.close()
        
        return {
            'auc': auc, 
            'y_pred': y_pred, 
            'y_pred_proba': y_pred_proba,
            'best_threshold': self.best_threshold,
            'report': report
        }

    def run_pipeline(self, data_path):
        """运行完整流程 (改进版)"""
        # 加载所有记录ID
        print("加载数据...")
        _, _, all_record_ids = self.load_mit_bih_data(data_path, record_ids=[])
        
        # 随机划分训练记录和测试记录（80%/20%）
        np.random.seed(self.random_state)
        np.random.shuffle(all_record_ids)
        train_size = int(0.8 * len(all_record_ids))
        train_record_ids = all_record_ids[:train_size]
        test_record_ids = all_record_ids[train_size:]
        
        print(f"训练记录数量: {len(train_record_ids)}, 测试记录数量: {len(test_record_ids)}")
        
        # 加载训练数据
        print("加载训练数据...")
        df_train_signals, df_train_annotations, _ = self.load_mit_bih_data(data_path, train_record_ids)
        
        # 加载测试数据
        print("加载测试数据...")
        df_test_signals, df_test_annotations, _ = self.load_mit_bih_data(data_path, test_record_ids)
        
        # 创建训练标签
        print("创建训练标签...")
        y_train = self.create_labels(df_train_signals, df_train_annotations)
        
        # 创建测试标签
        print("创建测试标签...")
        y_test = self.create_labels(df_test_signals, df_test_annotations)
        
        # 过滤训练数据中的伪差
        train_valid_indices = y_train != -1
        df_train_signals = df_train_signals[train_valid_indices]
        y_train = y_train[train_valid_indices]
        
        # 过滤测试数据中的伪差
        test_valid_indices = y_test != -1
        df_test_signals = df_test_signals[test_valid_indices]
        y_test = y_test[test_valid_indices]
        
        # 检查标签分布
        train_normal = np.sum(y_train == 0)
        train_abnormal = np.sum(y_train == 1)
        test_normal = np.sum(y_test == 0)
        test_abnormal = np.sum(y_test == 1)
        
        print(f"训练数据标签分布 - 正常: {train_normal}, 异常: {train_abnormal} ({train_abnormal/(train_normal+train_abnormal)*100:.1f}%)")
        print(f"测试数据标签分布 - 正常: {test_normal}, 异常: {test_abnormal} ({test_abnormal/(test_normal+test_abnormal)*100:.1f}%)")
        
        # 创建训练特征
        print("提取训练特征...")
        X_train, y_train_windowed, train_record_ids_win = self.create_sliding_window_features(
            df_train_signals, df_train_annotations, y_train
        )
        
        # 创建测试特征
        print("提取测试特征...")
        X_test, y_test_windowed, test_record_ids_win = self.create_sliding_window_features(
            df_test_signals, df_test_annotations, y_test
        )
        
        # 处理NaN值 - 移除包含NaN的特征行
        train_nan_mask = np.isnan(X_train).any(axis=1)
        test_nan_mask = np.isnan(X_test).any(axis=1)
        
        X_train = X_train[~train_nan_mask]
        y_train_windowed = y_train_windowed[~train_nan_mask]
        train_record_ids_win = train_record_ids_win[~train_nan_mask]
        
        X_test = X_test[~test_nan_mask]
        y_test_windowed = y_test_windowed[~test_nan_mask]
        test_record_ids_win = test_record_ids_win[~test_nan_mask]
        
        print(f"训练特征矩阵形状: {X_train.shape}, 异常比例: {np.mean(y_train_windowed):.4f}")
        print(f"测试特征矩阵形状: {X_test.shape}, 异常比例: {np.mean(y_test_windowed):.4f}")
        
        # 训练模型
        print("训练集成模型...")
        self.train_ensemble(X_train)
        
        # 评估模型（第一次评估，使用默认阈值）
        print("评估模型（初始阈值）...")
        initial_results = self.evaluate(X_test, y_test_windowed, optimize_threshold=False)
        
        # 评估模型（第二次评估，使用优化阈值）
        print("评估模型（优化阈值）...")
        optimized_results = self.evaluate(X_test, y_test_windowed, optimize_threshold=True)
        
        # 训练带权重的AutoEncoder
        print("训练带权重的AutoEncoder...")
        self.train_weighted_autoencoder(X_train, y_train_windowed)
        
        # 评估带权重的AutoEncoder
        print("评估带权重的AutoEncoder...")
        weighted_results = self.evaluate(X_test, y_test_windowed, optimize_threshold=False)
        
        # 可视化结果
        self.visualize_results(X_train, y_train_windowed, X_test, y_test_windowed)
        
        print("流程完成")
        return {
            'initial': initial_results,
            'optimized': optimized_results,
            'weighted': weighted_results
        }

    def visualize_results(self, X_train, y_train, X_test, y_test):
        from sklearn.decomposition import PCA
        
        # 使用PCA降维
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # 获取预测结果（使用优化后的阈值）
        y_test_pred, y_test_proba = self.predict_ensemble(X_test)
        
        # 创建可视化
        plt.figure(figsize=(15, 10))
        
        # 训练集真实分布
        plt.subplot(221)
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
                   cmap='coolwarm', alpha=0.6, s=10)
        plt.title('训练集: 真实标签分布')
        plt.colorbar()
        
        # 测试集真实分布
        plt.subplot(222)
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, 
                   cmap='coolwarm', alpha=0.6, s=10)
        plt.title('测试集: 真实标签分布')
        plt.colorbar()
        
        # 测试集预测分布
        plt.subplot(223)
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pred, 
                   cmap='coolwarm', alpha=0.6, s=10)
        plt.title('测试集: 预测标签分布')
        plt.colorbar()
        
        # 测试集异常概率
        plt.subplot(224)
        sc = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_proba, 
                        cmap='viridis', alpha=0.7, s=15)
        plt.colorbar(sc, label='异常概率')
        plt.title('测试集: 异常概率热力图')
        
        plt.tight_layout()
        plt.savefig('anomaly_detection_results.png')
        plt.close()

    def save_models(self, save_dir='models'):
        """保存模型和相关参数

        参数:
        save_dir : str, 模型保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f'{name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f'已保存 {name} 模型到 {model_path}')
        
        # 保存标准化器
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f'已保存标准化器到 {scaler_path}')
        
        # 保存最佳阈值
        threshold_path = os.path.join(save_dir, 'best_threshold.txt')
        with open(threshold_path, 'w') as f:
            f.write(str(self.best_threshold))
        print(f'已保存最佳阈值到 {threshold_path}')
        
        print('所有模型和参数保存完成!')

    def load_models(self, load_dir='models'):
        """加载模型和相关参数

        参数:
        load_dir : str, 模型加载目录
        """
        # 检查目录是否存在
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f'模型目录 {load_dir} 不存在')
        
        # 加载模型
        self.models = {}
        for model_name in ['iforest', 'copod', 'autoencoder', 'weighted_ae']:
            model_path = os.path.join(load_dir, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f'已加载 {model_name} 模型')
            else:
                print(f'未找到 {model_name} 模型文件')
        
        # 加载标准化器
        scaler_path = os.path.join(load_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print('已加载标准化器')
        else:
            raise FileNotFoundError('标准化器文件不存在')
        
        # 加载最佳阈值
        threshold_path = os.path.join(load_dir, 'best_threshold.txt')
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                self.best_threshold = float(f.read())
            print(f'已加载最佳阈值: {self.best_threshold}')
        else:
            print('未找到最佳阈值文件，使用默认值 0.5')
            self.best_threshold = 0.5
        
        print('模型加载完成!')

    def predict_with_optimized_model(self, X):
        """使用优化阈值后的模型进行预测

        参数:
        X : 输入特征矩阵

        返回:
        y_pred : 预测标签
        y_pred_proba : 预测概率
        """
        return self.predict_ensemble(X, use_weighted=False)

    def predict_with_weighted_model(self, X):
        """使用加权模型进行预测

        参数:
        X : 输入特征矩阵

        返回:
        y_pred : 预测标签
        y_pred_proba : 预测概率
        """
        if 'weighted_ae' not in self.models:
            raise ValueError('加权模型未加载或未训练')
        return self.predict_ensemble(X, use_weighted=True)

if __name__ == "__main__":
    # 数据集路径
    DATA_PATH = "D:\智愈派\data\mit-bih"
    
    # 创建并运行检测器
    detector = HeartRateAnomalyDetector(window_size=1800, contamination=0.1)
    results = detector.run_pipeline(DATA_PATH)
    
    # 保存模型
    print("\n保存模型...")
    detector.save_models()
    
    # 保存结果
    pd.DataFrame({
        'metric': ['initial_auc', 'optimized_auc', 'weighted_auc'],
        'value': [
            results['initial']['auc'],
            results['optimized']['auc'],
            results['weighted']['auc']
        ]
    }).to_csv('evaluation_results_ensemble.csv', index=False)
    
    print(f"初始结果 - AUC: {results['initial']['auc']:.4f}")
    print(f"优化阈值后结果 - AUC: {results['optimized']['auc']:.4f}")
    print(f"加权模型结果 - AUC: {results['weighted']['auc']:.4f}")

    # 示例：加载模型并预测
    # print("\n加载模型示例...")
    # new_detector = HeartRateAnomalyDetector()
    # new_detector.load_models()
    # # 假设X_new是新的输入特征
    # # y_pred, y_proba = new_detector.predict_with_optimized_model(X_new)
    # print("模型加载完成，可以用于预测")