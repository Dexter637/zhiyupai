import numpy as np
import os
import pandas as pd
import wfdb
from train_pyod_model import HeartRateAnomalyDetector

# 创建检测器实例
print("创建检测器实例...")
detector = HeartRateAnomalyDetector(window_size=1800)  # 设置与训练时一致的窗口大小

detector.window_size = 1800  # 确保窗口大小正确设置

# 加载模型
print("加载模型...")
detector.load_models()

# MIT-BIH数据库路径
mit_bih_path = "d:\智愈派\data\mit-bih"

# 获取所有记录文件名
record_files = [f for f in os.listdir(mit_bih_path) if f.endswith('.hea')]
record_names = [f[:-4] for f in record_files]

print(f"找到 {len(record_names)} 个MIT-BIH记录文件")

# 初始化总测评数据变量
total_original_samples = 0
total_clean_samples = 0
total_predicted_abnormal = 0
actual_abnormal = 0
true_positives = 0
false_positives = 0

# 处理每个记录
for record_name in record_names:  # 处理所有记录
    print(f"\n处理记录: {record_name}")
    
    # 读取记录
    record = wfdb.rdrecord(os.path.join(mit_bih_path, record_name))
    annotation = wfdb.rdann(os.path.join(mit_bih_path, record_name), 'atr')
    
    # 提取心率数据 (所有通道)
    signal_data = record.p_signal
    num_channels = signal_data.shape[1]

    # 创建与信号数据长度一致的标签数组
    labels = np.zeros(len(signal_data), dtype=int)

    # 定义异常标注符号集合
    abnormal_symbols = {'V', 'E', '!', '[', ']', 'r', 'e', 'n', 'f'}
    artifacts_symbols = {'~', '|', 'x', '(', ')', 'u', '?'}

    # 将异常和伪差标注位置及其周围区域标记为异常
    for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
        if sample_idx < len(labels):
            # 异常标注 - 标记周围区域
            if symbol in abnormal_symbols:
                start = max(0, sample_idx - 5)  # 前5个样本
                end = min(len(labels), sample_idx + 5)  # 后5个样本
                labels[start:end] = 1
            
            # 伪差标注 - 标记周围更大区域
            elif symbol in artifacts_symbols:
                start = max(0, sample_idx - 10)  # 前10个样本
                end = min(len(labels), sample_idx + 10)  # 后10个样本
                labels[start:end] = -1  # 与训练脚本一致，伪差标记为-1
    
    # 提取特征 (这里使用检测器的特征提取方法)
    # 注意: 需要将numpy数组转换为DataFrame以符合方法要求
    try:
        # 将numpy数组转换为DataFrame，包含所有通道
        channels = [f'channel_{i}' for i in range(num_channels)]
        df_signals = pd.DataFrame(signal_data, columns=channels)
        
        # 如果只有1个通道，复制一份作为channel_1以匹配训练数据格式
        if num_channels == 1:
            df_signals['channel_1'] = df_signals['channel_0']
        df_signals['record_id'] = record_name  # 添加记录ID
        df_signals['sample_index'] = np.arange(len(signal_data))  # 添加样本索引
        
        # 创建标签DataFrame
        df_annotations = pd.DataFrame({
            'sample_index': annotation.sample,
            'annotation': annotation.symbol,
            'record_id': [record_name] * len(annotation.sample)
        })
        
        # 创建滑动窗口特征
        window_size = 1800  # 与训练时保持一致的窗口大小 (5秒)
        try:
            # 正确接收3个返回值
            X, y, record_ids = detector.create_sliding_window_features(df_signals, df_annotations, labels)
        except ValueError as e:
            print(f"记录 {record_name} 特征创建失败: {e}")
            continue
        
        if len(X) == 0:
            print(f"记录 {record_name} 没有足够数据创建特征")
            continue
        
        # 处理缺失值
        print(f"处理记录 {record_name} 的缺失值...")
        # 选项1: 删除包含NaN的样本
        non_nan_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[non_nan_mask]
        y_clean = y[non_nan_mask]
        
        if len(X_clean) == 0:
            print(f"记录 {record_name} 清理后没有足够数据进行预测")
            continue
        
        # 使用带权重的模型进行预测
        print(f"使用带权重模型预测记录 {record_name}...")
        try:
            y_pred_optimized, y_proba_optimized = detector.predict_with_weighted_model(X_clean)
        except ValueError as e:
            print(f"带权重模型预测失败: {e}")
            # 回退到使用优化阈值模型
            print(f"回退到使用最佳阈值模型预测...")
            y_pred_optimized, y_proba_optimized = detector.predict_with_optimized_model(X_clean)
        
        # 输出结果
        print(f"记录 {record_name} 预测结果:")
        print(f"原始样本数: {len(X)}")
        print(f"清理后样本数: {len(X_clean)}")
        print(f"预测异常样本数: {np.sum(y_pred_optimized)}")
        print(f"实际异常样本数: {np.sum(y_clean)}")
        
        # 计算准确率
        accuracy = np.mean(y_pred_optimized == y_clean)
        print(f"预测准确率: {accuracy:.4f}")
        
        # 计算精确率和召回率
        if np.sum(y_pred_optimized) > 0:
            precision = np.sum((y_pred_optimized == 1) & (y_clean == 1)) / np.sum(y_pred_optimized)
        else:
            precision = 0
        
        if np.sum(y_clean) > 0:
            recall = np.sum((y_pred_optimized == 1) & (y_clean == 1)) / np.sum(y_clean)
        else:
            recall = 0
        
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        
        # 更新总测评数据
        total_original_samples += len(X)
        total_clean_samples += len(X_clean)
        total_predicted_abnormal += np.sum(y_pred_optimized)
        actual_abnormal += np.sum(y_clean)
        true_positives += np.sum((y_pred_optimized == 1) & (y_clean == 1))
        false_positives += np.sum((y_pred_optimized == 1) & (y_clean == 0))
        
    except Exception as e:
        print(f"处理记录 {record_name} 时出错: {e}")

# 计算总测评指标
print("\n总测评数据:")
print(f"总原始样本数: {total_original_samples}")
print(f"总清理后样本数: {total_clean_samples}")
print(f"总预测异常样本数: {total_predicted_abnormal}")
print(f"总实际异常样本数: {actual_abnormal}")

# 计算总准确率
if total_clean_samples > 0:
    total_accuracy = (total_clean_samples - (false_positives + (actual_abnormal - true_positives))) / total_clean_samples
    print(f"总准确率: {total_accuracy:.4f}")
else:
    print("总准确率: 0.0000")

# 计算总精确率
if total_predicted_abnormal > 0:
    total_precision = true_positives / total_predicted_abnormal
    print(f"总精确率: {total_precision:.4f}")
else:
    print("总精确率: 0.0000")

# 计算总召回率
if actual_abnormal > 0:
    total_recall = true_positives / actual_abnormal
    print(f"总召回率: {total_recall:.4f}")
else:
    print("总召回率: 0.0000")

print("\nMIT数据库文件处理完成!")