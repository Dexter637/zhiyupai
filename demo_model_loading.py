import numpy as np
from train_pyod_model import HeartRateAnomalyDetector

# 创建检测器实例
print("创建检测器实例...")
detector = HeartRateAnomalyDetector()

# 加载模型
print("\n加载模型...")
detector.load_models()

# 模拟新数据（实际使用时应替换为真实数据）
print("\n生成模拟数据...")
# 假设我们有一个包含42个特征的样本
X_new = np.random.rand(10, 42)  # 10个样本，每个样本42个特征

# 使用优化阈值后的模型进行预测
print("\n使用优化阈值后的模型进行预测...")
y_pred_optimized, y_proba_optimized = detector.predict_with_optimized_model(X_new)
print("预测标签:", y_pred_optimized)
print("预测概率:", y_proba_optimized)

# 使用加权模型进行预测
print("\n使用加权模型进行预测...")
try:
    y_pred_weighted, y_proba_weighted = detector.predict_with_weighted_model(X_new)
    print("预测标签:", y_pred_weighted)
    print("预测概率:", y_proba_weighted)
except ValueError as e:
    print(f"错误: {e}")

print("\n演示完成!")

# 实际应用建议:
# 1. 替换模拟数据为真实的生理信号数据
# 2. 确保数据经过与训练时相同的预处理
# 3. 根据具体应用场景选择合适的模型（优化阈值模型或加权模型）
# 4. 对于医疗监测等高风险场景，建议进一步验证模型性能