import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import lightgbm as lgb
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from imblearn.over_sampling import SMOTE
import shap
from sklearn.ensemble import IsolationForest

# 1. 数据加载与预处理
def load_data(path):
    df = pd.read_csv(path)
    # MIT数据库特定处理：处理缺失值和数据对齐
    df = df.interpolate(method='linear', limit_direction='both')
    # 时间戳对齐（假设有timestamp列）
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    return df

# 2. 高级时序特征工程
def extract_time_features(df, window_size=5):
    """提取时序统计特征"""
    features = []
    for col in df.columns.drop(['timestamp', 'label']):
        # 滑动窗口特征
        df[f'{col}_rolling_mean'] = df[col].rolling(window_size).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window_size).std()
        df[f'{col}_diff'] = df[col].diff()
        
        # 频域特征（FFT变换）
        fft = np.fft.fft(df[col].values)
        df[f'{col}_fft_real'] = np.real(fft)
        df[f'{col}_fft_imag'] = np.imag(fft)
    
    # 使用tsfresh自动提取特征（适用于时序数据）
    tsfresh_features = extract_features(
            df.drop('label', axis=1),
            column_id='patient_id',  # 假设有患者ID列
            column_sort='timestamp',
            impute_function=impute,
            n_jobs=1
        )
    return pd.concat([df, tsfresh_features], axis=1)

# 3. 处理类别不平衡
def balance_data(X, y):
    """混合采样策略"""
    # 第一步：使用SMOTE生成少数类样本
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # 第二步：使用NearMiss降低多数类样本
    from imblearn.under_sampling import NearMiss
    nm = NearMiss(version=2, sampling_strategy=0.7)
    return nm.fit_resample(X_res, y_res)

# 4. 时序交叉验证
def time_series_cv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = {'f1': [], 'auc': []}
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 仅对训练集做平衡处理
        X_train_bal, y_train_bal = balance_data(X_train, y_train)
        
        # 自适应归一化（使用RobustScaler处理离群值）
        scaler = RobustScaler().fit(X_train_bal)
        X_train_scaled = scaler.transform(X_train_bal)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        model.fit(X_train_scaled, y_train_bal,
                  eval_set=[(X_test_scaled, y_test)],
                  early_stopping_rounds=50,
                  verbose=False)
        
        # 评估
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['auc'].append(roc_auc_score(y_test, y_prob))
    
    print(f"平均F1: {np.mean(metrics['f1']):.4f}, 平均AUC: {np.mean(metrics['auc']):.4f}")
    return metrics

# 5. 对抗验证检测数据分布偏移
def adversarial_validation(X_train, X_test):
    """检测训练集和测试集分布差异"""
    combined = pd.concat([X_train, X_test])
    labels = np.array([0]*len(X_train) + [1]*len(X_test))
    
    # 训练分类器区分来源
    adv_model = lgb.LGBMClassifier()
    adv_model.fit(combined, labels)
    
    # 计算AUC
    preds = adv_model.predict_proba(combined)[:,1]
    auc = roc_auc_score(labels, preds)
    print(f"对抗验证AUC: {auc:.4f} (接近0.5表示分布一致)")

# 6. 特征选择
def select_features(X, y, threshold=0.01):
    """基于互信息的特征选择"""
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X, y)
    selected = X.columns[mi > threshold]
    print(f"从{X.shape[1]}特征中选择{len(selected)}个重要特征")
    return selected

# 主流程
if __name__ == "__main__":
    # 加载数据
    df = load_data("mit_heart_data.csv")
    
    # 特征工程
    df_features = extract_time_features(df)
    
    # 分离特征和标签
    X = df_features.drop('label', axis=1)
    y = df['label']
    
    # 特征选择
    selected_features = select_features(X, y)
    X = X[selected_features]
    
    # 划分时序数据集（保留时间顺序）
    split_idx = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 对抗验证
    adversarial_validation(X_train, X_test)
    
    # 配置LightGBM参数
    params = {
        'objective': 'binary',
        'boosting_type': 'goss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': 9,  # 正负样本比例
        'metric': ['auc', 'binary_logloss'],
        'n_jobs': -1,
        'seed': 42
    }
    
    # 初始化模型
    model = lgb.LGBMClassifier(**params, n_estimators=10000)
    
    # 时序交叉验证
    metrics = time_series_cv(model, X_train, y_train)
    
    # 训练最终模型
    X_train_bal, y_train_bal = balance_data(X_train, y_train)
    scaler = RobustScaler().fit(X_train_bal)
    X_train_scaled = scaler.transform(X_train_bal)
    model.fit(X_train_scaled, y_train_bal)
    
    # 测试集评估
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n测试集性能:")
    print(classification_report(y_test, y_pred))
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    # 解释性分析
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    shap.summary_plot(shap_values[1], X_test, plot_type="bar")
    
    # 保存模型
    model.booster_.save_model('heart_anomaly_model.txt')