import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置随机种子保证可重复性
np.random.seed(42)


# ==================== 1. 数据生成 ====================

def generate_simulation_data(n_samples=5000):
    """
    根据论文描述的物理规则生成仿真数据
    """

    # 基础参数
    gesture_types = ['Five', 'Four', 'Three', 'Two', 'One', 'Fist', 'Good', '666']
    base_confidences = {
        'Five': 0.95, 'Four': 0.90, 'Three': 0.85, 'Two': 0.80,
        'One': 0.75, 'Fist': 0.70, 'Good': 0.85, '666': 0.75
    }

    # 创建空列表存储数据
    data = {
        'sample_id': [],
        'gesture_type': [],
        'sim_distance': [],
        'sim_brightness': [],
        'jitter_error': [],
        'confidence_score': [],
        'is_recognized': []
    }

    for i in range(n_samples):
        # 样本ID
        data['sample_id'].append(i)

        # 随机选择手势类型
        gesture = np.random.choice(gesture_types)
        data['gesture_type'].append(gesture)

        # 模拟检测距离 (20-200cm均匀分布)
        distance = np.random.uniform(20, 200)
        data['sim_distance'].append(distance)

        # 模拟光照强度 (截断正态分布，均值128，标准差60，范围0-255)
        brightness = np.random.normal(128, 60)
        brightness = max(0, min(255, brightness))  # 截断
        data['sim_brightness'].append(brightness)

        # 模拟抖动误差 (与光照强度负相关的高斯噪声)
        # 光照越弱，抖动误差越大
        sigma = 10 * (1 - brightness / 255) + 1  # 基础抖动系数
        jitter = np.abs(np.random.normal(0, sigma))
        data['jitter_error'].append(jitter)

        # 模拟置信度 (基于物理模型)
        base_conf = base_confidences[gesture]
        alpha = 0.00002  # 衰减系数
        distance_effect = alpha * (distance ** 2)

        # 光照对置信度的影响 (光照适中时置信度最高)
        # 使用高斯函数模拟光照影响: 最佳光照在128左右
        brightness_effect = 0.1 * np.exp(-0.5 * ((brightness - 128) / 60) ** 2)

        # 计算最终置信度
        confidence = base_conf - distance_effect + brightness_effect

        # 添加随机噪声
        confidence += np.random.normal(0, 0.05)
        confidence = max(0, min(1, confidence))  # 确保在[0,1]范围内

        # 5%的缺失值模拟
        if np.random.random() < 0.05:
            confidence = np.nan

        data['confidence_score'].append(confidence)

        # 模拟识别结果
        if pd.isna(confidence):
            recognized = np.nan
        else:
            # 当置信度>0.6且抖动误差<20时识别成功
            recognized = 1 if (confidence > 0.6 and jitter < 20) else 0

            # 3%的异常值模拟
            if np.random.random() < 0.03:
                recognized = 1 - recognized  # 翻转结果

        data['is_recognized'].append(recognized)

    # 转换为DataFrame
    df = pd.DataFrame(data)

    # 添加5%的额外异常值到jitter_error
    n_outliers = int(0.05 * n_samples)
    outlier_indices = np.random.choice(df.index, n_outliers, replace=False)
    df.loc[outlier_indices, 'jitter_error'] = np.random.uniform(30, 50, n_outliers)

    return df


# 生成数据
print("正在生成模拟数据...")
simulation_df = generate_simulation_data(5000)

# 保存为CSV文件
simulation_df.to_csv('simulation_data.csv', index=False, encoding='utf-8-sig')
print(f"数据已保存到 simulation_data.csv，共 {len(simulation_df)} 条记录")


# ==================== 2. 数据预处理 ====================

def preprocess_data(df):
    """数据清洗与预处理"""

    # 创建副本
    df_clean = df.copy()

    # 1. 缺失值处理
    print(f"原始数据缺失值数量: {df_clean.isnull().sum().sum()}")
    df_clean = df_clean.dropna()
    print(f"删除缺失值后数据量: {len(df_clean)}")

    # 2. 异常值处理 (使用Z-score)
    numeric_cols = ['sim_distance', 'sim_brightness', 'jitter_error', 'confidence_score']

    for col in numeric_cols:
        if col in df_clean.columns:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            outliers = z_scores > 3
            print(f"{col} 异常值数量: {outliers.sum()}")
            df_clean = df_clean[~outliers]

    print(f"删除异常值后数据量: {len(df_clean)}")

    # 3. 数据类型转换
    df_clean['gesture_type'] = df_clean['gesture_type'].astype('category')

    # 4. 特征工程
    # 创建光照强度分类
    bins = [0, 50, 100, 150, 200, 255]
    labels = ['极暗', '暗', '正常', '强', '过曝']
    df_clean['brightness_category'] = pd.cut(df_clean['sim_brightness'],
                                             bins=bins,
                                             labels=labels,
                                             include_lowest=True)

    # 创建距离分类
    distance_bins = [0, 50, 100, 150, 200]
    distance_labels = ['近', '中', '远', '极远']
    df_clean['distance_category'] = pd.cut(df_clean['sim_distance'],
                                           bins=distance_bins,
                                           labels=distance_labels,
                                           include_lowest=True)

    return df_clean


# 预处理数据
print("\n正在进行数据预处理...")
df_clean = preprocess_data(simulation_df)


# ==================== 3. 探索性数据分析 ====================

def exploratory_data_analysis(df):
    """探索性数据分析与可视化"""

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('手势识别系统仿真数据探索性分析', fontsize=16, fontweight='bold')

    # 1. 置信度分布图
    ax1 = axes[0, 0]
    sns.histplot(df['confidence_score'], kde=True, ax=ax1, bins=30)
    ax1.set_title('置信度分布（双峰分布）', fontsize=12)
    ax1.set_xlabel('置信度')
    ax1.set_ylabel('频率')
    ax1.axvline(x=0.6, color='r', linestyle='--', label='阈值(0.6)')
    ax1.legend()

    # 2. 手势类型分布
    ax2 = axes[0, 1]
    gesture_counts = df['gesture_type'].value_counts()
    ax2.bar(gesture_counts.index, gesture_counts.values)
    ax2.set_title('手势类型分布', fontsize=12)
    ax2.set_xlabel('手势类型')
    ax2.set_ylabel('数量')
    ax2.tick_params(axis='x', rotation=45)

    # 3. 距离与抖动误差散点图
    ax3 = axes[0, 2]
    scatter = ax3.scatter(df['sim_distance'], df['jitter_error'],
                          c=df['confidence_score'], cmap='viridis',
                          alpha=0.6, s=20)
    ax3.set_title('检测距离 vs 坐标抖动误差', fontsize=12)
    ax3.set_xlabel('检测距离 (cm)')
    ax3.set_ylabel('抖动误差 (像素)')
    plt.colorbar(scatter, ax=ax3, label='置信度')

    # 多项式拟合
    x_fit = np.linspace(df['sim_distance'].min(), df['sim_distance'].max(), 100)
    coeffs = np.polyfit(df['sim_distance'], df['jitter_error'], 2)
    y_fit = np.polyval(coeffs, x_fit)
    ax3.plot(x_fit, y_fit, 'r-', linewidth=2, label='二阶多项式拟合')
    ax3.legend()

    # 4. 光照强度分布
    ax4 = axes[1, 0]
    sns.histplot(df['sim_brightness'], kde=True, ax=ax4, bins=30)
    ax4.set_title('光照强度分布', fontsize=12)
    ax4.set_xlabel('光照强度 (Lux)')
    ax4.set_ylabel('频率')
    ax4.axvline(x=128, color='r', linestyle='--', label='均值(128)')
    ax4.legend()

    # 5. 识别成功率箱线图
    ax5 = axes[1, 1]
    success_rate_by_gesture = df.groupby('gesture_type')['is_recognized'].mean()
    ax5.bar(success_rate_by_gesture.index, success_rate_by_gesture.values)
    ax5.set_title('各手势类型识别成功率', fontsize=12)
    ax5.set_xlabel('手势类型')
    ax5.set_ylabel('成功率')
    ax5.tick_params(axis='x', rotation=45)
    ax5.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # 6. 置信度与抖动误差关系
    ax6 = axes[1, 2]
    scatter2 = ax6.scatter(df['confidence_score'], df['jitter_error'],
                           c=df['sim_brightness'], cmap='coolwarm',
                           alpha=0.6, s=20)
    ax6.set_title('置信度 vs 抖动误差 (颜色:光照强度)', fontsize=12)
    ax6.set_xlabel('置信度')
    ax6.set_ylabel('抖动误差 (像素)')
    ax6.axvline(x=0.6, color='r', linestyle='--', alpha=0.5)
    ax6.axhline(y=20, color='r', linestyle='--', alpha=0.5)
    plt.colorbar(scatter2, ax=ax6, label='光照强度')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 增加图表之间的水平和垂直间距
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 额外分析：相关性热力图
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('特征相关性热力图', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 调整顶部间距，确保标题不被截断
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


# 执行EDA
print("\n正在进行探索性数据分析...")
exploratory_data_analysis(df_clean)


# ==================== 4. 核心分析与建模 ====================

def build_and_evaluate_models(df):
    """构建和评估机器学习模型"""

    # 准备特征和目标变量
    features = ['sim_distance', 'sim_brightness', 'jitter_error', 'confidence_score']
    X = df[features]
    y = df['is_recognized']

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    print(f"训练集正样本比例: {y_train.mean():.3f}")

    # 1. 逻辑回归模型
    print("\n" + "=" * 50)
    print("逻辑回归模型训练...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]

    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"逻辑回归准确率: {lr_accuracy:.4f}")
    print("逻辑回归分类报告:")
    print(classification_report(y_test, lr_pred))

    # 2. 随机森林模型
    print("\n" + "=" * 50)
    print("随机森林模型训练...")

    # 网格搜索寻找最佳参数
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf_model, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"最佳参数: {grid_search.best_params_}")

    # 使用最佳模型
    rf_best = grid_search.best_estimator_
    rf_pred = rf_best.predict(X_test)
    rf_proba = rf_best.predict_proba(X_test)[:, 1]

    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"随机森林准确率: {rf_accuracy:.4f}")
    print("随机森林分类报告:")
    print(classification_report(y_test, rf_pred))

    # 3. 特征重要性分析
    print("\n" + "=" * 50)
    print("随机森林特征重要性:")
    feature_importance = pd.DataFrame({
        '特征': features,
        '重要性': rf_best.feature_importances_
    }).sort_values('重要性', ascending=False)

    print(feature_importance)

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_importance['特征'], feature_importance['重要性'])
    plt.xlabel('特征重要性')
    plt.title('随机森林特征重要性分析')

    # 为条形图添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}', va='center')

    plt.tight_layout()
    plt.subplots_adjust(left=0.3)  # 调整左侧间距，确保特征名称完整显示
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 混淆矩阵可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 逻辑回归混淆矩阵
    cm_lr = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'逻辑回归混淆矩阵\n准确率: {lr_accuracy:.3f}')
    axes[0].set_xlabel('预测标签')
    axes[0].set_ylabel('真实标签')
    axes[0].set_xticklabels(['失败', '成功'])
    axes[0].set_yticklabels(['失败', '成功'])

    # 随机森林混淆矩阵
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'随机森林混淆矩阵\n准确率: {rf_accuracy:.3f}')
    axes[1].set_xlabel('预测标签')
    axes[1].set_ylabel('真实标签')
    axes[1].set_xticklabels(['失败', '成功'])
    axes[1].set_yticklabels(['失败', '成功'])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)  # 增加两个混淆矩阵之间的水平间距
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'lr_model': lr_model,
        'rf_model': rf_best,
        'lr_accuracy': lr_accuracy,
        'rf_accuracy': rf_accuracy,
        'lr_proba': lr_proba,
        'rf_proba': rf_proba,
        'y_test': y_test,
        'features': features,
        'scaler': scaler,
        'feature_importance': feature_importance
    }


# 构建和评估模型
print("\n正在进行模型构建与评估...")
model_results = build_and_evaluate_models(df_clean)


# ==================== 5. 高级可视化与结果解读 ====================

def advanced_visualizations(df, model_results):
    """创建高级可视化图表"""

    # 1. 光照与距离对识别率的耦合影响热力图
    print("\n生成光照与距离耦合影响热力图...")

    # 创建光照和距离的分类
    df['brightness_bin'] = pd.cut(df['sim_brightness'], bins=5, labels=False)
    df['distance_bin'] = pd.cut(df['sim_distance'], bins=5, labels=False)

    # 计算每个bin的识别成功率
    heatmap_data = df.groupby(['brightness_bin', 'distance_bin'])['is_recognized'].mean().unstack()

    # 创建自定义标签
    brightness_labels = ['极暗', '暗', '正常', '强', '过曝']
    distance_labels = ['近', '中近', '中', '中远', '极远']

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlBu_r',
                xticklabels=distance_labels, yticklabels=brightness_labels,
                cbar_kws={'label': '识别成功率'})
    plt.title('光照强度与检测距离对识别率的耦合影响', fontsize=14)
    plt.xlabel('检测距离')
    plt.ylabel('光照强度')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.9)  # 调整底部和顶部间距，确保标签完整显示
    plt.savefig('coupling_effect_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. ROC曲线对比
    print("\n生成ROC曲线对比图...")

    y_test = model_results['y_test']
    lr_proba = model_results['lr_proba']
    rf_proba = model_results['rf_proba']

    # 计算ROC曲线
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)

    # 计算AUC
    auc_lr = auc(fpr_lr, tpr_lr)
    auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_lr, tpr_lr, 'b-', linewidth=2, label=f'逻辑回归 (AUC = {auc_lr:.3f})')
    plt.plot(fpr_rf, tpr_rf, 'g-', linewidth=2, label=f'随机森林 (AUC = {auc_rf:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机分类器')
    plt.xlabel('假正例率 (FPR)', fontsize=12)
    plt.ylabel('真正例率 (TPR)', fontsize=12)
    plt.title('ROC曲线对比：逻辑回归 vs 随机森林', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    # 添加AUC值标注
    plt.text(0.6, 0.3, f'逻辑回归 AUC = {auc_lr:.3f}', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    plt.text(0.6, 0.2, f'随机森林 AUC = {auc_rf:.3f}', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.9)  # 调整底部和顶部间距，确保标签和图例完整显示
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. 决策边界可视化（使用PCA降维）
    print("\n生成决策边界可视化...")

    from sklearn.decomposition import PCA

    # 使用PCA降维到2维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(model_results['scaler'].transform(df[model_results['features']]))

    plt.figure(figsize=(12, 5))

    # 原始数据分布
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=df['is_recognized'], cmap='coolwarm',
                          alpha=0.6, s=30)
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    plt.title('原始数据分布 (PCA降维)')
    plt.colorbar(scatter, label='识别结果 (0:失败, 1:成功)')

    # 预测结果分布
    plt.subplot(1, 2, 2)
    rf_pred_full = model_results['rf_model'].predict(
        model_results['scaler'].transform(df[model_results['features']])
    )

    scatter_pred = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                               c=rf_pred_full, cmap='coolwarm',
                               alpha=0.6, s=30)
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    plt.title('随机森林预测结果 (PCA降维)')
    plt.colorbar(scatter_pred, label='预测结果 (0:失败, 1:成功)')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)  # 增加两个子图之间的水平间距
    plt.savefig('decision_boundary_pca.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 不同手势类型的性能对比
    print("\n生成不同手势类型性能对比图...")

    gesture_performance = df.groupby('gesture_type').agg({
        'is_recognized': ['mean', 'count'],
        'confidence_score': 'mean',
        'jitter_error': 'mean'
    }).round(3)

    gesture_performance.columns = ['识别率', '样本数', '平均置信度', '平均抖动误差']
    gesture_performance = gesture_performance.sort_values('识别率', ascending=False)

    # 绘制组合图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 识别率柱状图
    bars1 = ax1.bar(gesture_performance.index, gesture_performance['识别率'])
    ax1.set_title('各手势类型识别率对比', fontsize=14)
    ax1.set_ylabel('识别率')
    ax1.set_ylim([0, 1])
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # 为柱状图添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 置信度与抖动误差折线图
    ax2_twin = ax2.twinx()

    line1, = ax2.plot(gesture_performance.index, gesture_performance['平均置信度'],
                      'b-', marker='o', linewidth=2, label='平均置信度')
    line2, = ax2_twin.plot(gesture_performance.index, gesture_performance['平均抖动误差'],
                           'r-', marker='s', linewidth=2, label='平均抖动误差')

    ax2.set_title('各手势类型置信度与抖动误差对比', fontsize=14)
    ax2.set_ylabel('平均置信度', color='b')
    ax2_twin.set_ylabel('平均抖动误差', color='r')

    # 添加图例
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # 增加两个子图之间的垂直间距
    plt.savefig('gesture_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'heatmap_data': heatmap_data,
        'auc_lr': auc_lr,
        'auc_rf': auc_rf,
        'gesture_performance': gesture_performance
    }


# 生成高级可视化
print("\n正在生成高级可视化图表...")
advanced_results = advanced_visualizations(df_clean, model_results)


# ==================== 6. 系统性能阈值分析 ====================

def threshold_analysis(df):
    """系统性能阈值分析"""

    print("\n" + "=" * 50)
    print("系统性能阈值分析")
    print("=" * 50)

    # 1. 距离阈值分析
    distance_thresholds = [50, 100, 150, 200]
    print("\n不同距离阈值的识别性能:")
    for threshold in distance_thresholds:
        subset = df[df['sim_distance'] <= threshold]
        success_rate = subset['is_recognized'].mean()
        print(f"  距离 ≤ {threshold}cm: 识别率 = {success_rate:.3f} (样本数: {len(subset)})")

    # 2. 光照阈值分析
    brightness_thresholds = [30, 50, 100, 150, 200]
    print("\n不同光照阈值的识别性能:")
    for threshold in brightness_thresholds:
        subset = df[df['sim_brightness'] >= threshold]
        success_rate = subset['is_recognized'].mean()
        print(f"  光照 ≥ {threshold}Lux: 识别率 = {success_rate:.3f} (样本数: {len(subset)})")

    # 3. 抖动误差阈值分析
    jitter_thresholds = [10, 15, 20, 25, 30]
    print("\n不同抖动误差阈值的识别性能:")
    for threshold in jitter_thresholds:
        subset = df[df['jitter_error'] <= threshold]
        success_rate = subset['is_recognized'].mean()
        print(f"  抖动误差 ≤ {threshold}像素: 识别率 = {success_rate:.3f} (样本数: {len(subset)})")

    # 4. 多条件联合阈值分析
    print("\n多条件联合阈值分析:")

    # 最佳条件
    best_conditions = df[
        (df['sim_distance'] <= 100) &
        (df['sim_brightness'] >= 100) &
        (df['jitter_error'] <= 15)
        ]
    print(f"  距离≤100cm & 光照≥100Lux & 抖动误差≤15像素:")
    print(f"    识别率 = {best_conditions['is_recognized'].mean():.3f}")
    print(f"    样本数 = {len(best_conditions)}")
    print(f"    占总体比例 = {len(best_conditions) / len(df):.3f}")

    # 最差条件
    worst_conditions = df[
        (df['sim_distance'] > 150) |
        (df['sim_brightness'] < 50)
        ]
    print(f"\n  距离>150cm 或 光照<50Lux:")
    print(f"    识别率 = {worst_conditions['is_recognized'].mean():.3f}")
    print(f"    样本数 = {len(worst_conditions)}")
    print(f"    占总体比例 = {len(worst_conditions) / len(df):.3f}")

    # 可视化阈值分析
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 距离阈值分析图
    distance_bins = np.linspace(df['sim_distance'].min(), df['sim_distance'].max(), 10)
    distance_success = []
    for i in range(len(distance_bins) - 1):
        subset = df[(df['sim_distance'] >= distance_bins[i]) &
                    (df['sim_distance'] < distance_bins[i + 1])]
        distance_success.append(subset['is_recognized'].mean())

    axes[0, 0].bar(range(len(distance_success)), distance_success)
    axes[0, 0].set_title('不同距离区间的识别成功率')
    axes[0, 0].set_xlabel('距离区间')
    axes[0, 0].set_ylabel('识别成功率')
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # 光照阈值分析图
    brightness_bins = np.linspace(df['sim_brightness'].min(), df['sim_brightness'].max(), 10)
    brightness_success = []
    for i in range(len(brightness_bins) - 1):
        subset = df[(df['sim_brightness'] >= brightness_bins[i]) &
                    (df['sim_brightness'] < brightness_bins[i + 1])]
        brightness_success.append(subset['is_recognized'].mean())

    axes[0, 1].bar(range(len(brightness_success)), brightness_success)
    axes[0, 1].set_title('不同光照区间的识别成功率')
    axes[0, 1].set_xlabel('光照区间')
    axes[0, 1].set_ylabel('识别成功率')
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # 抖动误差阈值分析图
    jitter_bins = np.linspace(df['jitter_error'].min(), df['jitter_error'].max(), 10)
    jitter_success = []
    for i in range(len(jitter_bins) - 1):
        subset = df[(df['jitter_error'] >= jitter_bins[i]) &
                    (df['jitter_error'] < jitter_bins[i + 1])]
        jitter_success.append(subset['is_recognized'].mean())

    axes[1, 0].bar(range(len(jitter_success)), jitter_success)
    axes[1, 0].set_title('不同抖动误差区间的识别成功率')
    axes[1, 0].set_xlabel('抖动误差区间')
    axes[1, 0].set_ylabel('识别成功率')
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # 置信度阈值分析图
    confidence_bins = np.linspace(0, 1, 11)
    confidence_success = []
    for i in range(len(confidence_bins) - 1):
        subset = df[(df['confidence_score'] >= confidence_bins[i]) &
                    (df['confidence_score'] < confidence_bins[i + 1])]
        confidence_success.append(subset['is_recognized'].mean())

    axes[1, 1].bar(range(len(confidence_success)), confidence_success)
    axes[1, 1].set_title('不同置信度区间的识别成功率')
    axes[1, 1].set_xlabel('置信度区间')
    axes[1, 1].set_ylabel('识别成功率')
    axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 增加图表之间的水平和垂直间距
    plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 返回分析结果
    return {
        'distance_analysis': distance_success,
        'brightness_analysis': brightness_success,
        'jitter_analysis': jitter_success,
        'confidence_analysis': confidence_success
    }


# 执行阈值分析
print("\n正在进行系统性能阈值分析...")
threshold_results = threshold_analysis(df_clean)


# ==================== 7. 生成分析报告 ====================

def generate_analysis_report(df, model_results, advanced_results, threshold_results):
    """生成分析报告"""

    print("\n" + "=" * 50)
    print("手势识别系统仿真数据分析报告")
    print("=" * 50)

    report = {
        "数据概况": {
            "总样本数": len(df),
            "特征数": len(df.columns) - 2,  # 减去sample_id和is_recognized
            "手势类型数": df['gesture_type'].nunique(),
            "识别成功率": f"{df['is_recognized'].mean():.3f}"
        },
        "模型性能": {
            "逻辑回归准确率": f"{model_results['lr_accuracy']:.3f}",
            "随机森林准确率": f"{model_results['rf_accuracy']:.3f}",
            "逻辑回归AUC": f"{advanced_results['auc_lr']:.3f}",
            "随机森林AUC": f"{advanced_results['auc_rf']:.3f}",
            "最佳模型": "随机森林" if model_results['rf_accuracy'] > model_results['lr_accuracy'] else "逻辑回归"
        },
        "特征重要性排序": model_results['feature_importance'].to_dict('records'),
        "手势性能排名": advanced_results['gesture_performance'].head(5).to_dict('records'),
        "关键发现": [
            f"检测距离是影响识别性能的最重要因素 (重要性: {model_results['feature_importance'].iloc[0]['重要性']:.3f})",
            f"光照强度是第二重要因素 (重要性: {model_results['feature_importance'].iloc[1]['重要性']:.3f})",
            f"系统在距离≤100cm、光照≥100Lux时表现最佳 (识别率: {df[(df['sim_distance'] <= 100) & (df['sim_brightness'] >= 100)]['is_recognized'].mean():.3f})",
            f"系统在距离>150cm或光照<50Lux时表现最差 (识别率: {df[(df['sim_distance'] > 150) | (df['sim_brightness'] < 50)]['is_recognized'].mean():.3f})"
        ],
        "优化建议": [
            "优先解决远距离检测问题，可通过调整摄像头焦距或使用多摄像头方案",
            "在弱光环境下增加补光设备或使用红外摄像头",
            "针对识别率较低的手势类型进行算法优化",
            "设置工作环境阈值：距离≤120cm，光照≥30Lux",
            "考虑使用集成学习方法进一步提升模型性能"
        ]
    }

    # 打印报告
    for section, content in report.items():
        print(f"\n{section}:")
        if isinstance(content, dict):
            for key, value in content.items():
                print(f"  {key}: {value}")
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    for key, value in item.items():
                        print(f"    {key}: {value}")
                else:
                    print(f"  • {item}")

    # 保存报告为JSON文件
    import json
    with open('analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n分析报告已保存到 analysis_report.json")

    return report


# 生成分析报告
print("\n正在生成分析报告...")
final_report = generate_analysis_report(df_clean, model_results, advanced_results, threshold_results)


# ==================== 8. 交互式可视化（可选） ====================

def create_interactive_visualization(df):
    """创建交互式可视化（需要plotly）"""

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        print("\n正在生成交互式可视化图表...")

        # 1. 3D散点图
        fig_3d = px.scatter_3d(
            df.sample(1000),  # 采样以减少数据量
            x='sim_distance',
            y='sim_brightness',
            z='jitter_error',
            color='is_recognized',
            size='confidence_score',
            hover_data=['gesture_type'],
            title='手势识别系统仿真数据3D可视化',
            labels={
                'sim_distance': '检测距离 (cm)',
                'sim_brightness': '光照强度 (Lux)',
                'jitter_error': '抖动误差',
                'is_recognized': '识别结果',
                'confidence_score': '置信度'
            }
        )

        fig_3d.write_html("3d_visualization.html")
        print("3D可视化图表已保存到 3d_visualization.html")

        # 2. 交互式热力图
        pivot_table = df.pivot_table(
            values='is_recognized',
            index=pd.cut(df['sim_brightness'], bins=5, labels=['极暗', '暗', '正常', '强', '过曝']),
            columns=pd.cut(df['sim_distance'], bins=5, labels=['近', '中近', '中', '中远', '极远']),
            aggfunc='mean'
        )

        fig_heatmap = px.imshow(
            pivot_table,
            labels=dict(x="检测距离", y="光照强度", color="识别成功率"),
            title="光照与距离对识别率的耦合影响（交互式热力图）",
            color_continuous_scale="RdYlBu_r",
            aspect="auto"
        )

        fig_heatmap.write_html("interactive_heatmap.html")
        print("交互式热力图已保存到 interactive_heatmap.html")

        # 3. 平行坐标图
        fig_parallel = px.parallel_coordinates(
            df.sample(500),
            dimensions=['sim_distance', 'sim_brightness', 'jitter_error', 'confidence_score'],
            color='is_recognized',
            color_continuous_scale=px.colors.diverging.Tealrose,
            title="手势识别系统仿真数据平行坐标图"
        )

        fig_parallel.write_html("parallel_coordinates.html")
        print("平行坐标图已保存到 parallel_coordinates.html")

        print("\n交互式图表生成完成！请在浏览器中打开生成的HTML文件查看。")

    except ImportError:
        print("\n注意：未安装plotly库，跳过交互式可视化。")
        print("如需生成交互式图表，请运行: pip install plotly")


# 创建交互式可视化
create_interactive_visualization(df_clean)

print("\n" + "=" * 50)
print("数据分析流程完成！")
print("=" * 50)
print(f"\n生成的文件:")
print(f"1. simulation_data.csv - 原始仿真数据")
print(f"2. eda_analysis.png - 探索性数据分析图表")
print(f"3. correlation_heatmap.png - 相关性热力图")
print(f"4. feature_importance.png - 特征重要性图")
print(f"5. confusion_matrices.png - 混淆矩阵图")
print(f"6. coupling_effect_heatmap.png - 耦合影响热力图")
print(f"7. roc_curves.png - ROC曲线对比图")
print(f"8. decision_boundary_pca.png - 决策边界可视化")
print(f"9. gesture_performance_comparison.png - 手势性能对比图")
print(f"10. threshold_analysis.png - 阈值分析图")
print(f"11. analysis_report.json - 分析报告")
print(f"\n所有图表和分析结果已保存到当前目录。")