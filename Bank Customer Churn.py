import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("=" * 80)
print("银行客户流失预测分析项目")
print("=" * 80)

# 1. 数据加载
print("\n1. 数据加载")
print("-" * 40)

df = pd.read_csv('Churn_Modelling.csv')
print(f"✅ 数据加载成功！数据形状: {df.shape}")
print(f"数据列: {df.columns.tolist()}")

# 2. 数据探索性分析（EDA）
print("\n2. 数据探索性分析")
print("-" * 40)

print("\n2.1 数据基本信息:")
print(df.info())

print("\n2.2 描述性统计:")
print(df.describe())

print("\n2.3 缺失值检查:")
print(df.isnull().sum())

print("\n2.4 目标变量分布（客户流失情况）:")
print(df['Exited'].value_counts())
print(f"流失率: {df['Exited'].mean() * 100:.2f}%")

# 3. 数据清洗和预处理
print("\n3. 数据清洗和预处理")
print("-" * 40)

# 删除不需要的列
df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 处理分类变量
df_clean = pd.get_dummies(df_clean, columns=['Geography', 'Gender'], drop_first=True)

print(f"清洗后数据形状: {df_clean.shape}")
print(f"清洗后数据列: {df_clean.columns.tolist()}")

# 4. 客户画像分析
print("\n4. 客户画像分析")
print("-" * 40)

# 流失客户vs非流失客户对比
print("\n4.1 流失客户平均特征:")
print(df[df['Exited'] == 1][['Age', 'Balance', 'CreditScore', 'EstimatedSalary']].mean().round(2))

print("\n4.2 非流失客户平均特征:")
print(df[df['Exited'] == 0][['Age', 'Balance', 'CreditScore', 'EstimatedSalary']].mean().round(2))

print("\n4.3 流失客户vs非流失客户对比:")
comparison = pd.DataFrame({
    '特征': ['年龄', '余额', '信用评分', '预计薪资'],
    '流失客户': df[df['Exited'] == 1][['Age', 'Balance', 'CreditScore', 'EstimatedSalary']].mean().values,
    '非流失客户': df[df['Exited'] == 0][['Age', 'Balance', 'CreditScore', 'EstimatedSalary']].mean().values,
    '差异': df[df['Exited'] == 1][['Age', 'Balance', 'CreditScore', 'EstimatedSalary']].mean().values -
            df[df['Exited'] == 0][['Age', 'Balance', 'CreditScore', 'EstimatedSalary']].mean().values
})
print(comparison.round(2))

# 5. 数据可视化
print("\n5. 数据可视化")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 图1: 年龄分布与流失关系
ax1 = axes[0, 0]
df[df['Exited'] == 0]['Age'].hist(alpha=0.5, label='未流失', bins=30, color='blue', ax=ax1)
df[df['Exited'] == 1]['Age'].hist(alpha=0.5, label='流失', bins=30, color='red', ax=ax1)
ax1.set_xlabel('年龄')
ax1.set_ylabel('人数')
ax1.set_title('年龄分布与客户流失关系')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: 余额分布
ax2 = axes[0, 1]
df[df['Exited'] == 0]['Balance'].hist(alpha=0.5, label='未流失', bins=50, color='blue', ax=ax2)
df[df['Exited'] == 1]['Balance'].hist(alpha=0.5, label='流失', bins=50, color='red', ax=ax2)
ax2.set_xlabel('余额')
ax2.set_ylabel('人数')
ax2.set_title('余额分布与客户流失关系')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3: 产品数量分布
ax3 = axes[0, 2]
product_counts = pd.crosstab(df['NumOfProducts'], df['Exited'])
product_counts.columns = ['未流失', '流失']
product_counts.plot(kind='bar', ax=ax3, color=['blue', 'red'])
ax3.set_xlabel('产品数量')
ax3.set_ylabel('客户数量')
ax3.set_title('产品数量与流失关系')
ax3.legend(['未流失', '流失'])
ax3.grid(True, alpha=0.3)

# 图4: 地理分布
ax4 = axes[1, 0]
geo_churn = pd.crosstab(df['Geography'], df['Exited'])
geo_churn.columns = ['未流失', '流失']
geo_churn.plot(kind='bar', ax=ax4, color=['blue', 'red'])
ax4.set_xlabel('地区')
ax4.set_ylabel('客户数量')
ax4.set_title('各地区客户流失情况')
ax4.legend(['未流失', '流失'])
ax4.grid(True, alpha=0.3)

# 图5: 性别分布
ax5 = axes[1, 1]
gender_churn = pd.crosstab(df['Gender'], df['Exited'])
gender_churn.columns = ['未流失', '流失']
gender_churn.plot(kind='bar', ax=ax5, color=['blue', 'red'])
ax5.set_xlabel('性别')
ax5.set_ylabel('客户数量')
ax5.set_title('性别与客户流失关系')
ax5.legend(['未流失', '流失'])
ax5.grid(True, alpha=0.3)

# 图6: 活跃会员流失情况
ax6 = axes[1, 2]
active_churn = pd.crosstab(df['IsActiveMember'], df['Exited'])
active_churn.columns = ['未流失', '流失']
active_churn.index = ['非活跃', '活跃']
active_churn.plot(kind='bar', ax=ax6, color=['blue', 'red'])
ax6.set_xlabel('是否活跃会员')
ax6.set_ylabel('客户数量')
ax6.set_title('活跃度与客户流失关系')
ax6.legend(['未流失', '流失'])
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('银行客户流失分析.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ 图表已保存为 '银行客户流失分析.png'")

# 6. 相关性分析
print("\n6. 相关性分析")
print("-" * 40)

# 计算数值型变量的相关性
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary', 'Exited']
correlation = df[numeric_cols].corr()

print("\n与客户流失的相关性:")
print(correlation['Exited'].sort_values(ascending=False).round(4))

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.3f')
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig('相关性热力图.png', dpi=300)
plt.show()
print("✅ 热力图已保存为 '相关性热力图.png'")

# 7. 客户价值分层
print("\n7. 客户价值分层")
print("-" * 40)

# 创建客户价值分层
df['Balance_Level'] = pd.cut(df['Balance'],
                             bins=[-1, 0, 50000, 100000, 150000, df['Balance'].max()],
                             labels=['无余额', '低余额(0-5万)', '中余额(5-10万)', '高余额(10-15万)', '超高余额(15万+)'])

# 各价值层级的流失率
value_churn = df.groupby('Balance_Level')['Exited'].agg(['mean', 'count'])
value_churn.columns = ['流失率', '客户数']
value_churn['流失率'] = value_churn['流失率'] * 100
print("\n各余额层级流失情况:")
print(value_churn.round(2))

# 8. 构建预测模型
print("\n8. 构建预测模型")
print("-" * 40)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

    # 准备数据
    X = df_clean.drop('Exited', axis=1)
    y = df_clean['Exited']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 训练随机森林模型
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    # 预测
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    # 模型评估
    print("\n模型评估结果:")
    print(f"准确率: {rf_model.score(X_test, y_test):.4f}")
    print(f"AUC分数: {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['未流失', '流失']))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)

    # 特征重要性
    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': rf_model.feature_importances_
    }).sort_values('重要性', ascending=False)

    print("\n特征重要性排名:")
    print(feature_importance.head(10).round(4))

    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='重要性', y='特征', palette='viridis')
    plt.title('Top 10 特征重要性')
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.tight_layout()
    plt.savefig('特征重要性.png', dpi=300)
    plt.show()
    print("✅ 特征重要性图已保存为 '特征重要性.png'")

    # ROC曲线
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'随机森林 (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('ROC曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('roc曲线.png', dpi=300)
    plt.show()
    print("✅ ROC曲线已保存为 'roc曲线.png'")

except ImportError as e:
    print(f"请安装scikit-learn: pip install scikit-learn")
    print(f"错误信息: {e}")

# 9. 生成分析报告
print("\n9. 生成分析报告")
print("-" * 40)

# 核心发现
print("\n【项目核心发现】")
print("=" * 60)
print(f"1. 客户总体流失率: {df['Exited'].mean() * 100:.2f}%")
print("\n2. 最易流失客户特征:")
print(
    f"   - 平均年龄: {df[df['Exited'] == 1]['Age'].mean():.1f}岁 (非流失客户: {df[df['Exited'] == 0]['Age'].mean():.1f}岁)")
print(
    f"   - 平均余额: {df[df['Exited'] == 1]['Balance'].mean():.2f}元 (非流失客户: {df[df['Exited'] == 0]['Balance'].mean():.2f}元)")
print(
    f"   - 平均信用评分: {df[df['Exited'] == 1]['CreditScore'].mean():.1f} (非流失客户: {df[df['Exited'] == 0]['CreditScore'].mean():.1f})")

print("\n3. 最重要的流失预测因素:")
if 'feature_importance' in locals():
    for i, row in feature_importance.head(5).iterrows():
        print(f"   - {row['特征']}: {row['重要性']:.3f}")

print("\n4. 业务建议:")
print("   - 重点关注45岁以上的中年客户群体")
print("   - 对高余额客户（>10万）加强维护和关怀")
print("   - 提升会员活跃度可显著降低流失率（活跃会员流失率更低）")
print("   - 德国地区的客户流失率较高，需要针对性策略")
print("   - 女性客户的流失率略高，可考虑个性化服务")

# 10. 导出结果
print("\n10. 导出结果")
print("-" * 40)

# 保存清洗后的数据
df_clean.to_csv('银行客户数据_清洗后.csv', index=False, encoding='utf-8-sig')
print("✅ 清洗后数据已保存为 '银行客户数据_清洗后.csv'")

# 保存分析报告到Excel
try:
    with pd.ExcelWriter('银行客户流失分析报告.xlsx') as writer:
        # 原始数据摘要
        df.describe().to_excel(writer, sheet_name='数据摘要')

        # 流失分析
        pd.DataFrame({
            '客户类型': ['未流失', '流失'],
            '数量': [len(df[df['Exited'] == 0]), len(df[df['Exited'] == 1])],
            '占比': [1 - df['Exited'].mean(), df['Exited'].mean()]
        }).to_excel(writer, sheet_name='流失概况', index=False)

        # 特征重要性
        if 'feature_importance' in locals():
            feature_importance.to_excel(writer, sheet_name='特征重要性', index=False)

        # 各层级流失率
        value_churn.to_excel(writer, sheet_name='余额层级分析')

        # 地区分析
        geo_analysis = pd.crosstab(df['Geography'], df['Exited'], margins=True, margins_name='总计')
        geo_analysis.columns = ['未流失', '流失', '总计']
        geo_analysis.to_excel(writer, sheet_name='地区分析')

    print("✅ 分析报告已保存为 '银行客户流失分析报告.xlsx'")
except Exception as e:
    print(f"保存Excel失败: {e}")

print("\n" + "=" * 80)
print("✅ 项目分析完成！")
print("=" * 80)
print("\n生成的文件:")
print("1. 银行客户流失分析.png - 主要可视化图表")
print("2. 相关性热力图.png - 特征相关性分析")
print("3. 特征重要性.png - 模型特征重要性")
print("4. roc曲线.png - 模型性能评估")
print("5. 银行客户数据_清洗后.csv - 清洗后的数据")
print("6. 银行客户流失分析报告.xlsx - 完整分析报告")