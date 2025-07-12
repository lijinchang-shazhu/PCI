import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SMOTENC
from imblearn.under_sampling import TomekLinks, NearMiss, ClusterCentroids
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedBaggingClassifier

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import os
# 确保导入SMOTE

from collections import Counter

# 加载数据集
train_path = '../ori_data/train.csv'
train = pd.read_csv(train_path)

# 显示训练数据集的前几行以了解其结构
print(train.head())

# 数据预处理
# 1. 将分类特征转换为数值特征
train_df = train.copy()
train_df['Gender'] = train_df['Gender'].map({'Male': 1, 'Female': 0})
train_df['Vehicle_Damage'] = train_df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

# 2. 对多类别特征('Vehicle_Age')进行独热编码
one_hot_encoder = OneHotEncoder(drop='first')
vehicle_age_encoded = one_hot_encoder.fit_transform(train_df[['Vehicle_Age']]).toarray()
vehicle_age_encoded_df = pd.DataFrame(vehicle_age_encoded, columns=one_hot_encoder.get_feature_names_out(['Vehicle_Age']))
train_df = pd.concat([train_df.drop('Vehicle_Age', axis=1), vehicle_age_encoded_df], axis=1)

# 3. 标准化数值特征
scaler = StandardScaler()
numerical_features = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['Response'], axis=1), train_df['Response'], test_size=0.2, random_state=42)
# 将X_train和y_train转换为DataFrame，以便保存为CSV
X_train_df = pd.DataFrame(X_train)
y_train_df = pd.DataFrame(y_train, columns=['Response'])

def save_resampled_data(method_name, X_res, y_res):
    resampled_df_path = os.path.join('/mnt/data0/LJC/LassoNet/predData', f'{method_name}_train.csv')
    # 确保目录存在，不存在时创建
    os.makedirs(os.path.dirname(resampled_df_path), exist_ok=True)
    print(f"保存地址：{resampled_df_path}")
        
    resampled_df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res, columns=['Response'])], axis=1)
    resampled_df.to_csv(resampled_df_path, index=False)
    print(f"{method_name}处理后的训练数据集已保存至: {resampled_df_path}")

# 1. ADASYN
adasyn = ADASYN(sampling_strategy='minority', random_state=42)
X_res_adasyn, y_res_adasyn = adasyn.fit_resample(X_train, y_train)
save_resampled_data('adasyn', X_res_adasyn, y_res_adasyn)

# 2. SMOTE + Tomek Links
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_res_smote, y_res_smote = smote.fit_resample(X_train, y_train)
tomek = TomekLinks()
X_res_tomek, y_res_tomek = tomek.fit_resample(X_res_smote, y_res_smote)
save_resampled_data('smote_tomek', X_res_tomek, y_res_tomek)

# 3. NearMiss
nearmiss = NearMiss()
X_res_nearmiss, y_res_nearmiss = nearmiss.fit_resample(X_train, y_train)
save_resampled_data('nearmiss', X_res_nearmiss, y_res_nearmiss)

# 4. SMOTE-ENN
smote_enn = SMOTEENN(random_state=42)
X_res_smoteenn, y_res_smoteenn = smote_enn.fit_resample(X_train, y_train)
save_resampled_data('smoteenn', X_res_smoteenn, y_res_smoteenn)

# 5. ClusterCentroids
cluster_centroids = ClusterCentroids(random_state=42)
X_res_centroids, y_res_centroids = cluster_centroids.fit_resample(X_train, y_train)
save_resampled_data('cluster_centroids', X_res_centroids, y_res_centroids)

# 6. Borderline-SMOTE
borderline_smote = BorderlineSMOTE(random_state=42)
X_res_borderline, y_res_borderline = borderline_smote.fit_resample(X_train, y_train)
save_resampled_data('borderline_smote', X_res_borderline, y_res_borderline)

# 7. KMeans-SMOTE
# 尝试降低 cluster_balance_threshold 或增加簇数量
kmeans_smote = KMeansSMOTE(random_state=42, cluster_balance_threshold=0.01, k_neighbors=10)
# 执行采样
X_res_kmeans, y_res_kmeans = kmeans_smote.fit_resample(X_train, y_train)
# 保存结果
save_resampled_data('kmeans_smote', X_res_kmeans, y_res_kmeans)

# 8. Random Oversampling (SMOTE here as a placeholder)
random_oversample = SMOTE(random_state=42)  # SMOTE works as Random Oversampler with default settings
X_res_random, y_res_random = random_oversample.fit_resample(X_train, y_train)
save_resampled_data('random_oversampling', X_res_random, y_res_random)

# 9. BalancedBaggingClassifier (与随机森林结合)
# 使用 estimator 参数替换 base_estimator
bbc = BalancedBaggingClassifier(estimator=RandomForestClassifier(), random_state=42)

# 继续执行后续操作，例如训练模型
bbc.fit(X_train, y_train)
y_pred_bbc = bbc.predict(X_test)

# 输出或保存预测结果
print("Balanced Bagging Classifier predictions:", y_pred_bbc)
# BalancedBaggingClassifier 是一个集成方法，不输出resampled的X_train和y_train，故此略过保存步骤

# 10. SMOTENC (适用于类别特征)
smotenc = SMOTENC(categorical_features=[0, 1], random_state=42)  # 将categorical_features替换为你的类别特征索引
X_res_smotenc, y_res_smotenc = smotenc.fit_resample(X_train, y_train)
save_resampled_data('smotenc', X_res_smotenc, y_res_smotenc)