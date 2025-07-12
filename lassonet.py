

# 导入成功安装的库
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from lassonet import LassoNetRegressor
import xgboost as xgb

# 加载数据集
train_df = pd.read_csv('文件名')

# 查看数据集的列名
print("数据集的列名：")
print(train_df.columns)

# 检查前几行数据
print("数据集前几行预览：")
print(train_df.head())

# 假设目标变量的真实列名是 'Response'，根据列名进行调整
X = train_df.drop(columns=['Response'])  # 根据实际列名调整
y = train_df['Response']  # 根据实际列名调整

# 将数据集分为训练集和测试集（80-20分割）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对特征进行标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化LassoNet模型进行特征选择
lassonet = LassoNetRegressor(hidden_dims=(64,), lambda_start=1e-3, verbose=True)

# 训练LassoNet模型
lassonet.fit(X_train_scaled, y_train)

# 获取被选择的特征
selected_features = X.columns[lassonet.feature_importances_ > 0]
print("LassoNet选择的特征:")
print(selected_features)

# 选择的特征数据
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 决策树
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_selected, y_train)
dt_pred = dt_model.predict(X_test_selected)
print("\n决策树模型报告:")
print(classification_report(y_test, dt_pred))

# 支持向量机（参数调优）
svm_model = SVC(random_state=42)
svm_param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svm_grid = GridSearchCV(svm_model, svm_param_grid, cv=5)
svm_grid.fit(X_train_selected, y_train)
svm_pred = svm_grid.predict(X_test_selected)
print("\n支持向量机模型报告:")
print(classification_report(y_test, svm_pred))

# K近邻（参数调优）
knn_model = KNeighborsClassifier()
knn_param_grid = {'n_neighbors': [3, 5, 7, 9]}
knn_grid = GridSearchCV(knn_model, knn_param_grid, cv=5)
knn_grid.fit(X_train_selected, y_train)
knn_pred = knn_grid.predict(X_test_selected)
print("\nK近邻模型报告:")
print(classification_report(y_test, knn_pred))

# 随机森林（增强模型）
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_selected, y_train)
rf_pred = rf_model.predict(X_test_selected)
print("\n随机森林模型报告:")
print(classification_report(y_test, rf_pred))

# XGBoost（使用最佳参数）
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train_selected, y_train)
xgb_pred = xgb_model.predict(X_test_selected)
print("\nXGBoost模型报告:")
print(classification_report(y_test, xgb_pred))

# 计算每个模型的准确率
print(f"决策树准确率: {accuracy_score(y_test, dt_pred):.4f}")
print(f"支持向量机准确率: {accuracy_score(y_test, svm_pred):.4f}")
print(f"K近邻准确率: {accuracy_score(y_test, knn_pred):.4f}")
print(f"随机森林准确率: {accuracy_score(y_test, rf_pred):.4f}")
print(f"XGBoost准确率: {accuracy_score(y_test, xgb_pred):.4f}")
