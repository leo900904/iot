import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# 讀取資料
data = pd.read_csv('2330-training.csv')

# 資料理解
st.write("## Data Overview")
st.write(data.describe())

# 資料準備
X = data.drop(columns=['y', 'Date'])  # 排除目標變數和日期
y = data['y']

# 特徵選擇
selected_features = st.multiselect("Select features for the model", options=X.columns.tolist(), default=X.columns.tolist())
X_selected = X[selected_features]

# 將資料分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 建立回歸模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 預測
y_pred = model.predict(X_test_scaled)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 顯示結果
st.write("## Model Evaluation")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"R-squared (R²): {r2}")

# 分析特徵重要性
importance = model.coef_
feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

st.write("## Feature Importance")
st.bar_chart(feature_importance.set_index('Feature'))

# 預測結果可視化
st.write("## Predicted vs Actual")
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
st.line_chart(comparison_df)

# 實時更新誤差分析
st.write("## Error Analysis")
error = y_test - y_pred
error_df = pd.DataFrame({'Error': error})
st.line_chart(error_df)
