# 客戶交易行為二元分類預測：Logistic Regression vs MLP

個人完成的機器學習期中專案，針對匿名化的客戶交易資料進行二元分類預測（Kaggle 風格的競賽題型，類似 Santander Customer Transaction Prediction）。

## 資料集

- `train.csv` / `test.csv`：每筆資料包含 `ID_code`（識別碼）、`target`（1 表示有交易、0 表示沒有交易）與 `var_0` ~ `var_199` 共 200 個匿名數值特徵
- 訓練集共 200,000 筆資料，正負樣本比例不平衡（約 1:9）

> 註：原始資料檔為課程提供的競賽資料，未包含在此 repo 中。

## 分析流程

1. **資料前處理**：分離特徵與標籤，以 8:2 分層抽樣（stratify）切分訓練/驗證集，並用 `StandardScaler` 標準化（只用訓練集 fit，再套用到驗證集與測試集）
2. **Baseline 模型：Logistic Regression** — Validation AUC 0.8599
3. **進階模型：MLP 神經網路**
   - 兩層隱藏層（128 → 64），搭配 ReLU、L2 正則化與 Dropout 防止過擬合
   - 使用 `EarlyStopping`（監控 val_auc）避免過度訓練
   - 繪製訓練過程的 AUC / Loss 曲線
   - Validation AUC 0.8578
4. **產出預測結果**：對測試集預測交易機率並輸出 `submission.csv`（Kaggle 格式：`ID_code` + `target` 機率值）

## 結果

| 模型 | Validation AUC |
|---|---|
| Logistic Regression | 0.8599 |
| MLP (Dropout + L2 + Early Stopping) | 0.8578 |

在此資料集上，簡單的 Logistic Regression baseline 表現與調校過的 MLP 相當接近，顯示這份資料的特徵可能已經相對線性可分。

## 使用工具

pandas、numpy、matplotlib、scikit-learn（Logistic Regression、StandardScaler、train_test_split、roc_auc_score）、TensorFlow / Keras（MLP、EarlyStopping）

## 檔案

- `transaction_prediction_lr_vs_mlp.ipynb`：完整分析程式碼

## 作者

鍾嬡 (Audrey)

