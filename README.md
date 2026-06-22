# 肠道准备不充分风险预测模型 Streamlit 中文版

本项目用于部署肠道准备不充分风险预测模型。最终模型为 Raw Stacking Classifier。

## 文件结构

```text
.
├── app.py
├── model_utils.py
├── shap_utils.py
├── counterfactual_utils.py
├── requirements.txt
├── runtime.txt
├── packages.txt
├── README.md
└── Final_Deploy_Stacking_V2/
    ├── Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl
    ├── standard_scaler_v2.pkl
    ├── deploy_info_v2.json
    └── feature_order_v2.csv
```

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 主要功能

1. 输入患者临床变量
2. 输出肠道准备不充分预测概率
3. 按 0.135 阈值输出高风险 / 低风险
4. 输出患者个体 SHAP 瀑布图
5. 输出全局 SHAP 蜂巢图
6. 输出可降低预测风险的反事实干预建议

## 注意事项

本工具仅用于科研和临床决策辅助，不替代医生判断或本单位肠道准备规范。
如果部署包中包含训练集、测试集或验证集患者数据，请使用私有仓库部署。
