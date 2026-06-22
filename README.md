# 肠道准备不充分风险预测模型 Streamlit 中文版 V2

本项目用于部署肠道准备不充分风险预测模型。最终模型为 Raw Stacking Classifier。

## 本版修改内容

1. 删除网页左侧“模型设置”模块；
2. 默认显示 SHAP 蜂巢图，抽样样本量固定为 100；
3. 反事实建议中默认排除“禁食 1 天及以上”策略；
4. 患者变量输入重新分为：
   - 基本信息：年龄、性别、BMI
   - 临床相关因素：患者状态、既往肠镜检查史、慢性便秘、平时大便性状、糖尿病、既往盆腔手术史
   - 肠道准备相关因素：饮食限制方式、饮食限制天数、泻药方案、泻药是否分次服用、肠道准备宣教方式、服用泻药后是否加强活动、肠道准备至肠镜检查时间间隔
5. 删除模型编码输入和临床输入汇总查看框；
6. 简化风险下降建议表格；
7. 删除 CSV 下载按钮；
8. 单项干预建议和组合干预建议默认展开，并简化显示列。

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

## 注意事项

本工具仅用于科研和临床决策辅助，不替代医生判断或本单位肠道准备规范。
如果部署包中包含训练集、测试集或验证集患者数据，请使用私有仓库部署。
