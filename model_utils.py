# ============================================================
# Model and Input Utilities 中文版
# ============================================================

from pathlib import Path
import cloudpickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ============================================================
# 中文字体设置
# ============================================================

def setup_chinese_font():
    """
    设置 matplotlib 中文字体。
    Streamlit Cloud 可配合 packages.txt 安装 fonts-noto-cjk。
    """
    candidate_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "WenQuanYi Zen Hei",
        "DejaVu Sans"
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font in candidate_fonts:
        if font in available_fonts:
            plt.rcParams["font.sans-serif"] = [font]
            break

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"


# ============================================================
# 读取部署包
# ============================================================

@st.cache_resource(show_spinner=False)
def load_deploy_pack(deploy_dir: str):
    """
    读取模型部署包。
    兼容两种结构：
    1. Final_Deploy_Stacking_V2/Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl
    2. 仓库根目录/Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl
    """
    deploy_dir = Path(deploy_dir)

    candidate_pack_paths = [
        deploy_dir / "Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl",
        Path("Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl"),
        Path("Final_Deploy_Stacking_V2") / "Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl",
    ]

    pack_path = None

    for path in candidate_pack_paths:
        if path.exists():
            pack_path = path
            break

    if pack_path is None:
        raise FileNotFoundError(
            "未找到 Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl。"
            "请确认模型部署包位于 Final_Deploy_Stacking_V2 文件夹或仓库根目录。"
        )

    with open(pack_path, "rb") as f:
        deploy_pack = cloudpickle.load(f)

    if deploy_pack.get("scaler", None) is None:
        candidate_scaler_paths = [
            deploy_dir / "standard_scaler_v2.pkl",
            Path("standard_scaler_v2.pkl"),
            Path("Final_Deploy_Stacking_V2") / "standard_scaler_v2.pkl",
        ]

        for scaler_path in candidate_scaler_paths:
            if scaler_path.exists():
                deploy_pack["scaler"] = joblib.load(scaler_path)
                break

    return deploy_pack


# ============================================================
# 中文变量名
# ============================================================

def get_feature_display_map():
    return {
        "Age": "年龄",
        "BMI": "体质指数",
        "DietaryRestrictionDays": "饮食限制时间（天）",

        "HospitalGrade": "医院等级",
        "Sex": "性别",
        "InpatientStatus": "患者状态",
        "PreviousColonoscopy": "既往肠镜检查史",
        "ChronicConstipation": "慢性便秘",
        "ChronicDiarrhea": "慢性腹泻",
        "DiabetesMellitus": "糖尿病",
        "StoolForm": "平时大便性状",
        "BPEducationModality": "肠道准备宣教方式",
        "SplitDose_BP": "泻药是否分次服用",
        "PreColonoscopyPhysicalActivity": "服用泻药后是否加强活动",

        "BPtoColonoscopyinterval_1": "肠道准备至肠镜间隔：<120 min",
        "BPtoColonoscopyinterval_2": "肠道准备至肠镜间隔：120–240 min",
        "BPtoColonoscopyinterval_3": "肠道准备至肠镜间隔：240–360 min",
        "BPtoColonoscopyinterval_4": "肠道准备至肠镜间隔：≥360 min",

        "DietaryRestriction_1": "饮食限制方式：禁食",
        "DietaryRestriction_2": "饮食限制方式：低渣饮食",
        "DietaryRestriction_3": "饮食限制方式：流质饮食",
        "DietaryRestriction_4": "饮食限制方式：普通饮食",

        "LaxativeRegimen_1": "泻药方案：PEG 2L",
        "LaxativeRegimen_2": "泻药方案：PEG 3L",
        "LaxativeRegimen_3": "泻药方案：PEG 4L",
        "LaxativeRegimen_4": "泻药方案：磷酸钠盐",
        "LaxativeRegimen_5": "泻药方案：甘露醇",
        "LaxativeRegimen_6": "泻药方案：硫酸镁",

        "PsychotropicMedication_2": "精神类药物：三环类抗抑郁药",
        "PreviousAbdominopelvicSurgery_1": "既往盆腔手术史"
    }


# ============================================================
# 连续变量标准化 / 反标准化
# ============================================================

def get_scaler_feature_order(scaler):
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return ["Age", "BMI", "DietaryRestrictionDays"]


def standardize_value(scaler, feature_name, original_value):
    scaler_feature_order = get_scaler_feature_order(scaler)

    if feature_name not in scaler_feature_order:
        return original_value

    idx = scaler_feature_order.index(feature_name)

    return (float(original_value) - scaler.mean_[idx]) / scaler.scale_[idx]


def inverse_standardized_value(scaler, feature_name, standardized_value):
    scaler_feature_order = get_scaler_feature_order(scaler)

    if feature_name not in scaler_feature_order:
        return standardized_value

    idx = scaler_feature_order.index(feature_name)

    return float(standardized_value) * scaler.scale_[idx] + scaler.mean_[idx]


# ============================================================
# 通用工具
# ============================================================

def set_onehot(row, group, selected_col):
    for col in group:
        if col in row:
            row[col] = 0
    if selected_col in row:
        row[selected_col] = 1
    return row


def format_probability(prob):
    if pd.isna(prob):
        return "不适用"
    return f"{float(prob) * 100:.1f}%"


def classify_binary_risk(prob, threshold=0.135):
    return "高风险" if prob >= threshold else "低风险"


def predict_patient(model, patient_model_df, feature_order):
    patient_model_df = patient_model_df[feature_order]
    return float(model.predict_proba(patient_model_df)[:, 1][0])


# ============================================================
# Streamlit 输入表单
# ============================================================


def build_patient_input_form(feature_order, scaler, feature_name_map):
    """
    构建患者变量输入表单。
    仅显示用户指定的变量；未展示但模型需要的变量默认设为 0。
    """

    row = {col: 0 for col in feature_order}
    raw_values = {}
    summary_rows = []
    used_features = set()

    dietary_group = [c for c in [
        "DietaryRestriction_1", "DietaryRestriction_2",
        "DietaryRestriction_3", "DietaryRestriction_4"
    ] if c in feature_order]

    laxative_group = [c for c in [
        "LaxativeRegimen_1", "LaxativeRegimen_2", "LaxativeRegimen_3",
        "LaxativeRegimen_4", "LaxativeRegimen_5", "LaxativeRegimen_6"
    ] if c in feature_order]

    interval_group = [c for c in [
        "BPtoColonoscopyinterval_1", "BPtoColonoscopyinterval_2",
        "BPtoColonoscopyinterval_3", "BPtoColonoscopyinterval_4"
    ] if c in feature_order]

    # ------------------------------------------------------------
    # 1. 基本信息：年龄、性别、BMI
    # ------------------------------------------------------------
    st.subheader("基本信息")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("年龄（岁）", min_value=18.0, max_value=100.0, value=60.0, step=1.0)

    with c2:
        sex_options = {"男": 1, "女": 0}
        selected_sex = st.selectbox("性别", list(sex_options.keys()))

    with c3:
        bmi = st.number_input("BMI（kg/m²）", min_value=10.0, max_value=50.0, value=23.0, step=0.1)

    if "Age" in row:
        row["Age"] = standardize_value(scaler, "Age", age)
        used_features.add("Age")
    if "Sex" in row:
        row["Sex"] = sex_options[selected_sex]
        used_features.add("Sex")
    if "BMI" in row:
        row["BMI"] = standardize_value(scaler, "BMI", bmi)
        used_features.add("BMI")

    raw_values.update({"Age": age, "Sex": sex_options[selected_sex], "BMI": bmi})
    summary_rows.extend([
        {"变量": "年龄", "取值": age},
        {"变量": "性别", "取值": selected_sex},
        {"变量": "BMI", "取值": bmi},
    ])

    # ------------------------------------------------------------
    # 2. 临床相关因素
    # ------------------------------------------------------------
    st.subheader("临床相关因素")

    clinical_configs = {
        "InpatientStatus": {"label": "患者状态", "options": {"门诊": 0, "住院": 1}},
        "PreviousColonoscopy": {"label": "既往肠镜检查史", "options": {"无": 0, "有": 1}},
        "ChronicConstipation": {"label": "慢性便秘", "options": {"否": 0, "是": 1}},
        "StoolForm": {"label": "平时大便性状", "options": {"布里斯托 3–7 分": 0, "布里斯托 1–2 分 / 硬便": 1}},
        "DiabetesMellitus": {"label": "糖尿病", "options": {"否": 0, "是": 1}},
        "PreviousAbdominopelvicSurgery_1": {"label": "既往盆腔手术史", "options": {"无": 0, "有": 1}},
    }

    clinical_cols = st.columns(3)
    shown_idx = 0

    for feature, cfg in clinical_configs.items():
        if feature not in feature_order:
            continue

        with clinical_cols[shown_idx % 3]:
            selected = st.selectbox(
                cfg["label"],
                options=list(cfg["options"].keys()),
                index=0,
                key=f"input_{feature}"
            )

        value = cfg["options"][selected]
        row[feature] = value
        raw_values[feature] = value
        summary_rows.append({"变量": cfg["label"], "取值": selected})
        used_features.add(feature)
        shown_idx += 1

    # ------------------------------------------------------------
    # 3. 肠道准备相关因素
    # ------------------------------------------------------------
    st.subheader("肠道准备相关因素")

    bowel_cols1 = st.columns(3)
    bowel_cols2 = st.columns(4)

    dietary_options = {
        "禁食": "DietaryRestriction_1",
        "低渣饮食": "DietaryRestriction_2",
        "流质饮食": "DietaryRestriction_3",
        "普通饮食": "DietaryRestriction_4",
    }

    laxative_options = {
        "PEG 2L": "LaxativeRegimen_1",
        "PEG 3L": "LaxativeRegimen_2",
        "PEG 4L": "LaxativeRegimen_3",
        "磷酸钠盐": "LaxativeRegimen_4",
        "甘露醇": "LaxativeRegimen_5",
        "硫酸镁": "LaxativeRegimen_6",
    }

    interval_options = {
        "<120 min": "BPtoColonoscopyinterval_1",
        "120–240 min": "BPtoColonoscopyinterval_2",
        "240–360 min": "BPtoColonoscopyinterval_3",
        "≥360 min": "BPtoColonoscopyinterval_4",
    }

    # 饮食限制方式
    if dietary_group:
        valid_diet_options = {k: v for k, v in dietary_options.items() if v in dietary_group}
        with bowel_cols1[0]:
            selected_diet = st.selectbox("饮食限制方式", list(valid_diet_options.keys()))
        row = set_onehot(row, dietary_group, valid_diet_options[selected_diet])
        summary_rows.append({"变量": "饮食限制方式", "取值": selected_diet})
        used_features.update(dietary_group)

    # 饮食限制天数
    with bowel_cols1[1]:
        diet_days = st.number_input("饮食限制天数", min_value=0.0, max_value=7.0, value=1.0, step=1.0)

    if "DietaryRestrictionDays" in row:
        row["DietaryRestrictionDays"] = standardize_value(scaler, "DietaryRestrictionDays", diet_days)
        raw_values["DietaryRestrictionDays"] = diet_days
        summary_rows.append({"变量": "饮食限制天数", "取值": f"{diet_days} 天"})
        used_features.add("DietaryRestrictionDays")

    # 泻药方案
    if laxative_group:
        valid_lax_options = {k: v for k, v in laxative_options.items() if v in laxative_group}
        with bowel_cols1[2]:
            selected_lax = st.selectbox("泻药方案", list(valid_lax_options.keys()))
        row = set_onehot(row, laxative_group, valid_lax_options[selected_lax])
        summary_rows.append({"变量": "泻药方案", "取值": selected_lax})
        used_features.update(laxative_group)

    # 泻药是否分次服用
    if "SplitDose_BP" in feature_order:
        with bowel_cols2[0]:
            split_options = {"否": 0, "是": 1}
            selected_split = st.selectbox("泻药是否分次服用", list(split_options.keys()))
        row["SplitDose_BP"] = split_options[selected_split]
        raw_values["SplitDose_BP"] = split_options[selected_split]
        summary_rows.append({"变量": "泻药是否分次服用", "取值": selected_split})
        used_features.add("SplitDose_BP")

    # 肠道准备宣教方式
    if "BPEducationModality" in feature_order:
        with bowel_cols2[1]:
            edu_options = {"文字 + 图文或影像宣教": 0, "口头或文字宣教": 1}
            selected_edu = st.selectbox("肠道准备宣教方式", list(edu_options.keys()))
        row["BPEducationModality"] = edu_options[selected_edu]
        raw_values["BPEducationModality"] = edu_options[selected_edu]
        summary_rows.append({"变量": "肠道准备宣教方式", "取值": selected_edu})
        used_features.add("BPEducationModality")

    # 服用泻药后是否加强活动
    if "PreColonoscopyPhysicalActivity" in feature_order:
        with bowel_cols2[2]:
            activity_options = {"否": 0, "是": 1}
            selected_activity = st.selectbox("服用泻药后是否加强活动", list(activity_options.keys()))
        row["PreColonoscopyPhysicalActivity"] = activity_options[selected_activity]
        raw_values["PreColonoscopyPhysicalActivity"] = activity_options[selected_activity]
        summary_rows.append({"变量": "服用泻药后是否加强活动", "取值": selected_activity})
        used_features.add("PreColonoscopyPhysicalActivity")

    # 肠道准备至肠镜检查时间间隔
    if interval_group:
        valid_interval_options = {k: v for k, v in interval_options.items() if v in interval_group}
        with bowel_cols2[3]:
            selected_interval = st.selectbox("肠道准备至肠镜检查时间间隔", list(valid_interval_options.keys()))
        row = set_onehot(row, interval_group, valid_interval_options[selected_interval])
        summary_rows.append({"变量": "肠道准备至肠镜检查时间间隔", "取值": selected_interval})
        used_features.update(interval_group)

    # ------------------------------------------------------------
    # 未展示但模型需要的变量：默认 0
    # ------------------------------------------------------------
    for col in feature_order:
        if col not in used_features:
            row[col] = row.get(col, 0)

    patient_model_df = pd.DataFrame([row])[feature_order]
    patient_raw_df = pd.DataFrame([raw_values])
    input_summary_df = pd.DataFrame(summary_rows)

    return patient_raw_df, patient_model_df, input_summary_df
