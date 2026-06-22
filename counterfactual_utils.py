# ============================================================
# Counterfactual / Intervention Scenario Utilities 中文版
# 可干预措施确定性扫描
# ============================================================

import itertools
import numpy as np
import pandas as pd

from model_utils import (
    format_probability,
    standardize_value,
    inverse_standardized_value,
)


DIET_DAYS_ORIGINAL_MIN = 0
DIET_DAYS_ORIGINAL_MAX = 3


dietary_label_map = {
    "DietaryRestriction_1": "禁食",
    "DietaryRestriction_2": "低渣饮食",
    "DietaryRestriction_3": "流质饮食",
    "DietaryRestriction_4": "普通饮食"
}

laxative_label_map = {
    "LaxativeRegimen_1": "PEG 2L",
    "LaxativeRegimen_2": "PEG 3L",
    "LaxativeRegimen_3": "PEG 4L",
    "LaxativeRegimen_4": "磷酸钠盐",
    "LaxativeRegimen_5": "甘露醇",
    "LaxativeRegimen_6": "硫酸镁"
}

interval_label_map = {
    "BPtoColonoscopyinterval_1": "<120 min",
    "BPtoColonoscopyinterval_2": "120–240 min",
    "BPtoColonoscopyinterval_3": "240–360 min",
    "BPtoColonoscopyinterval_4": "≥360 min"
}

binary_label_map = {
    "BPEducationModality": {
        0: "文字 + 图文或影像宣教",
        1: "口头或文字宣教"
    },
    "SplitDose_BP": {
        0: "未分次服药",
        1: "分次服药"
    },
    "PreColonoscopyPhysicalActivity": {
        0: "未进行肠镜前体力活动",
        1: "进行肠镜前体力活动"
    }
}


def predict_single_risk(model, row, feature_order):
    row_df = pd.DataFrame([row])[feature_order]
    prob = model.predict_proba(row_df)[0, 1]
    return float(prob)


def set_onehot_group(row, group, target_col):
    row = row.copy()

    for col in group:
        if col in row.index:
            row[col] = 0

    if target_col in row.index:
        row[target_col] = 1

    return row


def decode_onehot_group(row, group, label_map):
    group = [col for col in group if col in row.index]

    if len(group) == 0:
        return None

    active_cols = [
        col for col in group
        if int(round(float(row[col]))) == 1
    ]

    if len(active_cols) == 0:
        return "无"

    return label_map.get(active_cols[0], active_cols[0])


def get_binary_label(feature, value):
    value = int(round(float(value)))

    if feature in binary_label_map:
        return binary_label_map[feature].get(value, value)

    return value


def set_scalar_value(row, col, value):
    row = row.copy()
    row[col] = value
    return row


def get_diet_days(row, scaler):
    if "DietaryRestrictionDays" not in row.index:
        return None

    return round(
        float(
            inverse_standardized_value(
                scaler,
                "DietaryRestrictionDays",
                row["DietaryRestrictionDays"]
            )
        ),
        1
    )


def set_diet_days(row, scaler, target_days):
    row = row.copy()

    if "DietaryRestrictionDays" in row.index:
        row["DietaryRestrictionDays"] = standardize_value(
            scaler,
            "DietaryRestrictionDays",
            target_days
        )

    return row


def classify_suggestion(prob, intervention_threshold):
    if prob < intervention_threshold:
        return "预测风险已降至干预阈值以下"
    return "预测风险下降，但仍高于干预阈值"


def postprocess_row(row, feature_order):
    row = row.copy()
    return row[feature_order]


def generate_single_intervention_candidates(
    original_row,
    feature_order,
    scaler,
    feature_name_map,
    exclude_fasting=False
):
    actions = []

    dietary_group = [c for c in [
        "DietaryRestriction_1",
        "DietaryRestriction_2",
        "DietaryRestriction_3",
        "DietaryRestriction_4"
    ] if c in feature_order]

    laxative_group = [c for c in [
        "LaxativeRegimen_1",
        "LaxativeRegimen_2",
        "LaxativeRegimen_3",
        "LaxativeRegimen_4",
        "LaxativeRegimen_5",
        "LaxativeRegimen_6"
    ] if c in feature_order]

    interval_group = [c for c in [
        "BPtoColonoscopyinterval_1",
        "BPtoColonoscopyinterval_2",
        "BPtoColonoscopyinterval_3",
        "BPtoColonoscopyinterval_4"
    ] if c in feature_order]

    # 饮食限制天数
    if "DietaryRestrictionDays" in feature_order:
        original_days = get_diet_days(original_row, scaler)

        for target_days in range(DIET_DAYS_ORIGINAL_MIN, DIET_DAYS_ORIGINAL_MAX + 1):
            if original_days is None:
                continue

            if abs(float(target_days) - float(original_days)) < 1e-6:
                continue

            direction = "增加" if target_days > original_days else "减少"

            actions.append({
                "干预类别": "饮食限制时间",
                "干预措施": f"{direction}饮食限制时间至 {target_days} 天",
                "原始取值": f"{original_days} 天",
                "干预后取值": f"{target_days} 天",
                "改变变量": "DietaryRestrictionDays",
                "apply": lambda row, d=target_days: set_diet_days(row, scaler, d)
            })

    # 二元可干预变量
    for col in [
        "BPEducationModality",
        "SplitDose_BP",
        "PreColonoscopyPhysicalActivity"
    ]:
        if col not in feature_order:
            continue

        old_value = int(round(float(original_row[col])))

        for new_value in [0, 1]:
            if new_value == old_value:
                continue

            old_label = get_binary_label(col, old_value)
            new_label = get_binary_label(col, new_value)

            actions.append({
                "干预类别": feature_name_map.get(col, col),
                "干预措施": f"{feature_name_map.get(col, col)}：{old_label} → {new_label}",
                "原始取值": old_label,
                "干预后取值": new_label,
                "改变变量": col,
                "apply": lambda row, c=col, v=new_value: set_scalar_value(row, c, v)
            })

    # 饮食策略
    if dietary_group:
        old_diet = decode_onehot_group(original_row, dietary_group, dietary_label_map)

        for target_col in dietary_group:
            target_label = dietary_label_map.get(target_col, target_col)

            if exclude_fasting and target_label == "禁食":
                continue

            if target_label == old_diet:
                continue

            actions.append({
                "干预类别": "饮食限制策略",
                "干预措施": f"饮食策略：{old_diet} → {target_label}",
                "原始取值": old_diet,
                "干预后取值": target_label,
                "改变变量": ",".join(dietary_group),
                "apply": lambda row, g=dietary_group, t=target_col: set_onehot_group(row, g, t)
            })

    # 泻药方案
    if laxative_group:
        old_laxative = decode_onehot_group(original_row, laxative_group, laxative_label_map)

        for target_col in laxative_group:
            target_label = laxative_label_map.get(target_col, target_col)

            if target_label == old_laxative:
                continue

            if old_laxative == "PEG 2L" and target_label in ["PEG 3L", "PEG 4L"]:
                category = "泻药方案 / 剂量增加"
                intervention_text = f"增加泻药剂量：{old_laxative} → {target_label}"
            elif old_laxative == "PEG 3L" and target_label == "PEG 4L":
                category = "泻药方案 / 剂量增加"
                intervention_text = f"增加泻药剂量：{old_laxative} → {target_label}"
            else:
                category = "泻药方案"
                intervention_text = f"泻药方案：{old_laxative} → {target_label}"

            actions.append({
                "干预类别": category,
                "干预措施": intervention_text,
                "原始取值": old_laxative,
                "干预后取值": target_label,
                "改变变量": ",".join(laxative_group),
                "apply": lambda row, g=laxative_group, t=target_col: set_onehot_group(row, g, t)
            })

    # 肠道准备至肠镜检查时间间隔
    if interval_group:
        old_interval = decode_onehot_group(original_row, interval_group, interval_label_map)

        for target_col in interval_group:
            target_label = interval_label_map.get(target_col, target_col)

            if target_label == old_interval:
                continue

            actions.append({
                "干预类别": "肠道准备至肠镜检查时间间隔",
                "干预措施": f"肠道准备至肠镜间隔：{old_interval} → {target_label}",
                "原始取值": old_interval,
                "干预后取值": target_label,
                "改变变量": ",".join(interval_group),
                "apply": lambda row, g=interval_group, t=target_col: set_onehot_group(row, g, t)
            })

    return actions


def evaluate_single_interventions(
    model,
    patient_model_df,
    patient_raw_df,
    feature_order,
    scaler,
    feature_name_map,
    intervention_threshold,
    min_absolute_reduction=0.0,
    exclude_fasting=False
):
    original_row = patient_model_df.iloc[0][feature_order].copy()
    original_prob = predict_single_risk(model, original_row, feature_order)

    actions = generate_single_intervention_candidates(
        original_row=original_row,
        feature_order=feature_order,
        scaler=scaler,
        feature_name_map=feature_name_map,
        exclude_fasting=exclude_fasting
    )

    rows = []
    all_rows = []

    for i, action in enumerate(actions, start=1):
        cf_row = action["apply"](original_row)
        cf_row = postprocess_row(cf_row, feature_order)

        cf_prob = predict_single_risk(model, cf_row, feature_order)

        abs_red = original_prob - cf_prob
        rel_red = abs_red / original_prob if original_prob > 0 else np.nan
        risk_decreasing = abs_red > min_absolute_reduction

        scenario_row = {
            "情景编号": i,
            "干预类型": "单项干预",
            "干预类别": action["干预类别"],
            "干预措施": action["干预措施"],
            "改变变量": action["改变变量"],
            "原始取值": action["原始取值"],
            "干预后取值": action["干预后取值"],
            "原始预测概率": original_prob,
            "干预后预测概率": cf_prob,
            "绝对风险下降": abs_red,
            "相对风险下降": rel_red,
            "是否使风险下降": risk_decreasing,
            "是否低于干预阈值": cf_prob < intervention_threshold,
            "解释": classify_suggestion(cf_prob, intervention_threshold) if risk_decreasing else "预测风险未下降"
        }

        all_rows.append(scenario_row)

        if risk_decreasing:
            rows.append(scenario_row)

    df = pd.DataFrame(rows)
    all_df = pd.DataFrame(all_rows)

    if not df.empty:
        df = df.sort_values(
            by="绝对风险下降",
            ascending=False
        ).reset_index(drop=True)

    return df, all_df


def evaluate_pairwise_interventions(
    model,
    patient_model_df,
    patient_raw_df,
    feature_order,
    scaler,
    feature_name_map,
    intervention_threshold,
    min_absolute_reduction=0.0,
    exclude_fasting=False
):
    original_row = patient_model_df.iloc[0][feature_order].copy()
    original_prob = predict_single_risk(model, original_row, feature_order)

    actions = generate_single_intervention_candidates(
        original_row=original_row,
        feature_order=feature_order,
        scaler=scaler,
        feature_name_map=feature_name_map,
        exclude_fasting=exclude_fasting
    )

    rows = []
    all_rows = []

    for pair_id, (action_a, action_b) in enumerate(itertools.combinations(actions, 2), start=1):
        cf_row = original_row.copy()

        cf_row = action_a["apply"](cf_row)
        cf_row = action_b["apply"](cf_row)

        cf_row = postprocess_row(cf_row, feature_order)

        cf_prob = predict_single_risk(model, cf_row, feature_order)

        abs_red = original_prob - cf_prob
        rel_red = abs_red / original_prob if original_prob > 0 else np.nan
        risk_decreasing = abs_red > min_absolute_reduction

        scenario_row = {
            "情景编号": pair_id,
            "干预类型": "两两组合干预",
            "干预类别": "组合干预",
            "干预措施": action_a["干预措施"] + " + " + action_b["干预措施"],
            "改变变量": action_a["改变变量"] + " + " + action_b["改变变量"],
            "原始取值": str(action_a["原始取值"]) + " + " + str(action_b["原始取值"]),
            "干预后取值": str(action_a["干预后取值"]) + " + " + str(action_b["干预后取值"]),
            "原始预测概率": original_prob,
            "干预后预测概率": cf_prob,
            "绝对风险下降": abs_red,
            "相对风险下降": rel_red,
            "是否使风险下降": risk_decreasing,
            "是否低于干预阈值": cf_prob < intervention_threshold,
            "解释": classify_suggestion(cf_prob, intervention_threshold) if risk_decreasing else "预测风险未下降"
        }

        all_rows.append(scenario_row)

        if risk_decreasing:
            rows.append(scenario_row)

    df = pd.DataFrame(rows)
    all_df = pd.DataFrame(all_rows)

    if not df.empty:
        df = df.sort_values(
            by="绝对风险下降",
            ascending=False
        ).reset_index(drop=True)

    return df, all_df


def scan_risk_decreasing_measures(
    model,
    patient_model_df,
    patient_raw_df,
    feature_order,
    scaler,
    feature_name_map,
    intervention_threshold=0.135,
    min_absolute_reduction=0.0,
    evaluate_pairwise=True,
    exclude_fasting=False
):
    single_df, _ = evaluate_single_interventions(
        model=model,
        patient_model_df=patient_model_df,
        patient_raw_df=patient_raw_df,
        feature_order=feature_order,
        scaler=scaler,
        feature_name_map=feature_name_map,
        intervention_threshold=intervention_threshold,
        min_absolute_reduction=min_absolute_reduction,
        exclude_fasting=exclude_fasting
    )

    if evaluate_pairwise:
        pairwise_df, _ = evaluate_pairwise_interventions(
            model=model,
            patient_model_df=patient_model_df,
            patient_raw_df=patient_raw_df,
            feature_order=feature_order,
            scaler=scaler,
            feature_name_map=feature_name_map,
            intervention_threshold=intervention_threshold,
            min_absolute_reduction=min_absolute_reduction,
            exclude_fasting=exclude_fasting
        )
    else:
        pairwise_df = pd.DataFrame()

    all_list = []

    if not single_df.empty:
        all_list.append(single_df)

    if not pairwise_df.empty:
        all_list.append(pairwise_df)

    if all_list:
        all_df = pd.concat(all_list, axis=0, ignore_index=True)
        all_df = all_df.sort_values(
            by="绝对风险下降",
            ascending=False
        ).reset_index(drop=True)
    else:
        all_df = pd.DataFrame()

    return single_df, pairwise_df, all_df


def add_display_columns(df):
    if df.empty:
        return df

    df = df.copy()

    df["原始风险"] = df["原始预测概率"].apply(format_probability)
    df["干预后风险"] = df["干预后预测概率"].apply(format_probability)

    df["绝对风险下降百分点"] = (
        df["绝对风险下降"] * 100
    ).round(1)

    df["相对风险下降百分比"] = (
        df["相对风险下降"] * 100
    ).round(1)

    return df
