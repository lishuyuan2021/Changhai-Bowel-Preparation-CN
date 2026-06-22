# ============================================================
# Streamlit App 中文版
# 肠道准备不充分风险预测模型
# Final Model: Raw Stacking Classifier
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_utils import (
    load_deploy_pack,
    build_patient_input_form,
    predict_patient,
    classify_binary_risk,
    format_probability,
    get_feature_display_map,
    setup_chinese_font,
)

from shap_utils import (
    get_shap_explainer,
    compute_patient_shap,
    compute_global_shap,
    plot_patient_waterfall,
    plot_global_beeswarm,
)

from counterfactual_utils import (
    scan_risk_decreasing_measures,
    add_display_columns,
)


# ============================================================
# 页面设置
# ============================================================

st.set_page_config(
    page_title="肠道准备风险预测工具",
    page_icon="🩺",
    layout="wide"
)

setup_chinese_font()

st.title("🩺 肠道准备风险预测工具")
st.caption(
    "最终模型：Stacking Classifier。"
    "本工具用于预测患者肠道准备不充分风险，并提供模型解释与风险下降干预建议。"
)

st.warning(
    "本工具仅用于科研与临床决策辅助，不能替代医生判断或本单位肠道准备规范。"
)


# ============================================================
# 侧边栏设置
# ============================================================

st.sidebar.header("模型设置")

DEPLOY_DIR = st.sidebar.text_input(
    "部署文件夹",
    value="Final_Deploy_Stacking_V2"
)

INTERVENTION_THRESHOLD = st.sidebar.number_input(
    "高风险判定阈值",
    min_value=0.000,
    max_value=1.000,
    value=0.135,
    step=0.001,
    format="%.3f"
)

RUN_GLOBAL_SHAP = st.sidebar.checkbox(
    "显示 SHAP 蜂巢图",
    value=True
)

GLOBAL_SHAP_N = st.sidebar.slider(
    "SHAP 蜂巢图抽样样本量",
    min_value=20,
    max_value=200,
    value=60,
    step=10
)

RUN_PAIRWISE_CF = st.sidebar.checkbox(
    "评估两两组合干预建议",
    value=True
)

EXCLUDE_FASTING = st.sidebar.checkbox(
    "反事实建议中排除“禁食”策略",
    value=False
)

MIN_ABSOLUTE_REDUCTION = st.sidebar.number_input(
    "建议保留的最小绝对风险下降值",
    min_value=0.000,
    max_value=0.200,
    value=0.000,
    step=0.001,
    format="%.3f"
)


# ============================================================
# 读取模型
# ============================================================

try:
    deploy_pack = load_deploy_pack(DEPLOY_DIR)
except Exception as e:
    st.error(f"模型部署包读取失败：{e}")
    st.stop()

model = deploy_pack["final_deploy_model"]
feature_order = deploy_pack["feature_order"]
deploy_info = deploy_pack.get("deploy_info", {})
datasets = deploy_pack["datasets"]
scaler = deploy_pack.get("scaler", None)

if scaler is None:
    st.error(
        "部署包中未找到 scaler。"
        "由于 Age、BMI、DietaryRestrictionDays 需要标准化，网页预测必须读取 scaler。"
    )
    st.stop()

X_train_final = datasets["X_train_final"][feature_order]
X_val_final = datasets["X_val_final"][feature_order]

feature_name_map = get_feature_display_map()

st.sidebar.success("模型部署包读取成功")


# ============================================================
# 患者变量输入
# ============================================================

st.header("1. 输入患者变量")

patient_raw_df, patient_model_df, input_summary_df = build_patient_input_form(
    feature_order=feature_order,
    scaler=scaler,
    feature_name_map=feature_name_map,
)

with st.expander("查看模型编码后的输入变量", expanded=False):
    st.dataframe(patient_model_df.T.rename(columns={0: "取值"}), use_container_width=True)

with st.expander("查看临床输入信息汇总", expanded=False):
    st.dataframe(input_summary_df, use_container_width=True)


# ============================================================
# 预测结果
# ============================================================

st.header("2. 预测结果")

if st.button("开始预测", type="primary"):

    predicted_prob = predict_patient(
        model=model,
        patient_model_df=patient_model_df,
        feature_order=feature_order
    )

    risk_label = classify_binary_risk(
        predicted_prob,
        threshold=INTERVENTION_THRESHOLD
    )

    st.session_state["predicted_prob"] = predicted_prob
    st.session_state["risk_label"] = risk_label
    st.session_state["patient_model_df"] = patient_model_df
    st.session_state["patient_raw_df"] = patient_raw_df
    st.session_state["input_summary_df"] = input_summary_df

if "predicted_prob" in st.session_state:

    predicted_prob = st.session_state["predicted_prob"]
    risk_label = st.session_state["risk_label"]
    patient_model_df = st.session_state["patient_model_df"]
    patient_raw_df = st.session_state["patient_raw_df"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "肠道准备不充分预测概率",
            format_probability(predicted_prob)
        )

    with col2:
        st.metric(
            "风险分层",
            risk_label
        )

    with col3:
        st.metric(
            "干预阈值",
            f"{INTERVENTION_THRESHOLD:.3f}"
        )

    if predicted_prob >= INTERVENTION_THRESHOLD:
        st.error("高风险：模型提示该患者可能需要强化肠道准备。")
    else:
        st.success("低风险：模型提示该患者可考虑常规肠道准备。")


    # ========================================================
    # SHAP 解释
    # ========================================================

    st.header("3. SHAP 模型解释")

    with st.spinner("正在初始化 SHAP 解释器……"):
        explainer = get_shap_explainer(
            _model=model,
            _background_data=X_train_final,
            feature_order=feature_order,
            background_n=80,
            random_state=2026
        )

    tab1, tab2 = st.tabs(["患者个体瀑布图", "全局 SHAP 蜂巢图"])

    with tab1:
        st.subheader("患者个体 SHAP 瀑布图")

        with st.spinner("正在计算该患者的 SHAP 值……"):
            patient_shap_values = compute_patient_shap(
                explainer=explainer,
                patient_model_df=patient_model_df,
                patient_raw_df=patient_raw_df,
                feature_order=feature_order,
                feature_name_map=feature_name_map,
            )

        fig_waterfall = plot_patient_waterfall(
            patient_shap_values,
            predicted_prob=predicted_prob,
            max_display=15
        )

        st.pyplot(fig_waterfall, clear_figure=True)

        st.info(
            "瀑布图展示各变量如何推动模型预测风险升高或降低。"
            "正向 SHAP 值表示增加肠道准备不充分预测概率，负向 SHAP 值表示降低预测概率。"
        )

    with tab2:
        st.subheader("全局 SHAP 蜂巢图")

        if RUN_GLOBAL_SHAP:
            with st.spinner("正在计算全局 SHAP 值，首次运行可能需要较长时间……"):
                global_shap_values = compute_global_shap(
                    explainer=explainer,
                    X_global_source=X_val_final,
                    feature_order=feature_order,
                    feature_name_map=feature_name_map,
                    sample_n=GLOBAL_SHAP_N,
                    random_state=2026
                )

            fig_beeswarm = plot_global_beeswarm(
                global_shap_values,
                max_display=20
            )

            st.pyplot(fig_beeswarm, clear_figure=True)

            st.info(
                "蜂巢图用于展示模型在样本总体中的主要影响因素。"
                "横轴越偏右，表示该变量越倾向于增加肠道准备不充分风险。"
            )
        else:
            st.info("如需显示蜂巢图，请在左侧边栏勾选“显示 SHAP 蜂巢图”。")


    # ========================================================
    # 反事实 / 干预建议
    # ========================================================

    st.header("4. 反事实干预建议")

    with st.spinner("正在扫描可降低预测风险的干预情景……"):

        cf_single_df, cf_pairwise_df, cf_all_df = scan_risk_decreasing_measures(
            model=model,
            patient_model_df=patient_model_df,
            patient_raw_df=patient_raw_df,
            feature_order=feature_order,
            scaler=scaler,
            feature_name_map=feature_name_map,
            intervention_threshold=INTERVENTION_THRESHOLD,
            min_absolute_reduction=MIN_ABSOLUTE_REDUCTION,
            evaluate_pairwise=RUN_PAIRWISE_CF,
            exclude_fasting=EXCLUDE_FASTING
        )

    if cf_all_df.empty:
        st.info(
            "在预设的可干预变量范围内，未发现可使模型预测风险下降的干预情景。"
        )
    else:
        cf_all_display = add_display_columns(cf_all_df)

        st.subheader("风险下降建议 Top 20")

        display_cols = [
            "干预类型",
            "干预措施",
            "原始风险",
            "干预后风险",
            "绝对风险下降百分点",
            "相对风险下降百分比",
            "是否低于干预阈值",
            "解释"
        ]

        st.dataframe(
            cf_all_display[display_cols].head(20),
            use_container_width=True
        )

        csv_bytes = cf_all_display.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        st.download_button(
            label="下载反事实建议 CSV",
            data=csv_bytes,
            file_name="反事实干预建议.csv",
            mime="text/csv"
        )

        with st.expander("单项干预建议", expanded=False):
            if cf_single_df.empty:
                st.info("未发现可降低预测风险的单项干预。")
            else:
                st.dataframe(
                    add_display_columns(cf_single_df),
                    use_container_width=True
                )

        with st.expander("两两组合干预建议", expanded=False):
            if cf_pairwise_df.empty:
                st.info("未发现可降低预测风险的两两组合干预，或未启用组合干预扫描。")
            else:
                st.dataframe(
                    add_display_columns(cf_pairwise_df).head(50),
                    use_container_width=True
                )

else:
    st.info("请输入患者变量后点击“开始预测”。")
