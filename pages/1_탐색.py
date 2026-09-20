import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------------------------------
# 1. 페이지 기본 설정
# ----------------------------------------------------
st.set_page_config(
    page_title="탐색 - 뇌졸중 예측 실습실",
    page_icon="🔍",
    layout="wide"
)

# ----------------------------------------------------
# 2. 데이터 불러오기 (첫 화면과 동일한 방식)
# ----------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.title("🔍 데이터 탐색")
st.markdown("뇌졸중 데이터를 다양한 그래프와 표로 살펴봅니다.")
st.divider()

# ----------------------------------------------------
# 3. 나이 & 평균 혈당 분포 히스토그램 (나란히 배치)
# ----------------------------------------------------
st.subheader("📊 나이와 평균 혈당 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df, x="age", nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이"}
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df, x="avg_glucose_level", nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()

# ----------------------------------------------------
# 4. 뇌졸중 여부에 따른 나이 & 평균 혈당 상자그림 + 평균값 표
# ----------------------------------------------------
st.subheader("📦 뇌졸중 여부에 따른 나이·평균 혈당 비교")

# stroke 열(0, 1)을 사람이 읽기 편한 이름으로 바꾼 열을 새로 만듦
df["뇌졸중_여부"] = df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

col3, col4 = st.columns(2)

with col3:
    fig_box_age = px.box(
        df, x="뇌졸중_여부", y="age",
        title="뇌졸중 여부별 나이 분포",
        labels={"뇌졸중_여부": "뇌졸중 여부", "age": "나이"}
    )
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    fig_box_glucose = px.box(
        df, x="뇌졸중_여부", y="avg_glucose_level",
        title="뇌졸중 여부별 평균 혈당 분포",
        labels={"뇌졸중_여부": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값을 표로 정리
평균값_표 = df.groupby("뇌졸중_여부")[["age", "avg_glucose_level"]].mean().round(2)
평균값_표.columns = ["나이 평균", "평균 혈당 평균"]
st.markdown("**그룹별 평균값**")
st.dataframe(평균값_표, use_container_width=True)

st.divider()

# ----------------------------------------------------
# 5. 고혈압 / 심장병 유무에 따른 뇌졸중 비율 막대그래프
# ----------------------------------------------------
st.subheader("📈 고혈압·심장병 유무에 따른 뇌졸중 비율")

col5, col6 = st.columns(2)

with col5:
    고혈압_비율 = df.groupby("hypertension")["stroke"].mean().reset_index()
    고혈압_비율["hypertension"] = 고혈압_비율["hypertension"].map({0: "고혈압 없음", 1: "고혈압 있음"})
    고혈압_비율["stroke"] = (고혈압_비율["stroke"] * 100).round(2)

    fig_hyper = px.bar(
        고혈압_비율, x="hypertension", y="stroke",
        title="고혈압 유무에 따른 뇌졸중 비율(%)",
        labels={"hypertension": "고혈압 유무", "stroke": "뇌졸중 비율(%)"},
        text="stroke"
    )
    st.plotly_chart(fig_hyper, use_container_width=True)

with col6:
    심장병_비율 = df.groupby("heart_disease")["stroke"].mean().reset_index()
    심장병_비율["heart_disease"] = 심장병_비율["heart_disease"].map({0: "심장병 없음", 1: "심장병 있음"})
    심장병_비율["stroke"] = (심장병_비율["stroke"] * 100).round(2)

    fig_heart = px.bar(
        심장병_비율, x="heart_disease", y="stroke",
        title="심장병 유무에 따른 뇌졸중 비율(%)",
        labels={"heart_disease": "심장병 유무", "stroke": "뇌졸중 비율(%)"},
        text="stroke"
    )
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# ----------------------------------------------------
# 6. bmi 결측치 그룹의 뇌졸중 비율 vs 전체 뇌졸중 비율
# ----------------------------------------------------
st.subheader("🧮 BMI 결측치 그룹과 전체 그룹의 뇌졸중 비율 비교")

bmi_결측_인원수 = df["bmi"].isnull().sum()
bmi_결측_뇌졸중_비율 = round(df[df["bmi"].isnull()]["stroke"].mean() * 100, 2)
전체_뇌졸중_비율 = round(df["stroke"].mean() * 100, 2)

비교표 = pd.DataFrame({
    "구분": ["BMI 결측치 그룹", "전체 데이터"],
    "인원수": [bmi_결측_인원수, len(df)],
    "뇌졸중 비율(%)": [bmi_결측_뇌졸중_비율, 전체_뇌졸중_비율]
})

st.dataframe(비교표, use_container_width=True)

st.divider()

# ----------------------------------------------------
# 7. 흡연 상태별 사람 수 표
# ----------------------------------------------------
st.subheader("🚬 흡연 상태(smoking_status)별 사람 수")

흡연_상태별_인원수 = df["smoking_status"].value_counts().reset_index()
흡연_상태별_인원수.columns = ["흡연 상태", "사람 수"]

st.dataframe(흡연_상태별_인원수, use_container_width=True)
