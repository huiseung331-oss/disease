import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# ----------------------------------------------------
# 1. 페이지 기본 설정
# ----------------------------------------------------
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------------------------------
# 2. 데이터 불러오기
# ----------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.title("🤖 분류 모델 만들기")
st.markdown("뇌졸중(stroke=1)을 **양성**으로 두고, 여러 속성으로 뇌졸중을 예측하는 모델을 만들어 봅니다.")
st.divider()

# ----------------------------------------------------
# 3. 속성 선택 (열 이름 <-> 우리말 이름 매핑)
# ----------------------------------------------------
속성_이름표 = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
우리말_to_열이름 = {v: k for k, v in 속성_이름표.items()}

st.subheader("1️⃣ 입력 속성 선택")

기본_선택 = ["나이", "평균 혈당", "고혈압", "심장병"]  # bmi를 뺀 넷

선택된_우리말_목록 = st.multiselect(
    "모델의 입력으로 사용할 속성을 고르세요 (2개 이상)",
    options=list(속성_이름표.values()),
    default=기본_선택
)

if len(선택된_우리말_목록) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해야 모델을 만들 수 있어요. 위에서 속성을 더 선택해주세요.")
    st.stop()

선택된_열이름_목록 = [우리말_to_열이름[이름] for 이름 in 선택된_우리말_목록]

st.divider()

# ----------------------------------------------------
# 4. 데이터 준비: 번호순 정렬 -> 10명씩 묶어 앞 3명 테스트용
# ----------------------------------------------------
st.subheader("2️⃣ 데이터 나누기 (훈련용 / 테스트용)")

작업용_df = df.copy()
작업용_df = 작업용_df.sort_values("id").reset_index(drop=True)

# 10명씩 묶었을 때 묶음 안에서의 순서(0~9)를 계산
작업용_df["묶음_내_순서"] = np.arange(len(작업용_df)) % 10

# 순서가 0, 1, 2인 사람은 테스트용, 나머지는 훈련용
테스트용_df = 작업용_df[작업용_df["묶음_내_순서"] < 3].copy()
훈련용_df = 작업용_df[작업용_df["묶음_내_순서"] >= 3].copy()

st.markdown(f"- 전체 인원: **{len(작업용_df):,}명**")
st.markdown(f"- 훈련용 인원: **{len(훈련용_df):,}명**")
st.markdown(f"- 테스트용 인원: **{len(테스트용_df):,}명** (10명씩 묶어 앞 3명씩)")

# ----------------------------------------------------
# 5. bmi 결측치 처리 (선택된 경우에만, 훈련용 중앙값으로)
# ----------------------------------------------------
if "bmi" in 선택된_열이름_목록:
    bmi_중앙값 = 훈련용_df["bmi"].median()
    훈련용_df["bmi"] = 훈련용_df["bmi"].fillna(bmi_중앙값)
    테스트용_df["bmi"] = 테스트용_df["bmi"].fillna(bmi_중앙값)
    st.markdown(f"- 체질량지수(bmi)의 빈 값은 훈련용 중앙값인 **{bmi_중앙값:.2f}**로 채웠어요.")

X_train = 훈련용_df[선택된_열이름_목록]
y_train = 훈련용_df["stroke"]
X_test = 테스트용_df[선택된_열이름_목록]
y_test = 테스트용_df["stroke"]

st.divider()

# ----------------------------------------------------
# 6. 모델 학습
# ----------------------------------------------------
st.subheader("3️⃣ 모델 학습 및 정확도 비교")

# (1) 로지스틱 회귀
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)

# (2) 의사결정트리 (질문 3번까지, 마지막 마디 5명 미만이면 그만 나누기, 난수 고정)
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
tree_model.fit(X_train, y_train)

# (3) 더미 분류기 (입력을 보지 않고 많은 쪽으로만 답하는 모델)
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

# 각 모델의 훈련/테스트 정확도 계산
def 정확도_계산(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    return train_acc, test_acc

logreg_train_acc, logreg_test_acc = 정확도_계산(logreg_model, X_train, y_train, X_test, y_test)
tree_train_acc, tree_test_acc = 정확도_계산(tree_model, X_train, y_train, X_test, y_test)
dummy_train_acc, dummy_test_acc = 정확도_계산(dummy_model, X_train, y_train, X_test, y_test)

# ----------------------------------------------------
# 7. 정확도 카드 세 개
# ----------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("로지스틱 회귀 (확률로 답하는 모델)", f"{logreg_test_acc:.4f}")
    st.caption(f"훈련 정확도: {logreg_train_acc:.4f}  |  테스트 정확도: {logreg_test_acc:.4f}")

with col2:
    st.metric("의사결정트리 (질문으로 답하는 모델)", f"{tree_test_acc:.4f}")
    st.caption(f"훈련 정확도: {tree_train_acc:.4f}  |  테스트 정확도: {tree_test_acc:.4f}")

with col3:
    st.metric("항상 다수 쪽으로 답하는 모델", f"{dummy_test_acc:.4f}")
    st.caption(f"훈련 정확도: {dummy_train_acc:.4f}  |  테스트 정확도: {dummy_test_acc:.4f}")

st.divider()

# ----------------------------------------------------
# 8. 산점도 + 로지스틱 회귀 결정 경계선 + 트리 영역 색칠
# ----------------------------------------------------
st.subheader("4️⃣ 두 속성으로 보는 분류 결과")

col4, col5 = st.columns(2)
with col4:
    x축_우리말 = st.selectbox("가로축으로 사용할 속성", 선택된_우리말_목록, index=0)
with col5:
    남은_목록 = [이름 for 이름 in 선택된_우리말_목록 if 이름 != x축_우리말]
    y축_우리말 = st.selectbox("세로축으로 사용할 속성", 남은_목록, index=0)

x축_열이름 = 우리말_to_열이름[x축_우리말]
y축_열이름 = 우리말_to_열이름[y축_우리말]

# 두 축이 아닌 나머지 속성은 테스트 데이터의 중앙값으로 고정
고정할_속성_목록 = [c for c in 선택된_열이름_목록 if c not in [x축_열이름, y축_열이름]]
고정값_딕셔너리 = {}
for 속성 in 고정할_속성_목록:
    고정값 = X_test[속성].median()
    고정값_딕셔너리[속성] = 고정값

if 고정값_딕셔너리:
    고정_설명_목록 = [f"{속성_이름표[k]} = {v:.2f}" for k, v in 고정값_딕셔너리.items()]
    st.markdown("**고정한 속성 값 (테스트 데이터의 중앙값):** " + ", ".join(고정_설명_목록))
else:
    st.markdown("고정할 속성이 없습니다. (선택한 속성이 두 개뿐입니다)")

# 그래프 축 범위 설정
x_min, x_max = X_test[x축_열이름].min(), X_test[x축_열이름].max()
y_min, y_max = X_test[y축_열이름].min(), X_test[y축_열이름].max()
x_여백 = (x_max - x_min) * 0.05
y_여백 = (y_max - y_min) * 0.05

# 배경 색칠(의사결정트리의 영역)을 위한 격자 생성
격자_수 = 200
xx, yy = np.meshgrid(
    np.linspace(x_min - x_여백, x_max + x_여백, 격자_수),
    np.linspace(y_min - y_여백, y_max + y_여백, 격자_수)
)

# 격자용 입력 데이터프레임 만들기 (고정 속성은 중앙값으로 채움)
격자_df = pd.DataFrame({x축_열이름: xx.ravel(), y축_열이름: yy.ravel()})
for 속성, 값 in 고정값_딕셔너리.items():
    격자_df[속성] = 값
격자_df = 격자_df[선택된_열이름_목록]  # 학습 때와 같은 열 순서로 맞추기

# 의사결정트리로 격자의 각 점을 예측해서 배경색 결정
tree_예측 = tree_model.predict(격자_df).reshape(xx.shape)

fig_scatter = go.Figure()

# 트리의 영역을 옅은 색으로 칠하기 (등고선 방식 사용)
fig_scatter.add_trace(go.Contour(
    x=np.linspace(x_min - x_여백, x_max + x_여백, 격자_수),
    y=np.linspace(y_min - y_여백, y_max + y_여백, 격자_수),
    z=tree_예측,
    showscale=False,
    opacity=0.25,
    colorscale=[[0, "blue"], [1, "red"]],
    contours=dict(start=0, end=1, size=1),
    line=dict(width=0),
    name="의사결정트리 영역",
    hoverinfo="skip"
))

# 테스트 데이터 점 찍기 (실제 뇌졸중 여부로 색 구분)
테스트_시각화_df = X_test.copy()
테스트_시각화_df["실제_뇌졸중"] = y_test.map({0: "뇌졸중 없음", 1: "뇌졸중 있음"}).values

for 라벨, 색 in [("뇌졸중 없음", "blue"), ("뇌졸중 있음", "red")]:
    부분_df = 테스트_시각화_df[테스트_시각화_df["실제_뇌졸중"] == 라벨]
    fig_scatter.add_trace(go.Scatter(
        x=부분_df[x축_열이름],
        y=부분_df[y축_열이름],
        mode="markers",
        name=라벨,
        marker=dict(color=색, size=6, opacity=0.6)
    ))

# 로지스틱 회귀의 0.5 결정 경계선 계산
# 로지스틱 회귀 식: w0 + w1*x1 + w2*x2 + ... = 0  일 때 확률이 0.5
# 여기서는 x축, y축 속성의 계수만 사용해서 "x축 값에 따른 y축 경계값"을 구함
계수 = logreg_model.coef_[0]
절편 = logreg_model.intercept_[0]

열이름_순서 = list(X_train.columns)
x축_위치 = 열이름_순서.index(x축_열이름)
y축_위치 = 열이름_순서.index(y축_열이름)

x축_계수 = 계수[x축_위치]
y축_계수 = 계수[y축_위치]

# 고정된 속성들이 경계선 계산에 미치는 영향 미리 더해두기
고정_영향값 = 절편
for 속성, 값 in 고정값_딕셔너리.items():
    속성_위치 = 열이름_순서.index(속성)
    고정_영향값 += 계수[속성_위치] * 값

선이_그림안에_있음 = False

if abs(y축_계수) > 1e-10:
    # y = -(고정_영향값 + x축_계수 * x) / y축_계수
    x_선분 = np.linspace(x_min - x_여백, x_max + x_여백, 200)
    y_선분 = -(고정_영향값 + x축_계수 * x_선분) / y축_계수

    # 계산된 y값이 그래프 범위 안에 있는 부분만 표시
    범위안_인덱스 = (y_선분 >= y_min - y_여백) & (y_선분 <= y_max + y_여백)

    if 범위안_인덱스.sum() > 0:
        선이_그림안에_있음 = True
        fig_scatter.add_trace(go.Scatter(
            x=x_선분[범위안_인덱스],
            y=y_선분[범위안_인덱스],
            mode="lines",
            name="로지스틱 회귀 결정경계(0.5)",
            line=dict(color="black", dash="dash", width=2)
        ))

fig_scatter.update_layout(
    title="테스트 데이터 분포와 분류 경계",
    xaxis_title=x축_우리말,
    yaxis_title=y축_우리말,
    legend_title="구분"
)

st.plotly_chart(fig_scatter, use_container_width=True)

if not 선이_그림안에_있음:
    st.info("ℹ️ 로지스틱 회귀의 결정 경계선이 이 그래프의 범위 밖에 있어서 표시되지 않았습니다.")

st.divider()

# ----------------------------------------------------
# 9. 의사결정트리 가지 그림 (graphviz DOT 문자열)
# ----------------------------------------------------
st.subheader("5️⃣ 의사결정트리 구조")

tree = tree_model.tree_
특성_이름_목록 = 선택된_열이름_목록


def 노드를_DOT으로_변환(node_id, dot_라인_목록):
    """트리의 각 노드를 재귀적으로 방문하며 DOT 문자열 라인을 만드는 함수"""
    샘플수 = int(tree.n_node_samples[node_id])
    # value는 [클래스0 개수, 클래스1 개수] 형태
    클래스0_개수 = int(tree.value[node_id][0][0])
    클래스1_개수 = int(tree.value[node_id][0][1])
    양성_비율 = 클래스1_개수 / 샘플수 if 샘플수 > 0 else 0

    왼쪽_자식 = tree.children_left[node_id]
    오른쪽_자식 = tree.children_right[node_id]
    잎_노드인가 = (왼쪽_자식 == -1 and 오른쪽_자식 == -1)

    if 잎_노드인가:
        # 잎 노드: 다수결로 답을 정함
        if 클래스1_개수 > 클래스0_개수:
            답 = "뇌졸중 있음"
            색 = "#ffb3b3"  # 연한 빨강
        else:
            답 = "뇌졸중 없음"
            색 = "#b3d1ff"  # 연한 파랑
        라벨 = f"인원 {샘플수}명\\n뇌졸중 {클래스1_개수}명\\n비율 {양성_비율:.2f}\\n답: {답}"
        dot_라인_목록.append(f'{node_id} [label="{라벨}", style=filled, fillcolor="{색}"];')
    else:
        속성_영문 = 특성_이름_목록[tree.feature[node_id]]
        속성_우리말 = 속성_이름표[속성_영문]
        기준값 = tree.threshold[node_id]
        질문 = f"{속성_우리말} <= {기준값:.2f} ?"
        라벨 = f"{질문}\\n인원 {샘플수}명\\n뇌졸중 {클래스1_개수}명\\n비율 {양성_비율:.2f}"
        dot_라인_목록.append(f'{node_id} [label="{라벨}", style=filled, fillcolor="#f0f0f0"];')

        # 왼쪽(예, 조건 만족), 오른쪽(아니요, 조건 불만족) 가지 연결
        dot_라인_목록.append(f'{node_id} -> {왼쪽_자식} [label="예"];')
        dot_라인_목록.append(f'{node_id} -> {오른쪽_자식} [label="아니요"];')

        노드를_DOT으로_변환(왼쪽_자식, dot_라인_목록)
        노드를_DOT으로_변환(오른쪽_자식, dot_라인_목록)


dot_라인_목록 = []
노드를_DOT으로_변환(0, dot_라인_목록)

dot_문자열 = "digraph Tree {\nnode [shape=box, fontname=\"Malgun Gothic\"];\n" + "\n".join(dot_라인_목록) + "\n}"

st.graphviz_chart(dot_문자열)

st.divider()

# ----------------------------------------------------
# 10. 트리 요약 정보
# ----------------------------------------------------
st.subheader("6️⃣ 트리 요약")

# 잎 노드(답을 내는 마디) 찾기
잎_노드_개수 = 0
아님_답_노드_개수 = 0

for node_id in range(tree.node_count):
    왼쪽_자식 = tree.children_left[node_id]
    오른쪽_자식 = tree.children_right[node_id]
    if 왼쪽_자식 == -1 and 오른쪽_자식 == -1:
        잎_노드_개수 += 1
        클래스0_개수 = tree.value[node_id][0][0]
        클래스1_개수 = tree.value[node_id][0][1]
        if 클래스0_개수 > 클래스1_개수:
            아님_답_노드_개수 += 1

st.markdown(f"- 답을 내는 마디(잎 노드)는 모두 **{잎_노드_개수}칸**입니다.")
st.markdown(f"- 그중 **{아님_답_노드_개수}칸**이 '뇌졸중 아님'이라고 답합니다.")

# 트리가 실제로 사용한(질문에 등장한) 속성 찾기
실제_사용된_속성_인덱스 = set(tree.feature[tree.feature >= 0])
실제_사용된_속성_목록 = [특성_이름_목록[i] for i in 실제_사용된_속성_인덱스]

st.markdown("**이 나무가 실제로 물어본 속성:**")
if len(실제_사용된_속성_목록) == 0:
    st.markdown("- 트리가 어떤 속성도 사용하지 않았습니다. (뿌리가 곧 잎인 경우)")
else:
    for 속성 in 실제_사용된_속성_목록:
        st.markdown(f"- {속성_이름표[속성]}")
