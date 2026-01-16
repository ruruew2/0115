import streamlit as st

# 페이지 설정 
st.set_page_config(page_title="나의 스트림릿", page_icon="❌✖️👻👻")

# side bar
with st.sidebar:
    st.header('📩')
    user_color = st.color_picker("theme color choice", "#55abca")
    user_speed = st.slider("학습 속도 조절", 0, 100, 50)

    # user_speed를 소문자로 수정
    st.info(f"선택한색상: {user_color}, 학습속도 : {user_speed}")

# 메인 타이틀
st.title("🧊 멋진 대시보드")
st.subheader("good")


# tab 내용 분할

tab1, tab2, tab3 = st.tabs(["소개","데이터분석","채팅"])

with tab1:
    st.subheader("hi")
    st.write("첫 번째 탭, 사이드바 설정을 바꿔보세요.")
    # 칼럼 이용해서 화면 가로 분할
col1, col2 = st.columns(2)

with col1:
    # 하이픈(-)을 등호(=)로 수정
    st.metric(label="배터리 잔량 :", value="80%", delta="-10%")
with col2:
    st.metric(label="현재 속도", value=f"{user_speed} km/h", delta="1.2 km/h")


with tab2:
    st.subheader("hello")
    st.info("두 번째 탭, 사이드바 설정을 바꿔보세요.")

    with st.expander('도움말'):
        st.write('탭과 칼럼을 조절하면 복잡한 화면이 정리됩니다.')

with tab3:
    st.subheader("chat=bot 입니다.")
    st.title("::챗봇::")
    st.write("안녕하세요, 저는 여러분의 챗봇 입니다.")

    # text box (입력 위젯)
    msg = st.text_input("메세지를 입력하세요 : ")
    if msg:
        st.write(msg)

