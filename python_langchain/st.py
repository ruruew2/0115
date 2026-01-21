import streamlit as st  # streamlit으로 수정

st.title("::챗봇::")
st.write("안녕하세요, 저는 여러분의 챗봇 입니다.")

# text box (입력 위젯)
name = st.text_input("이름을 입력하세요 : ")
if name:
    st.write(f"안녕하세요, {name}님 만나서 반가워요.")
