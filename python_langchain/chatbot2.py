import streamlit as st

st.title(':: smu 채팅 ::')
st.write("welcome")

# clear_on_submit=True로 수정 (전송 누르면 입력창이 비워집니다)
with st.form("msg_form", clear_on_submit=True):
    msg = st.text_input("내용 입력")
    submitted = st.form_submit_button("전송")

# 전송후 메세지 출력
if submitted:
    st.write(f"Echo >> {msg}")