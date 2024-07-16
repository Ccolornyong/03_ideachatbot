# st_chatbot.py
import google.generativeai as genai 
import streamlit as st
import pandas as pd

# 엑셀 파일에서 시트 이름 가져오는 함수
def get_sheet_names(file_path):
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    # '건수' 시트를 리스트에서 제거
    if '건수' in sheet_names:
        sheet_names.remove('건수')
    return sheet_names

# csv 파일 가져오는 함수
def get_csv(file_path, sheet_name=None, filter_column=None, filter_value=None, fillna_column=None):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # NON 표시된 행 위쪽 값으로 채우기
    if fillna_column:  # 리스트가 비어있지 않은 경우
        for column in fillna_column:
            if column in df.columns:
                df[column].fillna(method='ffill', inplace=True)

    # 특정 열에서 조건에 맞는 행 가져오기
    if filter_column and filter_value:
        df = df[df[filter_column] == filter_value]
    
    # 특정 조건에 맞춰 보여줘야 할 때도 있으므로 보여주는건 포함하지 않음
    # st.dataframe(df)

    return df

    # df.head()
    # st.dataframe( df.head() )
    # st.write( df.head() )

# 챗봇화면
@st.cache_resource
def load_model():
    model = genai.GenerativeModel('gemini-pro')
    print("model loaded...")
    return model

# 사이드바 화면
st.sidebar.header('Sidebar')

## 사이드바 소제목
option = st.sidebar.selectbox(
'Menu',
    ('Gemini-Bot', '단종 된 제품', '뱅킹화 된 제품', '페이지3'))

# 페이지1 내용
if option == 'Gemini-Bot':
    st.title('Gemini-Bot')
    st.write('대화를 통해 원하는 데이터를 검색해보세요.')

    model = load_model()

    if "chat_session" not in st.session_state:    
        st.session_state["chat_session"] = model.start_chat(history=[]) 

    for content in st.session_state.chat_session.history:
        with st.chat_message("ai" if content.role == "model" else "user"):
            st.markdown(content.parts[0].text)

    if prompt := st.chat_input("메시지를 입력하세요."):    
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("ai"):
            response = st.session_state.chat_session.send_message(prompt)        
            st.markdown(response.text)

elif option == '단종 된 제품':
    st.title('단종 된 제품')
    file_path = "C:/Users/SUNJIN/Documents/인턴/03_ideachatbot/data/년도별 신제품 리스트_냉장-240425.xlsx"
    # 엑셀 파일의 시트 이름 가져오기
    sheet_name = st.selectbox('Select a sheet:', get_sheet_names(file_path))

    df = get_csv(file_path,
            sheet_name=sheet_name,
            fillna_column=["런칭여부","구분"],
            filter_column="단종 여부",
            filter_value="단종")
    if '구분' in df.columns:
        df['구분'] = df['구분'].apply(lambda x: 'ODM' if 'ODM' in str(x) else 'OEM')

    st.dataframe(df, use_container_width=True)

elif option == '뱅킹화 된 제품':
    st.title('뱅킹화 된 제품')
    file_path = "C:/Users/SUNJIN/Documents/인턴/03_ideachatbot/data/년도별 신제품 리스트_냉장-240425.xlsx"
    sheet_name = st.selectbox('Select a sheet:', get_sheet_names(file_path))

    df = get_csv(file_path,
        sheet_name=sheet_name,
        fillna_column=["런칭여부", "구분"],
        filter_column="런칭여부",
        filter_value="뱅킹화")
    
    if '구분' in df.columns:
        df['구분'] = df['구분'].apply(lambda x: 'ODM' if 'ODM' in str(x) else 'OEM')
    
    if '단종 여부' in df.columns:
        df = df.drop(columns=['단종 여부'])

    st.dataframe(df, use_container_width=True)
