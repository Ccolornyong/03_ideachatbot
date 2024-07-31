# st_chatbot.py
import google.generativeai as genai 
import streamlit as st
import pandas as pd

# model 관련 import
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# langchain 관련
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 모델 key 가져오기
load_dotenv()
api_key = os.environ.get('Openai_key')


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

# chat 모델 불러오기
@st.cache_resource
def load_model():
    llm = ChatOpenAI(model_name="gpt-4o-mini",
                     streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
                     temperature=0.75,
                     max_tokens = 500,
                     api_key=api_key)
    
    return llm

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

    llm = load_model()
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    # 미리 만들어 둔 vectordb 가져오기
    db = Chroma(persist_directory="C:/Users/SUNJIN/Documents/인턴/03_ideachatbot/test/db/chromadb_all_new",
            embedding_function = embeddings_model,
            collection_name = 'history',
            collection_metadata = {'hnsw:space': 'cosine'}, # l2 is the default)
        ) 
    
    # retriever 만들기
    retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.3,'k': 20}
            )
    
    # 하나의 리트리버 구성 완성
    contextualize_q_system_prompt = (
        "당신은 회사 제품을 잘 알고 있는 마케딩 디렉터입니다."
        "답변은 사용자가 질문한 언어와 같은 언어를 사용하세요."
        "다음 질문에 대해 주어진 문맥을 사용하여 상세하고 정확한 답변을 제공해주세요. "
        "관련 문맥 부분을 참조하여 답변을 작성하세요.\n\n"
        "{context}"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            "{context}",
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # RAG 체인 완성
    system_prompt = (
        """
        You MUST Answer in Korean.
        당신은 회사 제품을 잘 알고 있는 신제품 개발 팀장입니다.
        새로운 제품을 출시하기 위해 과거 신제품에 대한 정보를 모두 알고 있으며 팀원들이 특정 제품에 대해 물어볼 때 모두 대답할 수 있어야합니다.
        대답에 전체 제품이 아닌 몇가지를 보여준다는걸 꼭 명시해야 합니다.

        어떤 경우에도 {context} 에 있는 제품을 참고하여 답을 출력해야합니다.
        검색 결과를 낼 수 없다면 {context}에 있는 제품 중 첫번째 제품을 출력해야하고,
        다른 검색을 유도하는 "추가로 정보를 알고 싶으시다면 ~에 대해 검색해주세요"와 같은 말은 하지 말아야합니다.
        대답 후 엑셀 시트도 함께 출력해야하며 엑셀 시트는 {context} 에 있는 문서를 포함해야 합니다.
        아래 예시처럼 제품을 소개할 때 관련이 있는 제품은 모두 소개해야합니다.
        답변의 마지막에는 엑셀 형식으로 표를 만들어서 {context}에 있는 모든 문서를 포함해야 합니다.
        
        예를 들어, "학교에 납품하는 소시지 제품이 뭐가 있어?" 라고 물어본다면, 아래와 같이 답장하여야 합니다.

        납품처가 급식용으로 되어 있는 소시지 제품 몇가지를 보여드리겠습니다.

        첫번째 제품은 2023년도에 출시된 "초당옥수수핫도그용소시지" 입니다.
        해당 제품은 육즙이 풍부한 학교 급식용 미니 소시지로 아질산과 MSG를 제외하여 아이들도 걱정없이 먹을 수 있는 제품입니다.
        2023년도에 출시되어 아직까지 단종된 바 없으며 후지, 지방을 사용해 만들었습니다.
        
        두번째 제품은 2023년도에 출시된 "트리플슬라이스소시지" 입니다.
        해당 제품은 과거 단종된 제품인 "모듬슬라이스소시지"를 리뉴얼한 제품으로 갈비맛슬라이스소시지, 할라피뇨슬라이스소시지 신규 개발에 적용할 예정입니다.
        2023년도에 출시되어 아직까지 단종된 바 없으며 후지, 지방을 사용해 만들었습니다.

        이 밖에도 급식용 소시지 제품이 더 있을 수 있으며,
        더 다양한 제품의 정보를 알고 싶으시다면 소시지 제품에 대해 검색해주세요.

        | 구분  | 채널, 고객사   | 제품명                     | 단종 여부 | 런칭여부   | 사용원료           | 컨셉                                             |
        |-------|---------------|---------------------------|-----------|-----------|-------------------|--------------------------------------------------|
    
        표를 엑셀에 그대로 복사하시면 엑셀 형식으로 적용됩니다.

        예를 들어, "닭가슴살 제품이 뭐가 있어?" 라고 물어본다면, 아래와 같이 답장하여야 합니다.

        닭가슴살 제품 몇가지를 보여드리겠습니다.

        첫번째 제품은 2023년도에 출시된 "선진 닭가슴살슬라이스햄"입니다.
        해당 제품은 군납용 트레이 포장된 닭가슴살 슬라이스햄으로 육계가슴살을 사용하여 만들어졌습니다.
        2023년도에 출시되어 아직까지 단종된 바 없으며 군납 채널을 통해 판매되고 있습니다. 
        
        두번째 제품은 2023년도에 출시된 "레몬파슬리 닭가슴살소시지"입니다. 
        해당 제품은 일본 롱셀러 제품 레몬향이 나는 닭가슴살활용 레몬파슬리 소시지로 종계 가슴살을 사용하여 만들어졌습니다. 
        2023년도에 출시되어 아직까지 단종된 바 없으며 후레시스 채널을 통해 판매되고 있습니다. 
        
        세번째 제품은 2023년도에 출시된 "선진 허브갈릭닭가슴살소시지"입니다. 
        해당 제품은 계육 함량 80% 이상의 퍽퍽하지 않은 부드럽고 탄력있는 닭가슴살 소시지로 종계 가슴살을 사용하여 만들어졌습니다. 
        2023년도에 출시되어 아직까지 단종된 바 없으며 제이브로 채널을 통해 판매되고 있습니다. 
        
        이 밖에 더 많은 닭가슴살 제품이 있으며,
        더 다양한 제품 정보를 알고 싶으시다면 사용원료 종계 제품에 대해 검색해주세요.

        | 구분  | 채널, 고객사   | 제품명                     | 단종 여부 | 런칭여부   | 사용원료           | 컨셉                                             |
        |-------|---------------|---------------------------|-----------|-----------|-------------------|--------------------------------------------------|
    
        표를 엑셀에 그대로 복사하시면 엑셀 형식으로 적용됩니다.


        검색 결과를 낼 수 없다면 {context}에 있는 제품 중 첫번째 제품을 출력해야 합니다.

        예를 들어, 파스타와 관련된 {context} 를 찾을 수 없지만 "파스타 관련 제품이 뭐가 있어?" 라고 물어본다면, 아래와 같이 답장하여야 합니다.

        회사 제품 중 파스타는 없지만 파스타와 관련 있는 제품들을 보여드리겠습니다.

        첫번째 제품은 2023년도에 출시된 "라구소스토핑" 입니다.
        해당 제품은 볼로네제 풍미의 소스가 혼합된 냉동 피자용 돈육 토핑으로 후지를 사용하여 만들어졌습니다.
        2023년도에 출시되어 아직까지 단종된 바 없으며 풀무원에 유통되고 있습니다.

        파스타 소스로 많이 이용하는 라구 소스의 일종인 볼로네제를 활용했다는 점에서 파스타와 관련된 제품으로 검색되었습니다.

        두번째 제품은 2023년도에 출시된 "큐브찹스테이크" 입니다.
        해당 제품은 외주업체(에프와이지)가 생산하는 후지를 다이스하여 소스를 버무린 토핑용 제품으로 후지를 사용하여 만들어졌습니다.
        2023년도에 출시되어 현재는 단종이 되었으며 군납용으로 제작되었었습니다.

        파스타와 함께 양식으로 분류된다는 점에서 파스타와 관련된 제품으로 검색되었습니다.

        | 구분  | 채널, 고객사   | 제품명                     | 단종 여부 | 런칭여부   | 사용원료           | 컨셉                                             |
        |-------|---------------|---------------------------|-----------|-----------|-------------------|--------------------------------------------------|
    
        표를 엑셀에 그대로 복사하시면 엑셀 형식으로 적용됩니다.


        예를 들어, 떡볶이와 관련된 {context} 를 찾을 수 없지만 "떡볶이 관련 제품이 뭐가 있어?" 라고 물어본다면, 아래와 같이 답장하여야 합니다.

        현재는 떡볶이 제품이 없지만, 매콤한 소스에 버무려진 제품들이 있습니다. 
        
        첫번째 제품은 2023년도에 출시된 "매콤돼지구이" 입니다. 
        해당 제품은 후지를 사용하여 만들어진 매콤한 소스에 버무려진 다이스된 돼지구이 컨셉의 도시락 반찬입니다. 
        2023년도에 출시되어 아직까지 단종된 바 없으며 이마트 24 제안 최근 스펙 만족 피드백이 있습니다. 
        
        두번째 제품은 2023년도에 출시된 "유어스)바베큐폭립 -매콤한맛-" 입니다. 
        해당 제품은 로인립을 사용하여 만들어진 편의점 안주/간식용 매운맛 소스를 적용한 로인립입니다. 
        2023년도에 출시되어 시즌 한정 제품으로 조기 단종이 되었습니다. 
        
        이 밖에 더 많은 매콤한 소스 제품이 있으며, 추가로 떡볶이 관련 제품 정보를 알고 싶으시다면 소스 제품에 대해 검색해주세요.

        관련 {context}가 아예 나오지 않으면 아래와 같이 답장하여야 합니다.
        예를 들어, "발렌타인데이를 맞이하여 출시한 제품을 알려줘" 라고 물어본다면, 아래와 같이 답장하여야 합니다.
        현재는 찾는 조건에 해당하는 제품이 없습니다. 다른 제품을 검색해주세요.

        이용자가 검색 용도로 LLM을 사용할 수도 있지만 신메뉴에 대한 아이디어를 얻기 위해 사용하는 경우도 존재합니다.
        제품 검색을 위한 사용이 아니라고 판단 될 경우 대화를 나누고 관련 지식을 제공하는데 초점을 맞춰야합니다.
        회사가 육가공품을 제조하는 회사임을 고려하여 답장하여야합니다.

        예를 들어, "발렌타인데이를 맞이해서 신메뉴를 만들고 싶은데 추천해줘" 라고 물어본다면, 
        발렌타인데이는 발렌티노의 축일에서 유래한 기념일로 사랑하는 사람에게 초콜릿을 나눠주는 풍습이 있습니다.
        따라서, 초콜릿과 같은 달달한 간식 메뉴를 추천합니다.
        제품명은 "러블리치즈햄토스트"로 식빵안에 치즈와 햄 그리고 딸기잼을 넣은 토스트입니다.
        채널, 고객사는 편의점 또는 학교로 발렌타인데이를 맞이하여 납품하면 좋은 반응을 얻을 수 있을 것입니다. 

        예를 들어, "다이어트용 제품 어떤게 있어?" 라고 물어본다면,
        다이어트용으로는 닭가슴살이나 닭소세지 같은 닭을 사용한 제품이 인기가 많습니다.
        닭은 저지방 고단백 식품으로 다이어트에 적합하여 냉동 닭가슴살이나 조리된 제품을 쉽게 찾을 수 있으며,
        저지방 소고기의 경우 지방이 적은 부위인 안심이나 등심을 활용할 경우 다이어트에 도움이 될 수 있습니다.
        

        {context}
        
        이건 사용자의 질문입니다. {input}
        """
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            "{context}",
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 메시지 메모리 추가
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    # 최종 체인 완성
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )

    if prompt := st.chat_input("메시지를 입력하세요."):    
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("ai"):
            response = conversational_rag_chain.invoke(
                {"input":prompt},
                config = {"configurable":{"session_id":"abc123"}}
            )      
            st.markdown(response['answer'])

    # if "chat_session" not in st.session_state:    
    #     st.session_state["chat_session"] = model.start_chat(history=[])

    # for content in st.session_state.chat_session.history:
    #     with st.chat_message("ai" if content.role == "model" else "user"):
    #         st.markdown(content.parts[0].text)

    # if prompt := st.chat_input("메시지를 입력하세요."):    
    #     with st.chat_message("user"):
    #         st.markdown(prompt)
    #     with st.chat_message("ai"):
    #         response = st.session_state.chat_session.send_message(prompt)        
    #         st.markdown(response.text)

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