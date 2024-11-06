import streamlit as st
from typing import Tuple, List, Dict
from utils.vector_store import VectorStore
from utils.llm_chain import LLMChain
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from utils.data_loader import load_stackoverflow_data

# 환경 변수 로드
load_dotenv()


@st.cache_resource
def initialize_vector_store() -> VectorStore:
    """
    벡터 스토어 초기화 (캐싱)
    """
    embeddings = OpenAIEmbeddings()
    vector_store = VectorStore(embeddings)

    # 저장된 벡터 스토어가 있으면 로드
    if os.path.exists("faiss_index"):
        vector_store.load_vectorstore("faiss_index")
    else:
        # 데이터 로드 및 벡터 스토어 생성
        df = load_stackoverflow_data("data/stackoverflow_qa.csv")
        vector_store.create_vectorstore(df)
        vector_store.save_vectorstore("faiss_index")

    return vector_store


def process_question(
    question: str, vector_store: VectorStore, llm_chain: LLMChain
) -> Tuple[List[Dict], str]:
    """
    사용자 질문 처리
    """
    try:
        similar_results = vector_store.get_similar_questions(question)
        llm_response = llm_chain.generate_response(question, similar_results)
        return similar_results, llm_response
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        return [], "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."


def display_results(user_question: str, similar_results: List[Dict], llm_answer: str):
    """
    검색 결과 표시
    """
    st.write("### 📝 사용자 질문")
    st.write(user_question)

    st.write("### 🔍 유사한 스택오버플로우 Q&A")
    for idx, result in enumerate(similar_results, 1):
        with st.expander(f"참고 자료 {idx}: {result['question']}", expanded=True):
            st.write(f"**답변:** {result['answer']}")
            st.write(f"**원본 링크:** [스택오버플로우에서 보기]({result['link']})")

    st.write("### 💡 AI 답변")
    st.write(llm_answer)


def main():
    st.set_page_config(page_title="C# Q&A 챗봇", page_icon="💻", layout="wide")
    st.title("C# Q&A 챗봇 🤖")

    # 컴포넌트 초기화
    vector_store = initialize_vector_store()
    llm_chain = LLMChain()

    # 사이드바에 설명 추가
    with st.sidebar:
        st.markdown(
            """
        ### 사용 방법
        1. C# 관련 질문을 입력하세요
        2. 관련된 스택오버플로우 답변들을 검색합니다
        3. AI가 종합적인 답변을 제공합니다
        """
        )

    # 사용자 입력
    user_question = st.text_input(
        "C# 관련 질문을 입력하세요:",
        placeholder="예: C#에서 string과 String의 차이점은 무엇인가요?",
    )

    if user_question:
        with st.spinner("답변을 찾는 중..."):
            similar_results, llm_response = process_question(
                user_question, vector_store, llm_chain
            )
            display_results(user_question, similar_results, llm_response)


if __name__ == "__main__":
    main()
