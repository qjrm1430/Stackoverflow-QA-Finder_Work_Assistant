import os
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from utils.data_loader import load_stackoverflow_data
from utils.evaluation import QAEvaluator
from utils.llm_chain import LLMChain
from utils.vector_store import VectorStore

# 환경 변수 로드
load_dotenv()


@st.cache_resource
def initialize_vector_stores() -> Dict[str, VectorStore]:
    """
    각 언어별 벡터 스토어 초기화 (캐싱)
    """
    embeddings = OpenAIEmbeddings()
    vector_stores = {}
    languages = ["c#", "javascript", "java"]

    for lang in languages:
        vector_store = VectorStore(embeddings)
        index_path = f"faiss_index_{lang}"

        # 저장된 벡터 스토어가 있으면 로드
        if os.path.exists(index_path):
            vector_store.load_vectorstore(index_path)
        else:
            # 데이터 로드 및 벡터 스토어 생성
            df = load_stackoverflow_data(f"data/stackoverflow_{lang}_qa.csv")
            vector_store.create_vectorstore(df)
            vector_store.save_vectorstore(index_path)

        vector_stores[lang] = vector_store

    return vector_stores


def process_question(
    question: str,
    language: str,
    vector_stores: Dict[str, VectorStore],
    llm_chain: LLMChain,
) -> Tuple[List[Dict], str]:
    """
    사용자 질문 처리
    """
    try:
        vector_store = vector_stores[language]
        similar_results = vector_store.get_similar_questions(question)
        llm_response = llm_chain.generate_response(question, similar_results, language)
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

            # 유사도 점수에 따라 수준 결정
            similarity_score = result["similarity_score"]
            if similarity_score < 0.45:
                level = "높음"
            elif 0.45 <= similarity_score < 0.65:
                level = "중간"
            else:
                level = "낮음"

            st.write(f"**유사도 : {level}")  # 유사도 점수와 수준 표시

    st.write("### 💡 AI 답변")
    st.write(llm_answer)


def main():
    st.set_page_config(page_title="프로그래밍 Q&A 챗봇", page_icon="💻", layout="wide")

    # 탭 생성
    tab1, tab2 = st.tabs(["챗봇", "시스템 평가"])

    # 컴포넌트 초기화
    vector_stores = initialize_vector_stores()
    llm_chain = LLMChain()
    evaluator = QAEvaluator()

    with tab1:
        st.title("프로그래밍 Q&A 챗봇 🤖")

        # 사이드바에 설명 추가
        with st.sidebar:
            st.markdown(
                """
                ### 사용 방법
                1. 프로그래밍 언어를 선택하세요
                2. 관련 질문을 입력하세요
                3. 관련된 스택오버플로우 답변들을 검색합니다
                4. AI가 종합적인 답변을 제공합니다
                """
            )

            # 언어 선택
            language = st.selectbox(
                "프로그래밍 언어를 선택하세요:", ["c#", "javascript", "java"]
            )

        # 사용자 입력
        user_question = st.text_input(
            f"{language} 관련 질문을 입력하세요:",
            placeholder=f"예: {language}에서 문자열을 다루는 방법은?",
        )

        if user_question:
            with st.spinner("답변을 찾는 중..."):
                similar_results, llm_response = process_question(
                    user_question, language, vector_stores, llm_chain
                )
                display_results(user_question, similar_results, llm_response)

    with tab2:
        st.title("시스템 평가 📊")
        if st.button("시스템 평가 실행"):
            with st.spinner("시스템 성능을 평가하는 중..."):
                # 테스트 데이터 준비
                test_questions = [
                    "C#에서 문자열을 다루는 방법은?",
                    "LINQ란 무엇인가요?",
                    "async/await의 사용법은?",
                ]

                # 각 질문에 대한 답변과 컨텍스트 수집
                test_answers = []
                test_contexts = []

                for question in test_questions:
                    similar_results, llm_response = process_question(
                        question, "c#", vector_stores, llm_chain
                    )
                    test_answers.append(llm_response)
                    test_contexts.append([r["answer"] for r in similar_results])

                # 평가 실행
                eval_results = evaluator.evaluate_qa_system(
                    test_questions, test_answers, test_contexts
                )

                # 평가 보고서 표시
                st.markdown(evaluator.generate_evaluation_report(eval_results))


if __name__ == "__main__":
    main()
