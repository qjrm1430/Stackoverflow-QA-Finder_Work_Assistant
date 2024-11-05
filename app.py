from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from utils.data_processor import DataProcessor
from utils.vector_store import VectorStore

# 환경변수 로드
load_dotenv()


class QABot:
    """C# QA 챗봇 클래스"""

    def __init__(self):
        """벡터 저장소와 LLM 초기화"""
        self.vector_store = VectorStore()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 벡터 저장소가 없는 경우에만 새로 생성
        if self.vector_store.vector_store is None:
            print("벡터 저장소가 없어 새로 생성합니다.")
            qa_data = DataProcessor.load_and_process_data("data/stackoverflow_qa.csv")
            self.vector_store.create_vector_store(qa_data)
        else:
            print("기존 벡터 저장소를 사용합니다.")

        # 프롬프트 템플릿 수정
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 C# 프로그래밍 전문가입니다. 주어진 질문에 대해 정확하고 자세한 답변을 제공해주세요.
                다음 규칙을 반드시 따라주세요:
                1. 코드는 한 번만 표시하고 중복해서 보여주지 마세요.
                2. 답변은 간단명료하게 작성해주세요.
                3. 코드 블록은 한 번만 포함시키고, 설명은 코드 앞에 작성해주세요.
                4. 불필요한 반복을 피하고 핵심 내용만 전달해주세요.""",
                ),
                (
                    "user",
                    "질문: {question}\n\n참고할 만한 유사한 질문/답변들:\n{similar_qa}",
                ),
            ]
        )

    def get_llm_response(self, question: str, similar_qa: List[Dict]) -> str:
        """LLM을 사용하여 답변 생성"""
        # 유사 QA 텍스트 형식 수정
        similar_qa_text = "\n\n".join(
            [
                f"참고 {i+1}:\n질문: {qa['question']}\n답변: {qa['answer'].strip()}"
                for i, qa in enumerate(similar_qa)
            ]
        )

        chain = self.prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": question, "similar_qa": similar_qa_text})

        return response


def initialize_qa_bot():
    """QA 봇 초기화 함수"""
    with st.spinner("챗봇을 초기화하고 있습니다..."):
        try:
            return QABot()
        except Exception as e:
            st.error(f"초기화 중 오류가 발생했습니다: {str(e)}")
            return None


def main():
    st.title("C# 스택오버플로우 QA 챗봇")

    # 세션 상태 초기화
    if "qa_bot" not in st.session_state:
        qa_bot = initialize_qa_bot()
        if qa_bot is None:
            return
        st.session_state.qa_bot = qa_bot

    # 사용자 입력
    question = st.text_area("C# 관련 질문을 입력하세요:")

    if st.button("답변 받기"):
        if not question.strip():
            st.warning("질문을 입력해주세요.")
            return

        try:
            with st.spinner("답변을 생성하고 있습니다..."):
                # 진행 상태 표시
                progress_bar = st.progress(0)

                # 유사한 QA 쌍 검색
                progress_bar.progress(30)
                similar_qa = st.session_state.qa_bot.vector_store.find_similar_qa(
                    question
                )

                if not similar_qa:
                    st.warning("유사한 질문을 찾을 수 없습니다.")
                    return

                progress_bar.progress(60)

                # LLM 답변 생성
                llm_response = st.session_state.qa_bot.get_llm_response(
                    question, similar_qa
                )
                progress_bar.progress(100)

                # 결과 표시
                st.write("### 입력한 질문:")
                st.write(question)

                st.write("### 유사한 질문/답변:")
                for i, qa in enumerate(similar_qa, 1):
                    with st.expander(f"유사 질문/답변 {i}"):
                        st.write(f"**질문:** {qa['question']}")
                        st.write(f"**답변:** {qa['answer']}")
                        st.write(f"**유사도 점수:** {qa['similarity_score']:.4f}")

                st.write("### LLM 답변:")
                st.write(llm_response)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            st.button("다시 시도")


if __name__ == "__main__":
    main()
