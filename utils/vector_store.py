import os
from typing import Dict, List

import streamlit as st
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .data_processor import QAData


@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-ada-002")


class VectorStore:
    """FAISS 벡터 저장소 관리 클래스"""

    VECTOR_STORE_PATH = "vector_store"  # 벡터 저장소 저장 경로

    def __init__(self):
        """OpenAI 임베딩 모델과 FAISS 벡터 저장소 초기화"""
        self.embeddings = get_embeddings()
        self.vector_store = None
        self._load_or_create_store()

    def _load_or_create_store(self) -> None:
        """벡터 저장소를 로드하거나 없는 경우 생성"""
        if os.path.exists(self.VECTOR_STORE_PATH):
            try:
                print("기존 벡터 저장소를 로드합니다.")
                self.vector_store = FAISS.load_local(
                    self.VECTOR_STORE_PATH, self.embeddings
                )
            except Exception as e:
                print(f"벡터 저장소 로드 중 오류 발생: {str(e)}")
                self.vector_store = None

    def create_vector_store(self, qa_data: List[QAData]) -> None:
        """QA 데이터로부터 벡터 저장소 생성 및 저장

        Args:
            qa_data: QAData 객체 리스트
        """
        # 이미 벡터 저장소가 있으면 생성하지 않음
        if self.vector_store is not None:
            print("벡터 저장소가 이미 존재합니다.")
            return

        print("새로운 벡터 저장소를 생성합니다.")
        documents = [
            Document(
                page_content=f"질문: {qa.question}\n답변: {qa.answer}",
                metadata={"question": qa.question, "answer": qa.answer},
            )
            for qa in qa_data
        ]

        self.vector_store = FAISS.from_documents(
            documents=documents, embedding=self.embeddings
        )

        # 벡터 저장소를 로컬에 저장
        self.save_vector_store()

    def save_vector_store(self) -> None:
        """벡터 저장소를 로컬에 저장"""
        if self.vector_store:
            os.makedirs(self.VECTOR_STORE_PATH, exist_ok=True)
            self.vector_store.save_local(self.VECTOR_STORE_PATH)
            print("벡터 저장소가 저장되었습니다.")

    def find_similar_qa(self, query: str, k: int = 3) -> List[Dict]:
        """쿼리와 유사한 QA 쌍을 검색

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수

        Returns:
            유사한 QA 쌍 리스트
        """
        if not self.vector_store:
            raise ValueError("Vector store has not been initialized")

        results = self.vector_store.similarity_search_with_score(
            query, k=k, score_threshold=0.8  # 유사도 임계값 추가
        )

        # 중복 제거를 위한 딕셔너리
        unique_results = {}
        for doc, score in results:
            question = doc.metadata["question"]
            if question not in unique_results:
                unique_results[question] = {
                    "question": question,
                    "answer": doc.metadata["answer"],
                    "similarity_score": score,
                }

        return list(unique_results.values())[:k]
