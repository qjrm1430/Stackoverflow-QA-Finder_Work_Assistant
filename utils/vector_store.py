from typing import Dict, List

import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    def __init__(self, embeddings: OpenAIEmbeddings):
        """
        벡터 스토어 초기화

        Args:
            embeddings: OpenAI 임베딩 모델
        """
        self.embeddings = embeddings
        self.vectorstore = None

    def create_vectorstore(self, df: pd.DataFrame) -> FAISS:
        """
        DataFrame으로부터 FAISS 벡터 스토어 생성

        Args:
            df: 전처리된 DataFrame (question_title, question_link, clean_answer 포함)
        Returns:
            FAISS 벡터 스토어
        """
        documents = []

        for idx, row in df.iterrows():
            # 검색을 위한 텍스트는 질문과 답변을 결합
            text_content = f"{row['question_title']}\n{row['clean_answer']}"

            # 메타데이터에 모든 필요한 정보 저장
            metadata = {
                "question_title": row["question_title"],
                "clean_answer": row["clean_answer"],
                "question_link": row["question_link"],
            }

            doc = Document(page_content=text_content, metadata=metadata)
            documents.append(doc)

        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self.vectorstore

    def get_similar_questions(self, query: str, k: int = 3) -> List[Dict]:
        """
        유사한 질문 검색 및 결과 포맷팅

        Args:
            query: 사용자 질문
            k: 반환할 결과 수
        Returns:
            유사 질문, 답변, 링크, 유사도 점수를 포함한 결과 리스트
        """
        if not self.vectorstore:
            raise ValueError("Vector store가 초기화되지 않았습니다.")

        # 유사도 검색 수행
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        results = []

        for doc, score in docs_with_scores:
            results.append(
                {
                    "question": doc.metadata["question_title"],
                    "answer": doc.metadata["clean_answer"],
                    "link": doc.metadata["question_link"],
                    "similarity_score": score,  # 유사도 점수 추가
                }
            )

        return results

    def save_vectorstore(self, path: str):
        """벡터 스토어를 로컬에 저장"""
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str) -> FAISS:
        """로컬에서 벡터 스토어 로드"""
        self.vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True,  # 신뢰할 수 있는 로컬 데이터에 대해서만 사용
        )
        return self.vectorstore
