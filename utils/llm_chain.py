from typing import Dict, List

from langchain.callbacks import LangChainTracer
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client


class LLMChain:
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        LLM 체인 초기화

        Args:
            model: 사용할 OpenAI 모델명
        """
        # Langsmith 트레이서 설정 - 예외 처리 추가
        try:
            tracer = LangChainTracer(project_name="stackoverflow_qa_project")
            callbacks = [tracer]
        except Exception as e:
            print(f"Langsmith 트레이서 초기화 실패: {str(e)}")
            callbacks = []

        self.llm = ChatOpenAI(
            model=model,
            callbacks=callbacks,
            tags=["stackoverflow_qa"],
            metadata={"model": model, "type": "qa_system"},
        )

        self.prompts = {
            "c#": """
            You are a C# expert. Please provide clear and practical answers to the following questions.
            
            Question: {question}
            
            Good stackoverflow answers:
            {context}
            
            Please keep the following in mind when answering:
            1. Focus on C# best practices and modern approaches
            2. Include code examples using C# syntax
            3. Mention any .NET specific considerations
            4. All answers should be in Korean
            """,
            "javascript": """
            You are a JavaScript expert. Please provide clear and practical answers to the following questions.
            
            Question: {question}
            
            Good stackoverflow answers:
            {context}
            
            Please keep the following in mind when answering:
            1. Focus on modern JavaScript features and best practices
            2. Include code examples using JavaScript syntax
            3. Mention browser compatibility when relevant
            4. All answers should be in Korean
            """,
            "java": """
            You are a Java expert. Please provide clear and practical answers to the following questions.
            
            Question: {question}
            
            Good stackoverflow answers:
            {context}
            
            Please keep the following in mind when answering:
            1. Focus on Java best practices and modern approaches
            2. Include code examples using Java syntax
            3. Mention JVM considerations when relevant
            4. All answers should be in Korean
            """,
        }

    def generate_response(
        self, question: str, similar_results: List[Dict], language: str
    ) -> str:
        """
        LLM 응답 생성

        Args:
            question: 사용자 질문
            similar_results: 유사한 질문/답변 목록
            language: 프로그래밍 언어
        Returns:
            LLM이 생성한 답변
        """
        try:
            # 컨텍스트 생성
            context = "\n\n".join(
                [
                    f"질문: {result['question']}\n답변: {result['answer']}"
                    for result in similar_results
                ]
            )

            # 언어별 프롬프트 선택
            prompt = ChatPromptTemplate.from_template(self.prompts[language])

            # 프롬프트 생성 및 LLM 호출
            chain = prompt | self.llm

            response = chain.invoke({"question": question, "context": context})

            return response.content

        except Exception as e:
            print(f"LLM 응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
