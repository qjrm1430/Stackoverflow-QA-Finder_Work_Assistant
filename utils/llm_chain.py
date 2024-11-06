from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict


class LLMChain:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        LLM 체인 초기화

        Args:
            model: 사용할 OpenAI 모델명
        """
        self.llm = ChatOpenAI(model=model)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a C# expert. Please provide clear and practical answers to the following questions.
            
            Question: {question}
            
            Good stackoverflow answers:
            {context}
            
            Please keep the following in mind when answering
            1. include code examples if you have them
            2. clearly explain key concepts
            3. mention best practices whenever possible
            4. state any caveats or limitations you may have
            
            All answers should be in Korean.
            """
        )

    def generate_response(self, question: str, similar_results: List[Dict]) -> str:
        """
        LLM 응답 생성

        Args:
            question: 사용자 질문
            similar_results: 유사한 질문/답변 목록
        Returns:
            LLM이 생성한 답변
        """
        # 컨텍스트 생성
        context = "\n\n".join(
            [
                f"질문: {result['question']}\n답변: {result['answer']}"
                for result in similar_results
            ]
        )

        # 프롬프트 생성 및 LLM 호출
        chain = self.prompt | self.llm

        try:
            response = chain.invoke({"question": question, "context": context})
            return response.content
        except Exception as e:
            print(f"LLM 응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
