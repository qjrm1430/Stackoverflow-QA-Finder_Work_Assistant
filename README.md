# Stackoverflow-QA-Finder Work Assistant
  - [개요](#개요)
  - [기술 스택](#기술-스택)
  - [실험](#실험)
    - [문제 정의](#문제-정의)
    - [가설](#가설)
    - [검증](#검증)
    - [결과](#결과)
  - [파이프라인](#파이프라인)
  - [What Technologies were used](#What-Technologies-were-used)

## 개요
- StackOverflow에서 프로그래밍 언어와 관련된 질문을 받으면 해당 언어에 대한 Agent가 벡터 저장소(FAISS)에서 StackOverflow에 채택되어있는 유사성 있는 질문을 찾은 다음 해당 질문에 대한 답변에 대한 Re-Ranking을 진행하여 LLM에게 프롬프트를 적절하게 반환하게 되고 해당 프롬프트를 활용하여 LLM은 사용자에게 관련 질문에 대한 답변을 제공한다.

## 기술 스택

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Langchain](https://img.shields.io/badge/Langchain-00C7B7?logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-343541?logo=openai&logoColor=white)

## 실험

### 문제 정의
- 한국어 미지원 문제

![image](https://github.com/user-attachments/assets/ac63403d-10a3-49fb-873c-8549ffc87d51)

- 어떤 답변을 봐야 하는가?

![image](https://github.com/user-attachments/assets/2c78cb62-3ed6-433a-86af-425efbd3bf91)

- 너무 많은 태그 목록 -> 정보의 불확실성을 초례

![image](https://github.com/user-attachments/assets/9bd6ec1e-39c0-45d7-90f0-b4ea4820a662)
![image](https://github.com/user-attachments/assets/533fe99d-b5af-4752-b430-64fabe2bdbc5)

### 가설
- RAG LLM을 통한 태그 맞춤형 Agent를 제작하여 응답을 반환한다면 번역 문제를 해결할 수 있고 정보의 확실성을 보장할 수 있다.
- 이때 아래의 이유로 인한 Advanced RAG가 필요하다고 판단되어짐.
  - 현업의 사용자가 실제 업무중에 Work Assistant로 활용되어져야 한다.
  - 오류를 최소화하여 정보의 확실성을 높혀야 한다.

### 검증
- context_recall 부분에서 1.0 스코어를 유지
  
![image](https://github.com/user-attachments/assets/2f6aa9d8-b0dd-478b-92e5-d211bd6aed8d)

- 검증 평가를 프롬프트에 추가시키면 성능 개선이 보여짐
![image](https://github.com/user-attachments/assets/7491ee5a-e64d-48e4-98d2-2cccd6ec4197)


### 결과

## 파이프라인
![stack drawio](https://github.com/user-attachments/assets/c131ae91-aba5-41f0-a01f-7b5534ac6a0a)

## What-Technologies-were-used
![image](https://github.com/user-attachments/assets/f859e839-c8f7-41ad-b852-dcfaa1517cb9)

### Pre-Retriever (HyDE)
- 사용자 query를 Vector DB와 같은 지식 검색소 검색 전, LLM에 주입하여 가상의 답변을 얻고,query 대신 가상 답변으로 검색하여 Context를 얻는 기술
- 가상 답변은 query 대비, 풍부한 정보를 담고 있어 지식 검색 시, 더 많은 정보를 담은 context를 얻을 수 있다.
- 해당 프로젝트에서도 LLM에 주입하여 답변을 생성시킨 다음 가상 답변으로 검색하여 Context를 얻는다.

### Post-Retriever (Re-Ranking)
- 검색된 정보를 재순위하여 가장 관련성 높은 답변을 우선시하는 것.
- 원래는 검색된 청크를 재정렬하고 Top-K 가장 관련성 높은 청크를 식별하여 LLM에 사용할 컨텍스트로 제공한다.
- 해당 프로젝트에서는 청크가 아닌 여러개의 질문(청크)을 뽑은 후 가장 유사도가 높은 질문 순으로 식별하여 LLM에게 컨텍스트로 제공한다.

## Re-Traing
- RAG 평가지표를 바탕으로 프롬프트를 증강시키는 방식을 고려
- 실험-검증 과정에서 보이듯 일부 결과에서 성능 개선이 이루어지는 것을 일부 확인
