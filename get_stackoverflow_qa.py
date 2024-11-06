import requests
import pandas as pd
import time


def fetch_questions_with_accepted_answers(tag, api_key, pagesize=100):
    questions_data = []
    url = "https://api.stackexchange.com/2.3/questions"
    page = 1
    total_questions = 0
    answer_ids = []

    while True:
        params = {
            "order": "desc",
            "sort": "votes",
            "site": "stackoverflow",
            "pagesize": pagesize,
            "page": page,
            "tagged": tag,
            "key": api_key,
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # 응답 코드가 200이 아닌 경우 예외를 발생시킴
        except requests.exceptions.RequestException as e:
            print(f"요청 중 오류 발생: {e}, 페이지 {page} 재시도 중...")
            time.sleep(5)  # 5초 기다린 후 재시도
            continue

        response_data = response.json()
        questions = response_data.get("items", [])
        has_more = response_data.get("has_more", False)

        for question in questions:
            if "accepted_answer_id" in question:
                answer_ids.append(question["accepted_answer_id"])
                total_questions += 1

        print(f"페이지 {page} 처리 완료, 총 수집된 질문 수: {total_questions}")

        if answer_ids:
            accepted_answers = fetch_answers(answer_ids, api_key)
            for question, accepted_answer in zip(questions, accepted_answers):
                if accepted_answer:
                    questions_data.append(
                        {
                            "question_title": question["title"],
                            "question_link": question["link"],
                            "accepted_answer_body": accepted_answer,
                        }
                    )

        answer_ids = []  # 다음 페이지를 위해 answer_ids 초기화
        page += 1

        if not has_more:
            print(f"총 수집된 질문 수: {total_questions}")
            break

    df = pd.DataFrame(questions_data)
    return df


def fetch_answers(answer_ids, api_key):
    answer_ids_str = ";".join(map(str, answer_ids))
    url = f"https://api.stackexchange.com/2.3/answers/{answer_ids_str}"
    params = {
        "order": "desc",
        "sort": "votes",
        "site": "stackoverflow",
        "filter": "withbody",
        "key": api_key,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # 응답 코드가 200이 아닌 경우 예외를 발생시킴
    except requests.exceptions.RequestException as e:
        print(f"답변 요청 중 오류 발생: {e}")
        return [None] * len(answer_ids)

    items = response.json().get("items", [])
    return [item.get("body") for item in items]


api_key = "rl_hJuzJUfpBELxKYUhJZXcTYpEk"  # 여기에 API 키를 입력하세요
tag = "c#"  # 원하는 태그를 지정하세요
df = fetch_questions_with_accepted_answers(tag, api_key)

df.to_csv(
    "/content/drive/MyDrive/Colab Notebooks/ai_study/final_project/data/stackoverflow_qa.csv",
    index=False,
)
print("CSV 파일이 성공적으로 생성되었습니다!")
