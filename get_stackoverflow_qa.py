import requests
import pandas as pd
import time


def fetch_questions_with_accepted_answers(tag, api_key, pagesize=100):
    questions_data = []
    url = "https://api.stackexchange.com/2.3/questions"
    total_questions = 0
    page = 1
    i = 0
    while i < 4000:
        params = {
            "order": "desc",
            "sort": "votes",
            "site": "stackoverflow",
            "pagesize": pagesize,
            "page": page,
            "tagged": tag,
            "key": api_key,
        }

        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()  # 응답 코드가 200이 아닌 경우 예외 발생
                break
            except requests.exceptions.RequestException as e:
                retries += 1
                if response.status_code == 400:
                    print(
                        f"요청 중 오류 발생: {e}, 페이지 {page}에서 잘못된 요청(400)이 발생했습니다. 페이지를 건너뜁니다."
                    )
                    page += 1
                    i += 1
                    break
                elif response.status_code == 429:
                    print(
                        f"요청 중 오류 발생: {e}, 페이지 {page}에서 요청 초과(429)가 발생했습니다. 반복문을 종료합니다."
                    )
                    return pd.DataFrame(questions_data)  # 요청 초과 시 반복 종료
                else:
                    print(
                        f"요청 중 오류 발생: {e}, 페이지 {page} 재시도 중 ({retries}/{max_retries})..."
                    )
                    time.sleep(3)  # 3초 기다린 후 재시도
                    i += 1

        # 재시도 후에도 실패했으면 다음 페이지로 넘어감
        if retries == max_retries:
            print(f"페이지 {page} 요청 3회 실패. 다음 페이지로 넘어갑니다.")
            page += 1
            continue
        try:
            response_data = response.json()
        except ValueError as e:
            print(f"JSON 디코딩 중 오류 발생: {e}, 페이지 {page}를 건너뜁니다.")
            page += 1
            continue

        questions = response_data.get("items", [])
        has_more = response_data.get("has_more", False)

        # 더 이상 데이터가 없는 경우 반복 종료
        if not questions:
            print(f"페이지 {page}에 더 이상 질문이 없습니다. 반복문을 종료합니다.")
            break

        # accepted_answer_id가 있는 질문만 필터링하여 처리
        filtered_questions = [
            question for question in questions if "accepted_answer_id" in question
        ]
        answer_ids = [question["accepted_answer_id"] for question in filtered_questions]
        total_questions += len(filtered_questions)

        if answer_ids:
            accepted_answers = fetch_answers(answer_ids, api_key, max_retries)
            for question, accepted_answer in zip(filtered_questions, accepted_answers):
                if accepted_answer:
                    questions_data.append(
                        {
                            "question_title": question["title"],
                            "question_link": question["link"],
                            "accepted_answer_body": accepted_answer,
                        }
                    )
                    i += 2

        print(f"페이지 {page} 처리 완료, 총 수집된 질문 수: {total_questions}")

        # 더 이상 페이지가 없는 경우 반복문 종료
        if not has_more:
            print(f"페이지가 더 없습니다. 총 수집된 질문 수: {total_questions}")
            break
        page += 1
        time.sleep(0.5)

    # 수집된 데이터를 데이터프레임으로 반환
    df = pd.DataFrame(questions_data)
    return df


def fetch_answers(answer_ids, api_key, max_retries=3):
    answer_ids_str = ";".join(map(str, answer_ids))
    url = f"https://api.stackexchange.com/2.3/answers/{answer_ids_str}"
    params = {
        "order": "desc",
        "sort": "votes",
        "site": "stackoverflow",
        "filter": "withbody",
        "key": api_key,
    }
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # 응답 코드가 200이 아닌 경우 예외 발생
            break
        except requests.exceptions.RequestException as e:
            retries += 1
            print(
                f"답변 요청 중 오류 발생: {e}, 재시도 중 ({retries}/{max_retries})..."
            )
            time.sleep(3)

    # 재시도 후에도 실패한 경우 None 반환
    if retries == max_retries:
        print(f"답변 요청 3회 실패. 빈 값 반환.")
        return [None] * len(answer_ids)

    try:
        items = response.json().get("items", [])
    except ValueError as e:
        print(f"답변 JSON 디코딩 중 오류 발생: {e}")
        return [None] * len(answer_ids)

    return [item.get("body") for item in items]


api_key = "rl_hJuzJUfpBELxKYUhJZXcTYpEk"  # 여기에 API 키를 입력하세요
tag = "java"  # 원하는 태그를 지정하세요
df = fetch_questions_with_accepted_answers(tag, api_key)

df.to_csv(
    "/content/drive/MyDrive/Colab Notebooks/ai_study/final_project/data/stackoverflow_qa.csv",
    index=False,
)
print("CSV 파일이 성공적으로 생성되었습니다!")
