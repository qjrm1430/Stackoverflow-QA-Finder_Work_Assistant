import re
import pandas as pd
from bs4 import BeautifulSoup
from typing import List


def clean_html(html_text: str) -> str:
    """HTML 텍스트에서 코드와 텍스트를 추출하여 정제된 형태로 반환

    Args:
        html_text: HTML 형식의 텍스트

    Returns:
        정제된 텍스트
    """
    soup = BeautifulSoup(html_text, "html.parser")

    # 결과 텍스트를 저장할 리스트
    content_parts = []

    # HTML 구조를 순회하면서 내용 추출
    for element in soup.children:
        if element.name == "p":
            # 일반 텍스트 처리
            text = element.get_text().strip()
            if text:
                content_parts.append(text)

        elif element.name == "pre":
            # 코드 블록 처리
            code_element = element.find("code")
            if code_element:
                code = code_element.get_text().strip()
                # HTML 태그처럼 보이는 코드인 경우 (실제 소스코드)
                if re.match(r"<\w+.*?>", code) and not re.search(
                    r"<!DOCTYPE|<html|<head|<body", code
                ):
                    content_parts.append(f"```xml\n{code}\n```")
                # 일반 코드인 경우
                else:
                    content_parts.append(f"```csharp\n{code}\n```")

        elif element.name == "code":
            # 인라인 코드 처리
            code = element.get_text().strip()
            if code:
                content_parts.append(f"`{code}`")

    # 결과 조합
    cleaned_text = "\n\n".join(content_parts)

    # 연속된 빈 줄 제거
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text


def load_stackoverflow_data(file_path: str) -> pd.DataFrame:
    """
    스택오버플로우 데이터를 로드하고 전처리

    Args:
        file_path: CSV 파일 경로
    Returns:
        전처리된 DataFrame
    """
    df = pd.read_csv(file_path)

    # 필수 컬럼 확인
    required_columns = ["question_title", "question_link", "accepted_answer_body"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"CSV 파일에 필요한 컬럼이 없습니다. 필요한 컬럼: {required_columns}"
        )

    # HTML 답변 정제
    df["clean_answer"] = df["accepted_answer_body"].apply(clean_html)

    return df[["question_title", "question_link", "clean_answer"]]
