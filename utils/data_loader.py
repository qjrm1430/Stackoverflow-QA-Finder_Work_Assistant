import re
from typing import List

import pandas as pd
from bs4 import BeautifulSoup


def extract_code_blocks(element) -> List[str]:
    """코드 블록을 추출하고 포맷팅하는 함수

    Args:
        element: BeautifulSoup element
    Returns:
        포맷팅된 코드 블록 리스트
    """
    code_blocks = []

    # pre 태그 내의 코드 처리
    if element.name == "pre":
        code_element = element.find("code")
        if code_element:
            code = code_element.get_text(strip=True)
            lang = (
                code_element.get("class", [""])[0] if code_element.get("class") else ""
            )

            # 언어 지정이 없는 경우 내용을 기반으로 추측
            if not lang:
                # C# 패턴
                if re.search(
                    r"class\s+\w+|public\s+\w+|private\s+\w+|namespace|using\s+\w+|Console\.|string\[\]|int\[\]|List<|Dictionary<",
                    code,
                ):
                    lang = "csharp"

                # Java 패턴
                elif re.search(
                    r"public\s+class|private\s+class|protected\s+class|import\s+java\.|System\.out\.|String\[\]|ArrayList<|HashMap<",
                    code,
                ):
                    lang = "java"

                # JavaScript 패턴
                elif re.search(
                    r"function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+|document\.|window\.|=>|Promise<|async|await",
                    code,
                ):
                    lang = "javascript"

                # Python 패턴
                elif re.search(
                    r"def\s+\w+|class\s+\w+:|import\s+\w+|from\s+\w+\s+import|print\(|if\s+__name__\s*==\s*['\"]__main__['\"]",
                    code,
                ):
                    lang = "python"

                # SQL 패턴
                elif re.search(
                    r"SELECT|INSERT|UPDATE|DELETE|CREATE TABLE|ALTER TABLE|DROP TABLE|JOIN|WHERE|GROUP BY",
                    code,
                    re.IGNORECASE,
                ):
                    lang = "sql"

                # HTML 패턴
                elif re.search(r"<html|<body|<div|<p|<script|<style", code):
                    lang = "html"

                # XML 패턴
                elif re.match(r"<\?xml|<\w+.*?>", code):
                    lang = "xml"

                # CSS 패턴
                elif re.search(r"{[\s\S]*?}|@media|@keyframes|#\w+|\.\w+", code):
                    lang = "css"

                # PowerShell 패턴
                elif re.search(r"\$\w+|Get-|Set-|New-|Remove-", code):
                    lang = "powershell"

                # Bash/Shell 패턴
                elif re.search(r"#!/bin/|echo|grep|sed|awk|\$\(\w+\)", code):
                    lang = "bash"

                # 기본값
                else:
                    lang = "text"

            code_blocks.append(f"```{lang}\n{code}\n```")

    # 인라인 코드 처리
    elif element.name == "code":
        code = element.get_text(strip=True)
        if code:
            code_blocks.append(f"`{code}`")

    # LaTeX 수식 처리
    elif element.name == "math" or element.get("class", [""])[0] == "math":
        math_text = element.get_text(strip=True)
        if math_text:
            # 인라인 수식
            if element.get("display") != "block":
                code_blocks.append(f"${math_text}$")
            # 디스플레이 수식
            else:
                code_blocks.append(f"$${math_text}$$")

    return code_blocks


def clean_html(html_text: str) -> str:
    """HTML 텍스트에서 코드와 텍스트를 추출하여 정제된 형태로 반환

    Args:
        html_text: HTML 형식의 텍스트

    Returns:
        정제된 텍스트
    """
    if not html_text or not isinstance(html_text, str):
        return ""

    soup = BeautifulSoup(html_text, "html.parser")
    content_parts = []

    # 링크 텍스트 처리
    for a in soup.find_all("a"):
        if a.get("href"):
            a.replace_with(f"{a.get_text()} ({a['href']})")

    # MathJax/LaTeX 수식을 위한 특수 처리
    for math in soup.find_all(["script", "span"], {"type": "math/tex"}):
        if math.get("type") == "math/tex":
            if math.get("display") == "block":
                math.replace_with(f"$${math.string}$$")
            else:
                math.replace_with(f"${math.string}$")

    # 순차적으로 모든 요소 처리
    for element in soup.children:
        if not hasattr(element, "name"):  # NavigableString 처리
            continue

        if element.name == "p":
            text = element.get_text(strip=True)
            if text:
                content_parts.append(text)

        elif element.name in ["pre", "code", "math"]:
            code_blocks = extract_code_blocks(element)
            content_parts.extend(code_blocks)

        elif element.name == "ul" or element.name == "ol":
            for li in element.find_all("li", recursive=False):
                li_text = li.get_text(strip=True)
                if li_text:
                    content_parts.append(f"• {li_text}")

                # 리스트 아이템 내의 코드 블록 처리
                for code_element in li.find_all(["pre", "code", "math"]):
                    code_blocks = extract_code_blocks(code_element)
                    content_parts.extend(code_blocks)

        elif element.name == "blockquote":
            quote_text = element.get_text(strip=True)
            if quote_text:
                content_parts.append(f"> {quote_text}")

        elif element.name == "hr":
            content_parts.append("---")

    # 결과 조합
    cleaned_text = "\n\n".join(content_parts)

    # 연속된 빈 줄 및 불필요한 공백 제거
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)

    return cleaned_text.strip()


def load_stackoverflow_data(file_path: str) -> pd.DataFrame:
    """
    스택오버플로우 데이터를 로드하고 전처리

    Args:
        file_path: CSV 파일 경로
    Returns:
        전처리된 DataFrame
    """
    try:
        df = pd.read_csv(file_path)

        # 필수 컬럼 확인
        required_columns = ["question_title", "question_link", "accepted_answer_body"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"CSV 파일에 다음 필수 컬럼이 없습니다: {missing_columns}")

        # 결측치 처리
        df = df.dropna(subset=["accepted_answer_body"])

        # HTML 답변 정제
        df["clean_answer"] = df["accepted_answer_body"].apply(clean_html)

        # 빈 답변 제거
        df = df[df["clean_answer"].str.strip() != ""]

        return df[["question_title", "question_link", "clean_answer"]]

    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {str(e)}")
        raise
