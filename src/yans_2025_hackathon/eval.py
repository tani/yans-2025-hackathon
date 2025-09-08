import re


def normalize_number(number: str | int | float) -> str:
    if isinstance(number, int):
        return str(number)

    if isinstance(number, float):
        if number.is_integer():
            return str(int(number))
        return str(number)

    if not number.strip():
        return ""

    normalized_number = number.replace(",", "").replace(" ", "").strip("0").rstrip(".")
    if normalized_number in {"", "-0"}:
        return "0"
    return normalized_number


def extra_answer_from_response(response: str) -> str:
    """
    モデルの応答から回答部分を抽出する関数。
    応答の最後に出現する数字列を回答とみなす。

    Args:
        response (str): モデルの応答。

    Returns:
        str: 抽出された回答。回答が見つからない場合は空文字。
    """

    matched_answers = re.compile(r"-?[\d.,]+").findall(response)
    if not matched_answers:
        return ""
    return matched_answers[-1]


def evaluate_response(response: str, answer_number: int) -> bool:
    """
    モデルの応答が正しいかどうかを評価する関数。
    応答から回答部分を抽出し、正解と比較する。

    Args:
        response (str): モデルの応答。
        answer_number (int): 正解の数値。

    Returns:
        bool: 応答が正解と一致する場合はTrue、一致しない場合はFalse。
    """
    extracted_answer = extra_answer_from_response(response)
    if not extracted_answer:
        return False
    normalized_extracted = normalize_number(extracted_answer)
    normalized_answer = normalize_number(answer_number)
    return normalized_extracted == normalized_answer
