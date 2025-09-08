import re


def normalize_number(number: str | int) -> str:
    """Normalize a numeric representation to a canonical string.

    - Removes spaces/commas
    - Handles integers/decimals
    - Removes leading zeros for integer part
    - Trims trailing zeros in the fractional part and drops trailing dot
    - Keeps minus sign only when non-zero
    """
    if isinstance(number, int):
        return str(number)

    s = str(number).strip()
    if not s:
        return ""

    # Remove commas and spaces
    s = s.replace(",", "").replace(" ", "")

    neg = s.startswith("-")
    if neg:
        s = s[1:]

    if "." in s:
        int_part, frac_part = s.split(".", 1)
        int_part = int_part.lstrip("0") or "0"
        frac_part = frac_part.rstrip("0")
        if frac_part:
            out = f"{int_part}.{frac_part}"
        else:
            out = int_part
    else:
        out = s.lstrip("0") or "0"

    if neg and out != "0":
        out = "-" + out
    return out


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
    answer = matched_answers[-1]
    if answer.endswith("."):
        answer = answer[:-1]
    return answer


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
