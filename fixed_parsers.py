import re

def modified_tourn_id(line: str) -> int:
    match = re.search(r"Hand #\d+: Tournament #(\d+)", line)
    return int(match.group(1)) if match else 0

def modified_get_board(line: str) -> str:
    if "*** TURN ***" in line:
        match = re.search(r"\[.*?\] \[(.*?)\]", line)
        return match.group(1) if match else "-"
    else:
        match = re.search(r"\[(.*?)\]", line)
        return match.group(1) if match else ("0" if "*** SUMMARY ***" in line else "-")

def modified_get_combination(line: str) -> str:
    match = re.search(r"\(([^)]+)\)", line)
    return match.group(1) if match else ""
