import csv
import os
from typing import Dict, List, Tuple


def compute_cer(hyp: str, ref: str) -> float:
    if ref is None or len(ref) == 0:
        return 0.0
    h = list(hyp or "")
    r = list(ref)
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = r[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return round(dp[n][m] / float(n), 6)


def cer_for_folders(hyp_dir: str, ref_dir: str, csv_out: str) -> Dict[str, float]:
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    files = [f for f in os.listdir(hyp_dir) if f.lower().endswith(".txt")]
    rows: List[List[object]] = [["filename", "hyp_len", "ref_len", "cer"]]
    total_chars = 0
    total_edits = 0
    per_file: Dict[str, float] = {}
    for name in files:
        hyp_path = os.path.join(hyp_dir, name)
        ref_path = os.path.join(ref_dir, name)
        if not os.path.isfile(ref_path):
            continue
        with open(hyp_path, "r", encoding="utf-8", errors="ignore") as hf:
            hyp = hf.read()
        with open(ref_path, "r", encoding="utf-8", errors="ignore") as rf:
            ref = rf.read()
        cer = compute_cer(hyp, ref)
        rows.append([name, len(hyp), len(ref), cer])
        per_file[name] = cer
        total_chars += len(ref)

        total_edits += cer * len(ref)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    macro = round((total_edits / total_chars) if total_chars else 0.0, 6)
    per_file["__macro__"] = macro
    return per_file


