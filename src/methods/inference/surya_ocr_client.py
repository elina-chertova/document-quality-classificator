import csv
import os
import io
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import requests
from pdf2image import convert_from_path
from PIL import Image


@dataclass
class SuryaOCRConfig:
    base_url: str = "http://51.250.28.28:2214"
    recognize_endpoint: str = "/recognize-text"
    raw: bool = False  # query param
    timeout_sec: int = 120
    max_retries: int = 3
    retry_backoff_sec: float = 1.0


class SuryaOCRClient:
    def __init__(self, cfg: SuryaOCRConfig = SuryaOCRConfig()):
        self.cfg = cfg

    def recognize_file(self, file_path: str, raw: Optional[bool] = None) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + self.cfg.recognize_endpoint
        params = {"raw": self.cfg.raw if raw is None else raw}

        headers = {"Accept": "application/json"}

        def _send_file(tuple_data: Tuple[str, bytes, str]):
            files = {"file": tuple_data}
            return requests.post(url, params=params, files=files, headers=headers, timeout=self.cfg.timeout_sec)

        filename = os.path.basename(file_path)
        lower = filename.lower()

        if lower.endswith(".pdf"):
            try:
                pages = convert_from_path(file_path, dpi=200)
            except Exception as e:
                raise RuntimeError(f"Failed to render PDF {filename}: {e}")
            texts: List[str] = []
            last_err: Optional[Exception] = None
            for idx, page in enumerate(pages):
                buf = io.BytesIO()
                page.save(buf, format="JPEG", quality=92)
                jpeg_bytes = buf.getvalue()
                part_name = f"{os.path.splitext(filename)[0]}_p{idx+1}.jpg"
                for attempt in range(1, self.cfg.max_retries + 1):
                    try:
                        resp = _send_file((part_name, jpeg_bytes, 'image/jpeg'))
                        if resp.status_code >= 500:
                            raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {resp.text}")
                        resp.raise_for_status()
                        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"text": resp.text}
                        texts.append(data.get("text", ""))
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        if attempt < self.cfg.max_retries:
                            time.sleep(self.cfg.retry_backoff_sec * attempt)
                        else:
                            break
            if last_err:
                raise RuntimeError(f"OCR failed for {filename}: {last_err}")
            return {"text": "\f".join(texts)}

        # For images and other files, send as-is with retries
        mime = "image/jpeg" if lower.endswith(('.jpg', '.jpeg')) else (
            'image/png' if lower.endswith('.png') else 'application/octet-stream'
        )
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                with open(file_path, "rb") as f:
                    resp = _send_file((filename, f.read(), mime))
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {resp.text}")
                resp.raise_for_status()
                return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"text": resp.text}
            except Exception as e:
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec * attempt)
                else:
                    break
        raise RuntimeError(f"OCR request failed for {filename}: {last_err}")


def process_folder_with_ocr(input_dir: str, output_dir: str, csv_path: str, client: SuryaOCRClient) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    rows: List[List[Any]] = []
    headers = ["filename", "status", "error", "text_len", "raw", "ref_len", "cer"]
    files = [n for n in os.listdir(input_dir) if n.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    processed = success = failed = 0
    for name in files:
        processed += 1
        src = os.path.join(input_dir, name)
        try:
            data = client.recognize_file(src)
            text = data.get("text", "") if isinstance(data, dict) else ""
            # save txt
            txt_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}.txt")
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write(text)
            # CER: ищем референс рядом (optional: refs_dir same as input_dir/<stem>.ref.txt or input_dir/<stem>.txt)
            ref_len = ""
            cer_val = ""
            for candidate in (
                os.path.join(input_dir, f"{os.path.splitext(name)[0]}.txt"),
                os.path.join(input_dir, f"{os.path.splitext(name)[0]}.ref.txt"),
            ):
                if os.path.isfile(candidate):
                    with open(candidate, "r", encoding="utf-8", errors="ignore") as rf:
                        ref = rf.read()
                    ref_len = len(ref)
                    cer_val = _compute_cer(text, ref)
                    break
            rows.append([name, "ok", "", len(text), bool(data.get("raw")) if isinstance(data, dict) else False, ref_len, cer_val])
            success += 1
            print(f"[OCR] OK {name} → {txt_path} ({len(text)} chars)")
        except Exception as e:
            rows.append([name, "error", str(e), 0, "", "", ""])
            failed += 1
            print(f"[OCR] FAILED {name}: {e}")
    # write csv
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return {"processed": processed, "success": success, "failed": failed, "csv": csv_path, "output": output_dir}


def _compute_cer(hyp: str, ref: str) -> float:
    """Character Error Rate = Levenshtein(hyp, ref) / len(ref). Returns float rounded to 6 decimals."""
    if ref is None or len(ref) == 0:
        return 0.0
    # DP Levenshtein distance by characters
    h = list(hyp or "")
    r = list(ref)
    n, m = len(r), len(h)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        ri = r[i-1]
        for j in range(1, m+1):
            cost = 0 if ri == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    cer = dp[n][m] / float(n)
    return round(cer, 6)


