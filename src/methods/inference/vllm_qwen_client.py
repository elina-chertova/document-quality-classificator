import base64
import io
import os
from dataclasses import dataclass
from typing import List, Dict, Any

from openai import OpenAI
from pdf2image import convert_from_path

try:
    # optional: load .env if present
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

@dataclass
class VLLMQwenConfig:
    base_url: str = os.getenv("VLLM_BASE_URL", "https://51.250.28.28:10000/experimental/v1")
    api_key: str = os.getenv("VLLM_API_KEY", "EMPTY")

    model: str = "unsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit"
    dpi: int = 300
    max_side: int = 1600
    temperature: float = 0.0


PROMPT = (
    "Извлеки весь текст из документа, включая таблицы и текстовые блоки.\n"
    "Извлекай каждый фрагмент текста ровно один раз. Не исправляй форматирование (пробелы, переносы строк) и не изменяй расположение текста и не изменяй сам текст.\n"
    "Не исправляй никакие слова, пиши все так, как написано в документе. Не повторяй текст, который встречается единожды.\n"
    "Ответ верни строго в виде обычного текста, без добавления комментариев или инструкций. Не меняй текст или грамотность."
)


class VLLMQwenClient:
    def __init__(self, cfg: VLLMQwenConfig = VLLMQwenConfig()):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def _pdf_first_page_to_base64(self, pdf_path: str) -> str:
        page = convert_from_path(pdf_path, dpi=self.cfg.dpi)[0]
        page.thumbnail((self.cfg.max_side, self.cfg.max_side))
        buf = io.BytesIO()
        page.save(buf, format="JPEG")
        img_bytes = buf.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    def ocr_pdf(self, pdf_path: str) -> str:
        img_b64 = self._pdf_first_page_to_base64(pdf_path)
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                }],
                temperature=self.cfg.temperature,
            )
        except Exception as e:
            raise RuntimeError(f"VLLM request failed for {os.path.basename(pdf_path)} with model {self.cfg.model}: {e}")
        return resp.choices[0].message.content or ""


def process_pdfs_with_vllm(input_dir: str, output_dir: str, client: VLLMQwenClient) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    processed = success = failed = 0
    for name in os.listdir(input_dir):
        if not name.lower().endswith(".pdf"):
            continue
        processed += 1
        src = os.path.join(input_dir, name)
        try:
            text = client.ocr_pdf(src)
            out_name = os.path.splitext(name)[0] + "_qwen.txt"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[VLLM] OK {name} → {out_path} ({len(text)} chars)")
            success += 1
        except Exception as e:
            print(f"[VLLM] FAILED {name}: {e}")
            failed += 1
    return {"processed": processed, "success": success, "failed": failed, "output": output_dir}

