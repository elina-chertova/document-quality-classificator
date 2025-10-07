from src.methods.inference.cer_utils import cer_for_folders


def run_cer(hyp_dir: str, ref_dir: str, csv_out: str) -> dict:
    result = cer_for_folders(hyp_dir, ref_dir, csv_out)
    macro = result.get("__macro__", 0.0)
    print(f"[CER] Macro CER = {macro}. CSV: {csv_out}")
    return result



run_cer(
    hyp_dir="/Users/elinacertova/Downloads/documents_dataset/results/ocr",   # распознанные тексты
    ref_dir="/Users/elinacertova/Downloads/documents_dataset/results/ocr_labeled_pdf_true",   # эталоны с теми же именами файлов
    csv_out="cer_results.csv"
)


