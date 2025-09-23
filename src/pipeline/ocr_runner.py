import os

from src.methods.inference.surya_ocr_client import SuryaOCRClient, SuryaOCRConfig, process_folder_with_ocr
from src.pipeline.config import PipelineConfig


def run_ocr(root_dir: str = "/Users/elinacertova/Downloads/documents_dataset") -> dict:
    cfg = PipelineConfig()
    if root_dir:
        cfg.paths.root_dir = root_dir

    # input_dir = cfg.paths.output_folder
    input_dir = os.path.join(root_dir, "results", "labeled_pdf")
    out_dir = os.path.join(root_dir, "results", "ocr_labeled_pdf_true")
    csv_path = cfg.paths.ocr_csv_path

    client = SuryaOCRClient(SuryaOCRConfig())
    result = process_folder_with_ocr(input_dir, out_dir, csv_path, client)
    print(f"[OCR] Summary: processed={result['processed']} success={result['success']} failed={result['failed']}\nCSV: {result['csv']}\nTXT dir: {result['output']}")
    return result


run_ocr()


