"""
Оркестратор: сперва deskew, затем объединённая обработка.
"""

import sys
from pathlib import Path

from src.pipeline.classify_single_document import classify_single_document
from src.pipeline.compare_binarization_ocr import compare_binarization_ocr
# from src.pipeline.dark_docs_to_light import dark_documents_to_light
# from src.pipeline.deskew import deskew_documents
# from src.pipeline.fast_version.process_single_document_fast import process_single_document_fast
# from src.pipeline.quality_comparision.quality_comparison import compare_folder
# from src.pipeline.rotate_right_runner import rotate_right
# from src.pipeline.remove_lines_runner import remove_lines
# from src.pipeline.split_pages import split_pages
# from src.pipeline.quality_classifier_runner import classify_by_quality
from src.pipeline.process_single_document import process_single_document

# from src.pipeline.config import PipelineConfig


def main():

    # # input_pdf_path = '/Users/elinacertova/PycharmProjects/document-quality-classificator/data_example/Договор_купли_продажи_недвижимого_имущества_пример_2025_для_двух.pdf'
    # input_pdf_path = '/Users/elinacertova/Downloads/single_doc_test/Scan_20250213_120013.pdf'
    # output_base_dir = '/Users/elinacertova/Downloads/single_doc_test/output2'
    # output_csv_path = '/Users/elinacertova/Downloads/single_doc_test/results.csv'
    # pages_dir = process_single_document(
    # #     input_pdf_path='/Users/elinacertova/Downloads/single_doc_test/Scan_20250213_120013.pdf',
    #     input_pdf_path=input_pdf_path,
    #     output_base_dir=output_base_dir,
    #     output_csv_path=output_csv_path
    # )
    #
    # document_name = Path(input_pdf_path).stem
    #
    # results = classify_single_document(
    #     pages_dir=pages_dir,
    #     document_name=document_name,
    #     output_base_dir=output_base_dir,
    #     output_csv_path=output_csv_path,
    #     dpi=400,
    #     max_workers=4,
    #     classifier_dpi=300,
    #     device="cpu",
    #     optimized=False
    # )

    from src.pipeline.process_single_document_smart import process_single_document_smart

    results = process_single_document_smart(
        input_pdf_path="/Users/elinacertova/Downloads/single_doc_test/Scan_20250213_120013.pdf",
        output_base_dir="/Users/elinacertova/Downloads/single_doc_test/output3",
        output_csv_path="results.csv",
        dpi=400,
        max_workers=4,
        classifier_dpi=300,
        device="cpu",
        optimized=False
    )
    # compare_binarization_ocr(
    #     input_dir="/Users/elinacertova/Downloads/dataset_tester_full/deskewed",
    #     output_dir="/Users/elinacertova/Downloads/testing_dir",
    #     comparison_csv_path="/Users/elinacertova/Downloads/binarization_comparison.csv",
    #     dpi=300,
    #     method="adaptive_mean",
    #     block_size=15,
    #     c=10.0,
    # )
    # compare_binarization_ocr(
    #     input_dir="/Users/elinacertova/Downloads/dataset_tester_full/deskewed",
    #     output_dir="/Users/elinacertova/Downloads/testing_dir",
    #     comparison_csv_path="/Users/elinacertova/Downloads/binarization_comparison.csv",
    #     dpi=300,
    #     method="adaptive_mean",
    #     block_size=15,
    #     c=10.0,
    #     device="cpu",  # или "cuda" если доступен GPU
    #     optimized=False,
    # ) https://api.honeypot.is/v2/IsHoneypot?address=0x9254d92d576a10d3dec2966a5a6ebbf493b65827&chainID=8453
    # compare_folder(
    #     original_dir='/Users/elinacertova/Downloads/single_doc_test/output_optimized/splitted',
    #     processed_dir='/Users/elinacertova/Downloads/single_doc_test/output_optimized/deskewed',
    #     output_csv='/Users/elinacertova/Downloads/single_doc_test/quality_comparison.csv',
    #     dpi=400
    # )
    # compare_folder(
    #     original_dir='/Users/elinacertova/Downloads/dataset_tester_full/splitted',
    #     processed_dir='/Users/elinacertova/Downloads/dataset_tester_full/contrast_enhanced',
    #     output_csv='/Users/elinacertova/Downloads/dataset_tester_full/quality_comparison.csv',
    #     dpi=300
    # )

if __name__ == "__main__":
    sys.exit(main())



from src.pipeline.quality_improvements_runner import improve_dataset
# improve_dataset(mode="ocrmypdf", root_dir="/Users/elinacertova/Downloads/documents_dataset")
# improve_dataset(mode="scantailor_then_ocr", root_dir="/Users/elinacertova/Downloads/documents_dataset")
# improve_dataset(mode="unpaper_tesseract", root_dir="/Users/elinacertova/Downloads/documents_dataset")

