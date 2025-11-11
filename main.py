"""
Оркестратор: сперва deskew, затем объединённая обработка.
"""

import sys

from src.pipeline.dark_docs_to_light import dark_documents_to_light
from src.pipeline.deskew import deskew_documents
from src.pipeline.fast_version.process_single_document_fast import process_single_document_fast
from src.pipeline.quality_comparision.quality_comparison import compare_folder
from src.pipeline.rotate_right_runner import rotate_right
from src.pipeline.remove_lines_runner import remove_lines
from src.pipeline.split_pages import split_pages
from src.pipeline.quality_classifier_runner import classify_by_quality
from src.pipeline.process_single_document import process_single_document

from src.pipeline.enhance_contrast_runner import enhance_contrast_documents

# from src.pipeline.config import PipelineConfig


# def main(root_dir: str | None = None, full_dateset_dir: str | None = None):
#     cfg = PipelineConfig()
#     if root_dir:
#         cfg.paths.root_dir = root_dir
#     split_pages(root_dir=cfg.paths.root_dir, full_dateset_dir=full_dateset_dir)
#     rotate_right(root_dir=cfg.paths.root_dir)
#     deskew_documents(root_dir=cfg.paths.root_dir)
#     # remove_lines(root_dir=cfg.paths.root_dir, log_csv=None)
#     # dark_documents_to_light()

def main():

    # split_pages(input_dir="/Users/elinacertova/Downloads/dataset_tester_full/docs", output_dir="/Users/elinacertova/Downloads/dataset_tester_full/splitted")
    # rotate_right(input_dir='/Users/elinacertova/Downloads/dataset_tester_full/splitted',
    #              output_dir='/Users/elinacertova/Downloads/dataset_tester_full/rotated',
    #              failed_dir='/Users/elinacertova/Downloads/dataset_tester_full/failed')
    # deskew_documents(input_dir='/Users/elinacertova/Downloads/dataset_tester_full/rotated',
    #                  output_dir='/Users/elinacertova/Downloads/dataset_tester_full/deskewed',
    #                  failed_dir='/Users/elinacertova/Downloads/dataset_tester_full/failed')
    # remove_lines(input_dir='/Users/elinacertova/Downloads/dataset_tester_full/deskewed',
    #              lines_cleaned_folder='/Users/elinacertova/Downloads/dataset_tester_full/lines_cleaned',
    #              no_lines_ok_folder='/Users/elinacertova/Downloads/dataset_tester_full/lines_not_detected',
    #              combined_output_folder='/Users/elinacertova/Downloads/dataset_tester_full/combined')
    dark_documents_to_light(input_folder='/Users/elinacertova/Downloads/dataset_tester_full/combined',
                            output_folder='/Users/elinacertova/Downloads/dataset_tester_full/lightened',
                            dark_folder='/Users/elinacertova/Downloads/dataset_tester_full/dark',
                            combined_output_folder='/Users/elinacertova/Downloads/dataset_tester_full/lightened_combined',
                            lightening_method='bilateral_filter')
    # enhance_contrast_documents(input_dir='/Users/elinacertova/Downloads/dataset_tester_full/lightened_combined',
    #                            output_dir='/Users/elinacertova/Downloads/dataset_tester_full/contrast_enhanced')
    # classify_by_quality(input_folder='/Users/elinacertova/Downloads/dataset_tester_full/contrast_enhanced',
    #                     output_folder='/Users/elinacertova/Downloads/dataset_tester_full/classified')
    
    # process_single_document_fast(
    #     input_pdf_path='/Users/elinacertova/Downloads/single_doc_test/Scan_20250213_120013.pdf',
    #     output_base_dir='/Users/elinacertova/Downloads/single_doc_test/output_optimized',
    #     output_csv_path='/Users/elinacertova/Downloads/single_doc_test/results_optimized.csv',
    #     dpi=250,
    #     max_workers=4
    # )
    
    # process_single_document(
    #     input_pdf_path='/Users/elinacertova/Downloads/single_doc_test/Scan_20250213_120013.pdf',
    #     output_base_dir='/Users/elinacertova/Downloads/single_doc_test/output',
    #     output_csv_path='/Users/elinacertova/Downloads/single_doc_test/results.csv'
    # )
    
    # process_single_document_fastest(
    #     input_pdf_path='/Users/elinacertova/Downloads/single_doc_test/Scan_20250213_120013.pdf',
    #     output_base_dir='/Users/elinacertova/Downloads/single_doc_test/output_fastest',
    #     output_csv_path='/Users/elinacertova/Downloads/single_doc_test/results_fastest.csv',
    #     dpi=300,
    #     skip_deskew=False
    # )
    
    # process_single_document_ultrafast(
    #     input_pdf_path='/Users/elinacertova/Downloads/single_doc_test/Scan_20250213_120013.pdf',
    #     output_base_dir='/Users/elinacertova/Downloads/single_doc_test/output_ultrafast',
    #     output_csv_path='/Users/elinacertova/Downloads/single_doc_test/results_ultrafast.csv',
    #     dpi=200
    # )
    
    # compare_folder(
    #     original_dir='/Users/elinacertova/Downloads/single_doc_test/output_optimized/splitted',
    #     processed_dir='/Users/elinacertova/Downloads/single_doc_test/output_optimized/deskewed',
    #     output_csv='/Users/elinacertova/Downloads/single_doc_test/quality_comparison.csv',
    #     dpi=400
    # )
    compare_folder(
        original_dir='/Users/elinacertova/Downloads/dataset_tester_full/splitted',
        processed_dir='/Users/elinacertova/Downloads/dataset_tester_full/contrast_enhanced',
        output_csv='/Users/elinacertova/Downloads/dataset_tester_full/quality_comparison.csv',
        dpi=300
    )

if __name__ == "__main__":
    sys.exit(main())



from src.pipeline.quality_improvements_runner import improve_dataset
# improve_dataset(mode="ocrmypdf", root_dir="/Users/elinacertova/Downloads/documents_dataset")
# improve_dataset(mode="scantailor_then_ocr", root_dir="/Users/elinacertova/Downloads/documents_dataset")
# improve_dataset(mode="unpaper_tesseract", root_dir="/Users/elinacertova/Downloads/documents_dataset")

