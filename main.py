"""
Оркестратор: сперва deskew, затем объединённая обработка.
"""

import sys

from src.pipeline.dark_docs_to_light import dark_documents_to_light
from src.pipeline.deskew import deskew_documents
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

    # split_pages(input_dir="/Users/elinacertova/Downloads/dataset_tester_dir/docs", output_dir="/Users/elinacertova/Downloads/dataset_tester_dir/splitted")
    # rotate_right(input_dir='/Users/elinacertova/Downloads/dataset_tester_dir/splitted',
    #              output_dir='/Users/elinacertova/Downloads/dataset_tester_dir/rotated',
    #              failed_dir='/Users/elinacertova/Downloads/dataset_tester_dir/failed')
    # deskew_documents(input_dir='/Users/elinacertova/Downloads/dataset_tester_dir/rotated',
    #                  output_dir='/Users/elinacertova/Downloads/dataset_tester_dir/deskewed',
    #                  failed_dir='/Users/elinacertova/Downloads/dataset_tester_dir/failed')
    # remove_lines(input_dir='/Users/elinacertova/Downloads/dataset_tester_dir/deskewed',
    #              lines_cleaned_folder='/Users/elinacertova/Downloads/dataset_tester_dir/lines_cleaned',
    #              no_lines_ok_folder='/Users/elinacertova/Downloads/dataset_tester_dir/lines_not_detected',
    #              combined_output_folder='/Users/elinacertova/Downloads/dataset_tester_dir/combined')
    # dark_documents_to_light(input_folder='/Users/elinacertova/Downloads/dataset_tester_dir/combined',
    #                         output_folder='/Users/elinacertova/Downloads/dataset_tester_dir/lightened',
    #                         dark_folder='/Users/elinacertova/Downloads/dataset_tester_dir/dark',
    #                         combined_output_folder='/Users/elinacertova/Downloads/dataset_tester_dir/lightened_combined')
    # enhance_contrast_documents(input_dir='/Users/elinacertova/Downloads/dataset_tester_dir/lightened_combined',
    #                            output_dir='/Users/elinacertova/Downloads/dataset_tester_dir/contrast_enhanced')
    # classify_by_quality(input_folder='/Users/elinacertova/Downloads/dataset_tester_dir/contrast_enhanced',
    #                     output_folder='/Users/elinacertova/Downloads/dataset_tester_dir/classified')
    
    process_single_document(
        input_pdf_path='/Users/elinacertova/Downloads/single_doc_test/Scan_20250213_120013.pdf',
        output_base_dir='/Users/elinacertova/Downloads/single_doc_test/output',
        output_csv_path='/Users/elinacertova/Downloads/single_doc_test/results.csv'
    )


if __name__ == "__main__":
    sys.exit(main())



from src.pipeline.quality_improvements_runner import improve_dataset
# improve_dataset(mode="ocrmypdf", root_dir="/Users/elinacertova/Downloads/documents_dataset")
# improve_dataset(mode="scantailor_then_ocr", root_dir="/Users/elinacertova/Downloads/documents_dataset")
# improve_dataset(mode="unpaper_tesseract", root_dir="/Users/elinacertova/Downloads/documents_dataset")

