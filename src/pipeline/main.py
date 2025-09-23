"""
Оркестратор: сперва deskew, затем объединённая обработка.
"""

import sys

from src.pipeline.dark_docs_to_light import dark_documents_to_light
from src.pipeline.deskew import deskew_documents
from src.pipeline.rotate_right_runner import rotate_right
from src.pipeline.remove_lines_runner import remove_lines
from src.pipeline.split_pages import split_pages
from src.pipeline.config import PipelineConfig


def main(root_dir: str | None = None, full_dateset_dir: str | None = None):
    cfg = PipelineConfig()
    if root_dir:
        cfg.paths.root_dir = root_dir
    split_pages(root_dir=cfg.paths.root_dir, full_dateset_dir=full_dateset_dir)
    rotate_right(root_dir=cfg.paths.root_dir)
    deskew_documents(root_dir=cfg.paths.root_dir)
    remove_lines(root_dir=cfg.paths.root_dir, log_csv=None)
    dark_documents_to_light()



if __name__ == "__main__":
    sys.exit(main())



from src.pipeline.quality_improvements_runner import improve_dataset
# improve_dataset(mode="ocrmypdf", root_dir="/Users/elinacertova/Downloads/documents_dataset")
# improve_dataset(mode="scantailor_then_ocr", root_dir="/Users/elinacertova/Downloads/documents_dataset")
# improve_dataset(mode="unpaper_tesseract", root_dir="/Users/elinacertova/Downloads/documents_dataset")

