"""
Оркестратор: сперва deskew, затем объединённая обработка.
"""

import sys

from src.pipeline.dark_docs_to_light import dark_documents_to_light
from src.pipeline.deskew import deskew_documents
from src.pipeline.rotate_right_runner import rotate_right
from src.pipeline.split_pages import split_pages
from src.pipeline.config import PipelineConfig


def main(root_dir: str | None = None, full_dateset_dir: str | None = None):
    cfg = PipelineConfig()
    if root_dir:
        cfg.paths.root_dir = root_dir
    split_pages(root_dir=cfg.paths.root_dir, full_dateset_dir=full_dateset_dir)
    rotate_right(root_dir=cfg.paths.root_dir)
    deskew_documents(root_dir=cfg.paths.root_dir)
    dark_documents_to_light()


if __name__ == "__main__":
    sys.exit(main())