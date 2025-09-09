"""
Раннер для разбиения PDF по страницам.
"""
import os
from src.methods.page_splitter.pdf_splitter import split_pdfs
from .config import PipelineConfig


def split_pages(root_dir: str = '/Users/elinacertova/Downloads/documents_dataset',
                full_dateset_dir: str | None = None):
    cfg = PipelineConfig()
    if root_dir:
        cfg.paths.root_dir = root_dir
    input_dir = full_dateset_dir if full_dateset_dir else cfg.paths.full_dataset_folder
    output_dir = cfg.paths.split_folder
    split_pdfs(input_folder=input_dir, output_folder=output_dir)


if __name__ == "__main__":
    pass


