"""
Раннер для разбиения PDF по страницам.
"""
import os
from src.methods.page_splitter.pdf_splitter import split_pdfs
from .config import PipelineConfig


def split_pages(input_dir: str | None = None, output_dir: str | None = None):
    split_pdfs(input_folder=input_dir, output_folder=output_dir)

