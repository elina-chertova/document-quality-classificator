import os
from src.methods.improver.rotate.rotate_any_angle import PDFDeskewParams, PDFDeskewer
from .config import PipelineConfig


def deskew_documents(root_dir: str = "/Users/elinacertova/Downloads/documents_dataset"):
    cfg = PipelineConfig()
    if root_dir:
        cfg.paths.root_dir = root_dir
    params = PDFDeskewParams(dpi=300, angle_limit=35.0, jpeg_quality=85)
    deskewer = PDFDeskewer(params)
    input_folder = cfg.paths.rotated_folder
    output_folder = cfg.paths.deskewed_folder
    failed_folder = cfg.paths.failed_deskew_folder
    deskewer.process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        failed_folder=failed_folder,
    )

