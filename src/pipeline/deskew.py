from src.methods.improver.rotate.rotate_any_angle import PDFDeskewParams, PDFDeskewer


def deskew_documents(input_dir: str, output_dir: str, failed_dir: str):
    params = PDFDeskewParams(dpi=400, angle_limit=35.0, jpeg_quality=85)
    deskewer = PDFDeskewer(params)

    deskewer.process_folder(
        input_folder=input_dir,
        output_folder=output_dir,
        failed_folder=failed_dir,
    )

