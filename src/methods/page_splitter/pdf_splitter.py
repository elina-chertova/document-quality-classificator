import os
import shutil
import fitz

def split_pdf_with_pymupdf(input_pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    pdf_doc = fitz.open(input_pdf_path)
    base_name = os.path.splitext(os.path.basename(input_pdf_path))[0]

    for i in range(len(pdf_doc)):
        page = pdf_doc.load_page(i)
        new_doc = fitz.open()
        new_doc.insert_pdf(pdf_doc, from_page=i, to_page=i)
        output_path = os.path.join(output_folder, f"{base_name}_page_{i + 1}.pdf")
        new_doc.save(output_path)
        new_doc.close()

    pdf_doc.close()
    print(f"[OK] Processed PDF: {input_pdf_path}")

def split_pdfs(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        if filename.lower().endswith(".pdf"):
            try:
                split_pdf_with_pymupdf(filepath, output_folder)
            except Exception as e:
                print(f"[ERROR] Failed to process PDF {filename}: {e}")
        elif filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")):
            try:
                shutil.copy(filepath, output_folder)
                print(f"[OK] Copied image: {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to copy image {filename}: {e}")
        else:
            print(f"[SKIPPED] Unsupported file type: {filename}")
