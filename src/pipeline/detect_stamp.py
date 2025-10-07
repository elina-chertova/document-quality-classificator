from src.methods.detector import (
    detect_stamps_single,
    detect_stamps_folder,
    remove_stamps_from_image,
    remove_stamps_from_folder,
)

# # 1. НАЙТИ печати в ОДНОМ ФАЙЛЕ
# result = detect_stamps_single("document.pdf")
# print(f"Найдено: {result['num_stamps']}")


# # УДАЛИТЬ печати в ОДНОМ ФАЙЛЕ
# result = remove_stamps_from_image("document.pdf")
# print(f"Затерто: {result['num_stamps']}, очищенный PDF: {result['cleaned_pdf_path']}")


# 2. НАЙТИ печати в ПАПКЕ
summary = detect_stamps_folder("/Users/elinacertova/PycharmProjects/documents_preprocessing/test_output/")
print(f"Обработано: {summary['total_images']}, найдено: {summary['total_stamps']}")

# УДАЛИТЬ печати в ПАПКЕ
summary = remove_stamps_from_folder("/Users/elinacertova/PycharmProjects/documents_preprocessing/test_output/")
print(f"Обработано: {summary['total_images']}, затерто: {summary['total_stamps']}")
