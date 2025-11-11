from src.methods.classificator.classify_quality_folder import classify_quality_and_copy


def classify_by_quality(
    input_folder: str,
    output_folder: str,
    dpi: int = 400,
    max_workers: int = 4,
    good_subdir: str = "good",
    medium_subdir: str = "medium",
    failed_subdir: str = "failed",
    trash_subdir: str = "trash",
):
    print("=" * 60)
    print("КЛАССИФИКАЦИЯ ДОКУМЕНТОВ ПО КАЧЕСТВУ")
    print("=" * 60)
    print(f"Входная папка: {input_folder}")
    print(f"Выходная папка: {output_folder}")
    print()
    
    classify_quality_and_copy(input_folder, output_folder)
    
    print("=" * 60)
    print("КЛАССИФИКАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)



