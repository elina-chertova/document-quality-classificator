from src.methods.inference.infer_quality import predict_quality_for_folder


results = predict_quality_for_folder(
    "/Users/elinacertova/Downloads/documents_dataset/results/processed",
    model_path="/final_quality_classifier_model.pkl",
)
