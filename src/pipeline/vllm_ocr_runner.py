from src.methods.inference.vllm_qwen_client import VLLMQwenClient, VLLMQwenConfig, process_pdfs_with_vllm
from src.pipeline.config import PipelineConfig


def run_vllm_ocr(root_dir: str = "/Users/elinacertova/Downloads/documents_dataset") -> dict:
    cfg = PipelineConfig()
    if root_dir:
        cfg.paths.root_dir = root_dir

    input_dir = cfg.paths.output_folder
    out_dir = cfg.paths.ocr_output_folder + "_vllm"

    client = VLLMQwenClient(VLLMQwenConfig())
    result = process_pdfs_with_vllm(input_dir, out_dir, client)
    print(f"[VLLM] Summary: processed={result['processed']} success={result['success']} failed={result['failed']}\nTXT dir: {result['output']}")
    return result

run_vllm_ocr()


