import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer


def getGb(bites: int) -> str:
    """
    Преобразовать байты в гигабайты с учетом знака и округления до двух знаков после запятой
    """
    gb = bites / (1024 * 1024 * 1024)
    return f'{gb:.1f}'


def get_model_and_tokenizer() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # base_model_id = "mistralai/Ministral-8B-Instruct-2410"
    base_model_id = r"C:\Users\User\.cache\huggingface\hub\models--mistralai--Ministral-8B-Instruct-2410\snapshots\4847e87e5975a573a2a190399ca62cd266c899ad"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        add_bos_token=True,
        legacy=False,
    )
    return model, tokenizer
