import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

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


# eval_prompt = """Зачем ученикам изучать побитовые операции?"""
# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
#
# streamer = TextStreamer(tokenizer, skip_special_tokens=True)
# with torch.no_grad():
#     output_ids = model.generate(
#         **model_input,
#         max_new_tokens=128*1024,  # Ограничение длины генерируемого текста
#         do_sample=True,
#         temperature=0.7,
#         pad_token_id=tokenizer.eos_token_id,
#         streamer=streamer,
#     )
