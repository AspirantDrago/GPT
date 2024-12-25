import torch
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import QThread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from src.streamer import PyQtStreamer


class StreamingThread(QThread):
    update_response = pyqtSignal(str)  # Сигнал для передачи обновлений текста
    generation_finished = pyqtSignal()  # Сигнал для завершения генерации
    error_occurred = pyqtSignal(str)  # Сигнал для передачи ошибок

    def __init__(self,
                 model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 user_input: str,
                 temperature: float,
                 ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.user_input = user_input
        self.temperature = temperature

    def run(self):
        try:
            model_input = self.tokenizer(self.user_input, return_tensors="pt").to("cuda")
            streamer = PyQtStreamer(self.tokenizer, self.update_response)
            with torch.no_grad():
                self.model.generate(
                    **model_input,
                    max_new_tokens=128*1024,
                    do_sample=bool(self.temperature),
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                )
            self.generation_finished.emit()  # Сигнал завершения генерации
        except Exception as e:
            self.error_occurred.emit(str(e))
