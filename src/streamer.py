import torch
from transformers import TextStreamer


class PyQtStreamer(TextStreamer):
    def __init__(self, tokenizer, update_signal, skip_special_tokens=True):
        super().__init__(tokenizer, skip_special_tokens)
        self.update_signal = update_signal

    def put(self, value):
        text = self.tokenizer.decode(value[0], skip_special_tokens=True)
        self.update_signal.emit(text)  # Передача текста через сигнал
        super().put(value)
