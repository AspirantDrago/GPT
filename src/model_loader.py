from PyQt6.QtCore import QThread, pyqtSignal

from utils import get_model_and_tokenizer


class ModelLoaderThread(QThread):
    model_loaded = pyqtSignal(object, object)  # Сигнал для передачи модели и токенизатора
    error = pyqtSignal(str)  # Сигнал для передачи ошибки

    def run(self):
        try:
            self.model_loaded.emit(*get_model_and_tokenizer())
        except Exception as e:
            self.error.emit(str(e))
