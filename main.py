import gc
import sys

import GPUtil
import torch
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt, QEvent, QUrl
from PyQt6.QtGui import QIcon, QKeyEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget

import psutil

from forms.main_form import Ui_GPT
from src.model_loader import ModelLoaderThread
from src.streaming_thread import StreamingThread
from utils import getGb
from src.config import Config
from widgets.loader import LoaderWidget
from markdown import markdown


class MainWindow(QMainWindow, Ui_GPT):
    def __init__(self):
        super().__init__()

        self.model = None
        self.tokenizer = None

        self.initUi()
        self.temperature: float = 0.5
        self.changeTemperature()
        self.answer = ''

        self.reloadBtn.clicked.connect(self.reload_model)
        self.sliderTemperatureModel.valueChanged.connect(self.changeTemperature)
        self.btnPastPrompt.clicked.connect(self.slotPastPrompt)
        self.btnSend.clicked.connect(self.generate_response)

        self.setUIEnabled(False)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(Config.GPU_CPU_UPDATE_INTERVAL)
        self.timer.timeout.connect(self.updateGPU_RAM)
        self.timer.start()
        self.inputPrompt.installEventFilter(self)

        self.start_model_loading()

    def handle_ctrl_s(self):
        print('s')

    def initUi(self):
        self.setupUi(self)
        self.btnPastPrompt.setIcon(QIcon('images/paste.png'))
        self.setWindowIcon(QIcon('images/icon.svg'))

    def changeTemperature(self):
        self.temperature = Config.TEMPERATURE_MAXIMUM * self.sliderTemperatureModel.value() / self.sliderTemperatureModel.maximum()
        self.textTemperatureModel.setText(f't={self.temperature:.2f}')

    def slotPastPrompt(self):
        self.inputPrompt.setText(QtWidgets.QApplication.clipboard().text())

    def updateGPU_RAM(self):
        memory_info = psutil.virtual_memory()
        total_memory = memory_info.total
        free_memory = memory_info.available
        used_memory = total_memory - free_memory
        self.RAMprogressBar.setValue(round(self.GPUprogressBar.maximum() * used_memory / total_memory))
        self.textRAM.setText(f'{getGb(used_memory)}/{getGb(total_memory)} GB')

        gpu = GPUtil.getGPUs()[0]

        total_memory = round(gpu.memoryTotal * 1024 * 1024)
        used_memory = round(gpu.memoryUsed * 1024 * 1024)
        self.GPUprogressBar.setValue(round(self.GPUprogressBar.maximum() * used_memory / total_memory))
        self.textGPU.setText(f'{getGb(used_memory)}/{getGb(total_memory)} GB')
        temperature = int(gpu.temperature)
        self.textTemperatureGPU.setText(f'{temperature} °C')

    def setUIEnabled(self, status: bool) -> None:
        self.inputPrompt.setEnabled(status)
        self.reloadBtn.setEnabled(status)
        self.loadingProgressBar.setEnabled(not status)
        self.sliderTemperatureModel.setEnabled(status)
        self.btnPastPrompt.setEnabled(status)
        self.btnSend.setEnabled(status)
        self.resultView.setEnabled(status)

    def start_model_loading(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        self.loader_thread = ModelLoaderThread()
        self.loader_thread.model_loaded.connect(self.on_model_loaded)
        self.loader_thread.error.connect(self.on_model_error)
        self.loader_thread.start()

    def on_model_loaded(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.setUIEnabled(True)

    def on_model_error(self, error_message):
        print(error_message)

    def reload_model(self):
        self.setUIEnabled(False)
        self.answer = ''
        # self.resultView.close()
        self.start_model_loading()

    def generate_response(self):
        # Проверяем, что модель загружена
        if not self.model or not self.tokenizer:
            QMessageBox(QMessageBox.Icon.Warning, '', "Модель не загружена.").show()
            return

        # Получение ввода от пользователя
        user_input = self.inputPrompt.toPlainText().strip()
        if not user_input:
            QMessageBox(self, QMessageBox.Icon.Warning, "Введите текст, чтобы получить ответ.").show()
            return
        # self.resultView.close()
        self.answer = None
        self.streaming_thread = StreamingThread(self.model, self.tokenizer, user_input, self.temperature)
        self.streaming_thread.update_response.connect(self.update_response)
        self.streaming_thread.generation_finished.connect(self.on_generation_finished)
        self.streaming_thread.error_occurred.connect(self.on_generation_error)
        self.streaming_thread.start()

    def update_response(self, text):
        if self.answer is None:
            self.answer = ''
            return
        self.answer += text

        self.resultView.setHtml(markdown(self.answer), QUrl(''))
        # self.resultView.page().runJavaScript('window.scrollTo(0, document.body.scrollHeight);')

    def on_generation_finished(self):
        mathjax_script = """
                        <script type="text/javascript" async
                          src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
                        </script>
                        <script type="text/x-mathjax-config">
                          MathJax.Hub.Config({
                            tex2jax: {inlineMath: [['$','$']]},
                            messageStyle: 'none'
                          });
                        </script>
                        """
        html_content = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                        {mathjax_script}
                        </head>
                        <body>
                        {markdown(self.answer)}
                        </body>
                        </html>
                        """
        self.resultView.setHtml(html_content, QUrl(''))
        self.resultView.page().runJavaScript('window.scrollTo(0, document.body.scrollHeight);')
        print('finished')

    def on_generation_error(self, error_message):
        print(error_message)
        QMessageBox(QMessageBox.Icon.Critical, '', f"Ошибка: {error_message}").show()

    def eventFilter(self, source: QWidget, event: QEvent):
        if source == self.inputPrompt and event.type() == QEvent.Type.KeyPress.value:
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                self.generate_response()
                return True  # Прекращаем дальнейшую обработку события
        return super().eventFilter(source, event)



def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    sys.excepthook = except_hook
    app = QApplication(sys.argv)
    app.setStyle('fusion')
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec())
