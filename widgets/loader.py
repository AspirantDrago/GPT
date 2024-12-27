from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QWidget


class LoaderWidget(QWidget):
    def __init__(
            self,
            parent: QWidget | None = None
    ):
        super(LoaderWidget, self).__init__(parent)
        self._background_active = QColor("#2A00B4")
        self._color = QColor("#5DDFFF")
        self._border_pen = QtGui.QPen()
        self._border_pen.setWidth(1)
        self._border_pen.setColor(QtGui.QColor("#B1B1B1"))
        self._position = 0.0
        self._fps = 60
        self._duration = 2.0
        self._width_color = 0.25
        self._animation_timer = QTimer()
        self._animation_timer.setInterval(int(1000 / self._fps))
        self._animation_timer.timeout.connect(self._animate)
        self._animation_timer.start()

    def _animate(self) -> None:
        self._position += 1 / (self._fps * self.duration)
        if self._position >= 1.0:
            self._position = 0.0
        self.update()

    @property
    def width_color(self) -> float:
        return self._width_color * 2

    @width_color.setter
    def width_color(self, width_color: float) -> None:
        if not isinstance(width_color, float):
            raise TypeError("Width color must be a float")
        if width_color <= 0.0 or width_color >= 1.0:
            raise ValueError("Width color must be between 0.0 and 1.0")
        self._width_color = width_color / 2.0

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        if not isinstance(duration, (int, float)):
            raise TypeError("Duration must be a number")
        if duration <= 0:
            raise ValueError("Duration must be positive")
        self._duration = float(duration)

    @property
    def fps(self) -> int:
        return self._fps

    @fps.setter
    def fps(self, fps: int) -> None:
        if not isinstance(fps, int):
            raise TypeError("FPS must be an integer")
        if fps <= 0:
            raise ValueError("FPS must be positive")
        if fps > 1000:
            raise ValueError("FPS must be less than or equal to 1000")
        self._fps = fps
        self._animation_timer.stop()
        self._animation_timer.setInterval(1000 / self._fps)
        self._animation_timer.start()

    @property
    def background_active(self) -> QColor:
        return self._background_active

    @background_active.setter
    def background_active(self, color: QColor):
        self._background_active = QColor(color)
        self.update()

    @property
    def border_width(self) -> int:
        return self._border_pen.width()

    @border_width.setter
    def border_width(self, width: int) -> None:
        if not isinstance(width, int):
            raise TypeError("Border width must be an integer")
        if width < 0:
            raise ValueError("Border width must be non-negative")
        self._border_pen.setWidth(width)
        self.update()

    @property
    def border_color(self) -> QColor:
        return self._border_pen.color()

    @border_color.setter
    def border_color(self, color: QColor) -> None:
        self._border_pen.setColor(QColor(color))
        self.update()

    def paintEvent(self, event):
        if not self.isEnabled():
            return

        w = self.width()
        h = self.height()
        border = self._border_pen.width()
        qp = QPainter()
        qp.begin(self)

        qp.setPen(self._border_pen)
        qp.setBrush(self._background_active)
        qp.drawRect(0, 0, w, h)

        for x in range(border, w - border):
            float_x = x / w
            dx = min(
                abs(float_x - self._position),
                abs(1 + float_x - self._position),
                abs(-1 + float_x - self._position)
            )
            if dx < 0:
                dx += 1.0
            dx /= self._width_color
            if dx > 1:
                continue
            color_r = round(self._background_active.red() * dx + self._color.red() * (1 - dx))
            color_g = round(self._background_active.green() * dx + self._color.green() * (1 - dx))
            color_b = round(self._background_active.blue() * dx + self._color.blue() * (1 - dx))

            qp.setPen(QColor(color_r, color_g, color_b))
            qp.drawLine(x, border, x, h - border)

        qp.end()

    def setEnabled(self, a0: bool) -> None:
        if a0:
            self._animation_timer.start()
        else:
            self._animation_timer.stop()
        super().setEnabled(a0)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = LoaderWidget()
    w.show()
    sys.exit(app.exec())
