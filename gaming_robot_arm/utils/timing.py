import time


class FPSTracker:
    """Einfacher FPS-Tracker zur Messung der Bildrate in einer Schleife."""

    def __init__(self):
        self.last_time = time.time()
        self.fps = 0.0

    def update(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps = 1.0 / dt
        return self.fps
