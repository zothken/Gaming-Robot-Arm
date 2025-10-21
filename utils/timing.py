import time

class FPSTracker:
    """Simple FPS tracker to measure frame rate in a loop."""
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