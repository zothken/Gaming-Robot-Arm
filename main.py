"""Startpunkt fuer das Gaming-Robot-Arm Laufzeitsystem."""

from config import CAMERA_INDEX
from gaming_robot_arm import VisionControlRuntime


def main(camera_index: int = CAMERA_INDEX) -> None:
    """Initialisiert die Laufzeitumgebung und startet den Vision-Loop."""
    with VisionControlRuntime() as runtime:
        runtime.run(camera_index=camera_index)


if __name__ == "__main__":
    main()
