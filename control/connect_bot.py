"""Hilfsskript, um die Verbindung zum uArm Swift Pro zu pruefen."""

from __future__ import annotations

try:
    from config import UARM_CALLBACK_THREADS, UARM_PORT
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import UARM_CALLBACK_THREADS, UARM_PORT
from gaming_robot_arm import VisionControlRuntime
from utils.logger import logger
from vision.recording import recording_session
from threading import Event, Thread


def main(port: str | None = UARM_PORT) -> None:
    with VisionControlRuntime(display=False) as runtime:
        controller = runtime.ensure_controller(
            port=port,
            callback_thread_pool_size=UARM_CALLBACK_THREADS,
        )
        swift = controller.swift
        port_name = swift.port if swift else port
        logger.info("uArm bereit auf %s", port_name)

        if swift is None:
            logger.error("Keine Verbindung zum uArm verfuegbar.")
            return

        pump_on = False

        with recording_session() as session:
            stop_recording = Event()

            def _recording_worker() -> None:
                while not stop_recording.is_set():
                    try:
                        frame = session.read()
                    except RuntimeError as exc:
                        logger.warning("Kamera-Lesefehler, stoppe Aufnahme: %s", exc)
                        break
                    session.write(frame)

            recorder = Thread(target=_recording_worker, daemon=True)
            recorder.start()
            logger.info("Aufnahme laeuft: %s", session.output_path)
            logger.info(
                "Steuerung bereit: [m] Move, [s] Suction toggle, [r] Reset, [q] Quit."
            )

            while True:
                cmd = input("Befehl (m/s/r/q): ").strip().lower()

                if cmd == "q":
                    logger.info("Beende Test-Routine.")
                    stop_recording.set()
                    recorder.join(timeout=2.0)
                    swift.disconnect()
                    break
                if cmd == "m":
                    try:
                        x = float(input("x: ").strip())
                        y = float(input("y: ").strip())
                        z = float(input("z: ").strip())
                    except ValueError:
                        logger.warning("Ungueltige Koordinaten, bitte erneut versuchen.")
                        continue

                    logger.info("Bewege zu x=%.1f y=%.1f z=%.1f", x, y, z)
                    swift.set_position(x=x, y=y, z=z, wait=True)
                    continue

                if cmd == "s":
                    pump_on = not pump_on
                    swift.set_pump(on=pump_on)
                    logger.info("Suction %s.", "ON" if pump_on else "OFF")
                    continue

                if cmd == "r":
                    logger.info("Setze uArm zurueck.")
                    swift.reset()
                    if pump_on:
                        swift.set_pump(on=True)
                    continue

                logger.warning("Unbekannter Befehl: %s", cmd)
if __name__ == "__main__":
    main()
