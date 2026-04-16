"""Paket-Exporte fuer die Gaming-Robot-Arm-Laufzeit."""

try:
    from .runtime import VisionControlRuntime  # pyright: ignore[reportAssignmentType]
except ModuleNotFoundError as exc:
    _runtime_import_error = exc

    class VisionControlRuntime:  # type: ignore[no-redef,assignment]  # pyright: ignore[reportAssignmentType]
        """Laufzeit-Platzhalter, der bei Nutzung einen klaren Abhaengigkeitsfehler wirft."""

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "VisionControlRuntime benoetigt optionale Laufzeitabhaengigkeiten "
                "(zum Beispiel OpenCV). Bitte zuerst die Projektanforderungen installieren."
            ) from _runtime_import_error

__all__ = ["VisionControlRuntime"]
