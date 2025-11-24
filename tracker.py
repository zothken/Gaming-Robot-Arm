from collections import OrderedDict

import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=150, maxDistance=100):
        """
        Hybrid-Tracker, der zeitbasierte Ausblendung mit distanzbasierter Zuordnung kombiniert.

        Verfolgt erkannte Objekte anhand ihres Schwerpunktes.
        Jedes Objekt behaelt Farbe, ID und Bewegungsverlauf.

        Parameter:
        maxDisappeared (int): Anzahl der Frames, die ein Objekt fehlen darf, bevor es entfernt wird.
        maxDistance (float): Maximale Distanz fuer eine gueltige Zuordnung von Schwerpunkten.
        """
        self.nextBlackID = 1
        self.nextWhiteID = 1

        self.objects = OrderedDict()
        self.colors = OrderedDict()
        self.disappeared = OrderedDict()
        self.positions = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid, color):
        # Registriert ein neues Objekt mit Schwerpunkt und Farbzuordnung.
        if color == "black":
            objectID = f"B{self.nextBlackID}"
            self.nextBlackID += 1
        elif color == "white":
            objectID = f"W{self.nextWhiteID}"
            self.nextWhiteID += 1
        else:
            # Rueckfall fuer unbekannte Farbe
            objectID = f"U{len(self.objects) + 1}"

        self.objects[objectID] = centroid
        self.colors[objectID] = color
        self.positions[objectID] = [centroid]
        self.disappeared[objectID] = 0

    def deregister(self, objectID):
        # Entfernt ein Objekt, das zu lange nicht mehr erkannt wurde.
        del self.objects[objectID]
        del self.colors[objectID]
        del self.positions[objectID]
        del self.disappeared[objectID]

    def update(self, centroids, colors):
        # Aktualisiert die verfolgten Objekte anhand der neuesten Schwerpunkte und zugehoeriger Farben.
        if len(centroids) != len(colors):
            raise ValueError("Centroids and colors must have the same length")

        # Keine bestehenden Objekte -> alle neuen Erkennungen registrieren
        if len(self.objects) == 0:
            for i, centroid in enumerate(centroids):
                self.register(centroid, colors[i])
            return self.objects

        if len(centroids) == 0:
            for objectID in list(self.objects.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # IDs und letzte bekannte Schwerpunkte aller Objekte sammeln
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        # Distanzmatrix zwischen bestehenden und neuen Schwerpunkten berechnen
        D = np.linalg.norm(
            np.array(objectCentroids)[:, None] - np.array(centroids)[None, :],
            axis=2,
        )

        # Debug-Ausgabe
        # print("Distance matrix (pixels):")
        # print(D)
        # min_distances = D.min(axis=1)
        # print("Minimal distances for each object:", min_distances)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows, usedCols = set(), set()

        # Objekte anhand der minimalen Distanz zu den neuen Schwerpunkten zuordnen
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            # Nur zuordnen, wenn die Distanz unterhalb des Schwellwerts liegt
            if D[row, col] > self.maxDistance:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = centroids[col]
            self.positions[objectID].append(centroids[col])
            self.colors[objectID] = colors[col]
            self.disappeared[objectID] = 0

            usedRows.add(row)
            usedCols.add(col)

        # Neue, nicht zugeordnete Erkennungen registrieren
        unusedCols = set(range(D.shape[1])).difference(usedCols)
        for col in unusedCols:
            self.register(centroids[col], colors[col])

        # Ausblendungszaehler fuer nicht zugeordnete bestehende Objekte erhoehen
        unusedRows = set(range(D.shape[0])).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)

        return self.objects
