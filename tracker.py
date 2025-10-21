import numpy as np
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, maxDisappeared=30, maxDistance=60):
        """
        Hybrid tracker combining time-based disappearance handling and distance-based matching.

        Tracks detected objects based on centroid distance.
        Each object keeps its color, ID, and trajectory.

        Args:
        maxDisappeared (int): Number of frames an object can be missing before deregistration.
        maxDistance (float): Maximum distance for a valid match between centroids.
        """
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.colors = OrderedDict()
        self.disappeared = OrderedDict()
        self.positions = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid, color):
        # Registers a new object with its centroid and color.
        self.objects[self.nextObjectID] = centroid
        self.colors[self.nextObjectID] = color
        self.positions[self.nextObjectID] = [centroid]
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Removes an object that has been missing for too long.
        del self.objects[objectID]
        del self.colors[objectID]
        del self.positions[objectID]
        del self.disappeared[objectID]

    def update(self, centroids, colors):
        # Updates tracked objects based on the latest detected centroids and their colors.
        if len(centroids) != len(colors):
            raise ValueError("Centroids and colors must have the same length")


        # No existing objects → register all new detections
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

        # Retrieve object IDs and their last known centroids
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        # Compute distance matrix between existing and new centroids
        D = np.linalg.norm(
            np.array(objectCentroids)[:, None] - np.array(centroids)[None, :], 
            axis=2
            )

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows, usedCols = set(), set()

        # Match objects to centroids based on minimal distance
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            # Only match if distance is below the threshold
            if D[row, col] > self.maxDistance:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = centroids[col]
            self.positions[objectID].append(centroids[col])
            self.colors[objectID] = colors[col]
            self.disappeared[objectID] = 0

            usedRows.add(row)
            usedCols.add(col)

        # Register new, unmatched detections
        unusedCols = set(range(D.shape[1])).difference(usedCols)
        for col in unusedCols:
            self.register(centroids[col], colors[col])

        # Increase disappearance counter for unmatched existing objects
        unusedRows = set(range(D.shape[0])).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)

        return self.objects

    # def _debug_output(self):
    #     # Prints an overview of all active tracked objects.
    #     print("\n[CentroidTracker DEBUG]")
    #     for objectID in self.objects.keys():
    #         centroid = self.objects[objectID]
    #         color = self.colors[objectID]
    #         disappeared = self.disappeared[objectID]
    #         x, y = centroid
    #         print(f"  ID {objectID:02d}: {color:<5}  pos=({x:4d}, {y:4d})  disappeared={disappeared}")
    #     print("-" * 40)