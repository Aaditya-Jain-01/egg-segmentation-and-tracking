import os
import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class EggTracker:
    def __init__(self, max_disappeared=25, max_distance=60):
        self.next_object_id = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.trajectories = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trajectories[self.next_object_id] = [centroid]
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.trajectories[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.trajectories

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(input_centroids[col])
                if len(self.trajectories[object_id]) > 30:
                    self.trajectories[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects, self.trajectories

def detect_egg_candidates(frame):
    h, w = frame.shape[:2]
    total_area = h * w
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    
    thresh = cv2.adaptiveThreshold(
        l_chan, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 25, -5
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    masks = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.005 * total_area or area > 0.60 * total_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / hull_area if hull_area > 0 else 0

        if circularity > 0.40 and solidity > 0.80:
            x, y, bw, bh = cv2.boundingRect(cnt)
            rects.append((x, y, x + bw, y + bh))
            masks.append(cnt)

    return rects, masks

def process_video(video_input_path, video_output_path):
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    tracker = EggTracker(max_disappeared=20, max_distance=80)
    color_palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (255, 165, 0), (0, 128, 255), (128, 0, 255)
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rects, contours = detect_egg_candidates(frame)
        objects, trajectories = tracker.update(rects)

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        for object_id, centroid in objects.items():
            color = color_palette[object_id % len(color_palette)]
            
            box_match = None
            min_d = float("inf")
            for (sx, sy, ex, ey) in rects:
                bx = int((sx + ex) / 2.0)
                by = int((sy + ey) / 2.0)
                d = (bx - centroid[0]) ** 2 + (by - centroid[1]) ** 2
                if d < min_d:
                    min_d = d
                    box_match = (sx, sy, ex, ey)

            if box_match and min_d < 3600:
                cv2.rectangle(frame, (box_match[0], box_match[1]), (box_match[2], box_match[3]), color, 2)
                cv2.putText(
                    frame, f"Egg #{object_id}", (box_match[0], box_match[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
            else:
                cv2.circle(frame, (centroid[0], centroid[1]), 6, color, -1)
                cv2.putText(
                    frame, f"Egg #{object_id}", (centroid[0] - 15, centroid[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

            pts = trajectories.get(object_id, [])
            for j in range(1, len(pts)):
                if pts[j - 1] is None or pts[j] is None:
                    continue
                cv2.line(frame, tuple(pts[j - 1]), tuple(pts[j]), color, 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    input_path = "data/video/input_video.mp4"
    output_path = "outputs/tracked_output.mp4"
    if os.path.exists(input_path):
        process_video(input_path, output_path)