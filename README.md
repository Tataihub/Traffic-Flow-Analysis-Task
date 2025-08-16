"""
Traffic Flow Analysis - YOLOv5 + SORT
=====================================

Files included here:
- traffic_flow_yolov5_sort.py  (this script)

What this script does (full pipeline):
1. Downloads the YouTube video provided in the assignment (using pytube).
2. Loads a YOLOv5 detector from torch.hub (ultralytics/yolov5).
3. Runs detection on each frame, filters for vehicle classes (car, truck, bus, motorcycle).
4. Tracks detections using a lightweight SORT implementation (Kalman filter + Hungarian assignment).
5. Defines 3 vertical lanes (left, middle, right) by dividing frame width into 3 equal regions — you can change this to polygonal lanes if needed.
6. Counts each vehicle once when it crosses a counting line (a horizontal line placed at configurable Y). Counts are maintained per-lane.
7. Saves an overlayed output video, and exports a CSV with columns: VehicleID, Lane, FrameCountSeen, Timestamp_seconds

NOTES / LIMITATIONS
- This implementation aims to be reproducible and easy to run on standard hardware. It's not as robust as DeepSORT (no appearance embedding), but it satisfies the SORT requirement and avoids duplicate counts by counting on a crossing event.
- For real deployments calibrate lane polygons and counting line location according to camera perspective.

Dependencies (install with pip):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install opencv-python numpy pandas pytube filterpy scipy tqdm

If you have CUDA, install torch with the appropriate CUDA wheels for your system.

Usage:
    python traffic_flow_yolov5_sort.py --youtube_url "https://www.youtube.com/watch?v=MNn9qKG2UFI" \
        --output_video output_overlay.mp4 --output_csv counts.csv --device cpu

Configuration flags and defaults are inside the script (see argparse defaults).

"""

import os
import sys
import math
import time
import argparse
import tempfile
from pathlib import Path
import csv

import cv2
import numpy as np
import pandas as pd
from pytube import YouTube
import torch
from tqdm import tqdm

# SORT related imports
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


# ------------------------
# Simple SORT implementation (adapted, compact)
# ------------------------

def iou(bb_test, bb_gt):
    """ Computes IOU between two bboxes in [x1,y1,x2,y2] format
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area_a = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_b = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    o = inter / (area_a + area_b - inter + 1e-6)
    return o


class KalmanBoxTracker:
    """A single object tracker using a Kalman filter for bounding box states.
    State: [x, y, s, r, vx, vy, vs]
    where x,y - center, s - scale (area), r - aspect ratio
    """
    count = 0

    def __init__(self, bbox):
        # bbox: [x1,y1,x2,y2]
        # convert to center format
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w/2.
        y = y1 + h/2.
        s = w * h
        r = w / float(h + 1e-6)

        # 7 dim state
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # state transition
        self.kf.F = np.eye(7)
        for i in range(3):
            self.kf.F[i, i+4] = 1.0
        # measurement mapping
        self.kf.H = np.zeros((4,7))
        self.kf.H[0,0] = 1.0
        self.kf.H[1,1] = 1.0
        self.kf.H[2,2] = 1.0
        self.kf.H[3,3] = 1.0

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.x[:4] = np.array([x, y, s, r]).reshape((4,1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        # bbox: [x1,y1,x2,y2]
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w/2.
        y = y1 + h/2.
        s = w * h
        r = w / float(h + 1e-6)
        z = np.array([x, y, s, r]).reshape((4,1))
        self.kf.update(z)

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x.copy())
        return self.get_state()

    def get_state(self):
        x = self.kf.x
        cx = float(x[0])
        cy = float(x[1])
        s = float(x[2])
        r = float(x[3])
        w = math.sqrt(abs(s * r))
        h = s / (w + 1e-6)
        x1 = cx - w/2.
        y1 = cy - h/2.
        x2 = cx + w/2.
        y2 = cy + h/2.
        return [x1, y1, x2, y2]


class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets=np.empty((0,5))):
        # dets: N x [x1,y1,x2,y2,score]
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t][:4] = pos
            trks[t][4] = 0
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)

        if dets.shape[0] > 0:
            iou_matrix = np.zeros((dets.shape[0], len(self.trackers)), dtype=np.float32)
            for d, det in enumerate(dets):
                for t, trk in enumerate(self.trackers):
                    iou_matrix[d, t] = iou(det[:4], trk.get_state())

            if iou_matrix.size > 0:
                matched_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(matched_indices).T
            else:
                matched_indices = np.empty((0,2), dtype=int)

            unmatched_dets = []
            for d in range(dets.shape[0]):
                if d not in matched_indices[:,0]:
                    unmatched_dets.append(d)
            unmatched_trks = []
            for t in range(len(self.trackers)):
                if t not in matched_indices[:,1]:
                    unmatched_trks.append(t)

            # filter out matches with low IOU
            matches = []
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] < self.iou_threshold:
                    unmatched_dets.append(m[0])
                    unmatched_trks.append(m[1])
                else:
                    matches.append(m.reshape(2))
            matches = np.array(matches) if len(matches) > 0 else np.empty((0,2), dtype=int)

        else:
            unmatched_dets = []
            unmatched_trks = list(range(len(self.trackers)))
            matches = np.empty((0,2), dtype=int)

        # update matched trackers with assigned detections
        for m in matches:
            det_idx, trk_idx = int(m[0]), int(m[1])
            self.trackers[trk_idx].update(dets[det_idx,:4])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:4])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or trk.age <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0,5))


# ------------------------
# Helper utilities
# ------------------------

VEHICLE_CLASSES = {2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck'}  # COCO class indices (common for yolov5)


def download_youtube_video(youtube_url, out_path):
    print(f"Downloading video from: {youtube_url}")
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if stream is None:
        raise RuntimeError("No suitable mp4 stream found for the provided YouTube URL.")
    out_file = stream.download(output_path=out_path)
    print(f"Downloaded to: {out_file}")
    return out_file


def load_model(device='cpu', model_name='yolov5s'):
    # uses torch.hub to load ultralytics yolov5
    print(f"Loading YOLOv5 model '{model_name}' on device '{device}'...")
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    model.to(device)
    model.eval()
    return model


def detect_frame(model, frame, device='cpu', conf_thres=0.4):
    # model expects RGB images
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img, size=640)
    # results.xyxy[0] -> [x1,y1,x2,y2,conf,class]
    dets = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        cls = int(cls)
        if cls in VEHICLE_CLASSES and conf >= conf_thres:
            x1,y1,x2,y2 = box
            dets.append([x1,y1,x2,y2,conf])
    if len(dets) == 0:
        return np.empty((0,5))
    return np.array(dets)


# ------------------------
# Main processing
# ------------------------

def process_video(video_path, output_video_path, output_csv_path, device='cpu', show=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video file: " + str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"Video opened: {video_path} (fps={fps}, size={width}x{height}, frames={total_frames})")

    # prepare output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # load detector model
    model = load_model(device=device)

    # tracker
    tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)

    # lane boundaries -> 3 equal vertical lanes by default
    lane_bounds = [0, width//3, 2*width//3, width]
    lane_labels = {0: 'Left', 1: 'Center', 2: 'Right'}

    # counting line y coordinate (horizontal) - count when centroid crosses this line moving downward
    counting_line_y = int(height * 0.6)

    # bookkeeping
    counted_ids = set()
    per_lane_counts = {0:0, 1:0, 2:0}
    track_last_centroid = {}
    track_frame_seen = {}

    csv_rows = []

    frame_idx = 0
    pbar = tqdm(total=total_frames if total_frames>0 else None)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            dets = detect_frame(model, frame, device=device, conf_thres=0.4)

            # update tracker
            tracked = tracker.update(dets)
            # tracked: N x [x1,y1,x2,y2,id]

            # draw lanes
            overlay = frame.copy()
            # vertical lane separators
            for i in range(1, len(lane_bounds)-1):
                x = lane_bounds[i]
                cv2.line(overlay, (x,0), (x,height), (255,255,0), 2)
            # counting line
            cv2.line(overlay, (0, counting_line_y), (width, counting_line_y), (0,0,255), 2)

            # draw detections & trackers
            for d in tracked:
                x1,y1,x2,y2,tid = d
                tid = int(tid)
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                # determine lane by centroid x
                lane_idx = None
                for li in range(len(lane_bounds)-1):
                    if cx >= lane_bounds[li] and cx < lane_bounds[li+1]:
                        lane_idx = li
                        break
                if lane_idx is None:
                    lane_idx = len(lane_bounds)-2

                # draw box and id
                cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                cv2.putText(overlay, f"ID {tid} L{lane_idx}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.circle(overlay, (cx, cy), 3, (0,255,255), -1)

                # crossing detection: count when centroid crosses counting_line_y from above to below
                last = track_last_centroid.get(tid, None)
                if last is not None:
                    ly = last[1]
                    if (ly < counting_line_y) and (cy >= counting_line_y) and (tid not in counted_ids):
                        # count into this lane
                        per_lane_counts[lane_idx] += 1
                        counted_ids.add(tid)
                        # record csv row: Vehicle ID, Lane number, Frame count (frames seen), Timestamp
                        frames_seen = track_frame_seen.get(tid, 1)
                        timestamp = frame_idx / fps
                        csv_rows.append({
                            'VehicleID': tid,
                            'Lane': lane_idx,
                            'FrameCountSeen': frames_seen,
                            'Timestamp_s': round(timestamp, 3)
                        })

                # update last centroid and frame seen
                track_last_centroid[tid] = (cx, cy)
                track_frame_seen[tid] = track_frame_seen.get(tid, 0) + 1

            # overlay counts
            y0 = 30
            for li in range(3):
                text = f"Lane {li} ({lane_labels[li]}): {per_lane_counts[li]}"
                cv2.putText(overlay, text, (10, y0 + li*25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            out_writer.write(overlay)
            if show:
                cv2.imshow('overlay', overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()

    # save csv
    df = pd.DataFrame(csv_rows)
    if len(df) == 0:
        # if no counts happened, still output empty csv with columns
        df = pd.DataFrame(columns=['VehicleID','Lane','FrameCountSeen','Timestamp_s'])
    df.to_csv(output_csv_path, index=False)

    # print summary
    print('\nProcessing complete.')
    print('Total per-lane counts:')
    for li in range(3):
        print(f"  Lane {li} ({lane_labels[li]}): {per_lane_counts[li]}")

    return {
        'per_lane_counts': per_lane_counts,
        'csv_path': output_csv_path,
        'output_video': output_video_path
    }


# ------------------------
# CLI
# ------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traffic Flow Analysis (YOLOv5 + SORT)')
    parser.add_argument('--youtube_url', type=str, required=False, default='https://www.youtube.com/watch?v=MNn9qKG2UFI', help='YouTube video URL')
    parser.add_argument('--input_video', type=str, default=None, help='Path to local video (if provided, will skip downloading)')
    parser.add_argument('--output_video', type=str, default='output_overlay.mp4', help='Output overlay video path')
    parser.add_argument('--output_csv', type=str, default='counts.csv', help='Output CSV path')
    parser.add_argument('--device', type=str, default='cpu', help='Torch device: cpu or cuda')
    parser.add_argument('--show', action='store_true', help='Show live preview window')
    parser.add_argument('--tmp_dir', type=str, default=None, help='Temporary directory for downloads')

    args = parser.parse_args()

    if args.input_video is None:
        tmp_dir = args.tmp_dir or tempfile.gettempdir()
        try:
            video_path = download_youtube_video(args.youtube_url, out_path=tmp_dir)
        except Exception as e:
            print('Video download failed:', e)
            sys.exit(1)
    else:
        video_path = args.input_video

    os.makedirs(os.path.dirname(args.output_video) or '.', exist_ok=True)
    res = process_video(video_path, args.output_video, args.output_csv, device=args.device, show=args.show)
    print('\nSaved outputs:')
    print(' - Overlay video:', res['output_video'])
    print(' - CSV file:', res['csv_path'])

    print('\nDone.')
