"""
Baby sleep / wake tracking with debounced state detection and SQLite persistence.

States:
  - sleeping: baby visible, lying down, low sustained movement
  - awake: sitting/standing OR sustained high movement
  - out_of_frame: no reliable person detection

Single brief movements (roll, leg kick) do not flip awake thanks to:
  - rolling activity index over ~30 s
  - minimum dwell time before any state change (especially to awake)
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

# COCO keypoints used for visibility and motion
CORE_KP = (5, 6, 11, 12, 13, 14)
ALL_KP = tuple(range(17))

STATE_SLEEPING = "sleeping"
STATE_AWAKE = "awake"
STATE_OUT = "out_of_frame"
VALID_STATES = (STATE_SLEEPING, STATE_AWAKE, STATE_OUT)

# Motion thresholds (pixels on 640x480 frame)
MOVEMENT_LOW = 5.0
MOVEMENT_HIGH = 18.0

# Seconds candidate state must hold before switching
CONFIRM_SEC = {
    STATE_SLEEPING: 10,
    STATE_AWAKE: 5,
    STATE_OUT: 15,
}

ACTIVITY_WINDOW_SEC = 30
MIN_VISIBLE_KEYPOINTS = 3
KP_CONFIDENCE = 0.4

from server.paths import SLEEP_DB_PATH, ensure_data_dir

ensure_data_dir()
DB_PATH = SLEEP_DB_PATH


@dataclass
class SleepSegment:
    state: str
    start_ts: float
    end_ts: Optional[float]


class SleepTracker:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.lock = threading.Lock()
        self.state = STATE_OUT
        self.state_since = time.time()
        self.candidate: Optional[str] = None
        self.candidate_since: Optional[float] = None
        self.last_kps: List[Tuple[float, float, float]] = []
        self.motion_samples: Deque[Tuple[float, float]] = deque(maxlen=600)
        self.activity_index = 0.0
        self.last_pose = "unknown"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sleep_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT NOT NULL,
                    start_ts REAL NOT NULL,
                    end_ts REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sleep_start ON sleep_segments(start_ts)"
            )
            conn.commit()

    def update(
        self,
        keypoints: List[Tuple[float, float, float]],
        pose_label: str,
        timestamp: Optional[float] = None,
    ) -> str:
        now = timestamp or time.time()
        with self.lock:
            self.last_pose = pose_label
            instant = self._instant_state(keypoints, pose_label, now)
            self._apply_debounce(instant, now)
            return self.state

    def get_status(self) -> Dict:
        with self.lock:
            return {
                "state": self.state,
                "since": self.state_since,
                "activity_index": round(self.activity_index, 2),
                "pose": self.last_pose,
            }

    def _person_visible(self, kps: List[Tuple[float, float, float]]) -> bool:
        if not kps or len(kps) < 17:
            return False
        confident = sum(1 for i in CORE_KP if kps[i][2] >= KP_CONFIDENCE)
        return confident >= MIN_VISIBLE_KEYPOINTS

    def _body_motion(
        self,
        kps: List[Tuple[float, float, float]],
        prev: List[Tuple[float, float, float]],
    ) -> float:
        if not prev:
            return 0.0
        total = 0.0
        count = 0
        for i in ALL_KP:
            if kps[i][2] >= KP_CONFIDENCE and prev[i][2] >= KP_CONFIDENCE:
                total += math.hypot(kps[i][0] - prev[i][0], kps[i][1] - prev[i][1])
                count += 1
        return total / count if count else 0.0

    def _refresh_activity(self, now: float) -> None:
        cutoff = now - ACTIVITY_WINDOW_SEC
        while self.motion_samples and self.motion_samples[0][0] < cutoff:
            self.motion_samples.popleft()
        if not self.motion_samples:
            self.activity_index = 0.0
            return
        self.activity_index = sum(m for _, m in self.motion_samples) / len(
            self.motion_samples
        )

    def _instant_state(
        self,
        kps: List[Tuple[float, float, float]],
        pose_label: str,
        now: float,
    ) -> str:
        if not self._person_visible(kps):
            self.last_kps = []
            self._refresh_activity(now)
            return STATE_OUT

        motion = self._body_motion(kps, self.last_kps)
        self.last_kps = list(kps)
        self.motion_samples.append((now, motion))
        self._refresh_activity(now)

        if pose_label in ("sitting", "standing"):
            return STATE_AWAKE
        if self.activity_index >= MOVEMENT_HIGH:
            return STATE_AWAKE
        if pose_label == "lying" and self.activity_index <= MOVEMENT_LOW:
            return STATE_SLEEPING
        if pose_label == "lying":
            # Restless but still lying — treat as sleep unless movement stays high
            return STATE_SLEEPING
        return STATE_AWAKE

    def _apply_debounce(self, instant: str, now: float) -> None:
        if instant == self.state:
            self.candidate = None
            self.candidate_since = None
            return

        if self.candidate != instant:
            self.candidate = instant
            self.candidate_since = now
            return

        assert self.candidate_since is not None
        if now - self.candidate_since >= CONFIRM_SEC[instant]:
            self._transition(instant, now)

    def _transition(self, new_state: str, now: float) -> None:
        old_state = self.state
        if old_state == new_state:
            return
        self._close_open_segment(now)
        self._insert_segment(new_state, now)
        self.state = new_state
        self.state_since = now
        self.candidate = None
        self.candidate_since = None

    def _close_open_segment(self, end_ts: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE sleep_segments
                SET end_ts = ?
                WHERE end_ts IS NULL
                """,
                (end_ts,),
            )
            conn.commit()

    def _insert_segment(self, state: str, start_ts: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sleep_segments (state, start_ts, end_ts) VALUES (?, ?, NULL)",
                (state, start_ts),
            )
            conn.commit()

    def get_timeline(self, hours: float = 24.0) -> List[Dict]:
        now = time.time()
        start = now - hours * 3600
        segments: List[Dict] = []

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT state, start_ts, end_ts FROM sleep_segments
                WHERE (end_ts IS NULL OR end_ts > ?) AND start_ts < ?
                ORDER BY start_ts ASC
                """,
                (start, now),
            ).fetchall()

        for state, seg_start, seg_end in rows:
            if state not in VALID_STATES:
                continue
            end = seg_end if seg_end is not None else now
            clip_start = max(seg_start, start)
            clip_end = min(end, now)
            if clip_end <= clip_start:
                continue
            segments.append(
                {"state": state, "start": clip_start, "end": clip_end}
            )
        return segments

    def _minutes_in_state(self, state: str, since_ts: float) -> float:
        now = time.time()
        total = 0.0
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT start_ts, end_ts FROM sleep_segments
                WHERE state = ? AND start_ts >= ?
                """,
                (state, since_ts),
            ).fetchall()
        for seg_start, seg_end in rows:
            end = seg_end if seg_end is not None else now
            total += max(0.0, min(end, now) - max(seg_start, since_ts))
        return total / 60.0

    def get_analytics(self) -> Dict:
        now = time.time()
        day_sleep = self._minutes_in_state(STATE_SLEEPING, now - 86400)
        day_awake = self._minutes_in_state(STATE_AWAKE, now - 86400)
        week_sleep_hours = self._minutes_in_state(STATE_SLEEPING, now - 7 * 86400) / 60.0
        month_sleep_hours = self._minutes_in_state(STATE_SLEEPING, now - 30 * 86400) / 60.0

        return {
            "last_24h_sleep_minutes": round(day_sleep, 1),
            "last_24h_awake_minutes": round(day_awake, 1),
            "week_avg_sleep_hours": round(week_sleep_hours / 7.0, 2),
            "month_avg_sleep_hours": round(month_sleep_hours / 30.0, 2),
        }

    def get_full_report(self) -> Dict:
        return {
            "current": self.get_status(),
            "timeline_24h": self.get_timeline(24.0),
            "summary": self.get_analytics(),
        }


_tracker: Optional[SleepTracker] = None
_tracker_lock = threading.Lock()


def get_sleep_tracker() -> SleepTracker:
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = SleepTracker()
            _tracker._insert_segment(_tracker.state, _tracker.state_since)
        return _tracker
