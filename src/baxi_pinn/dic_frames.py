"""Exclude documented postfracture frames without modifying the raw recordings."""
import json
from pathlib import Path


FRAME_EXCLUSIONS_PATH = (Path(__file__).resolve().parents[2]
                         / "rebuild_data/source_backbone/dic_channels/frame_exclusions.json")


def load_frame_exclusions():
    return json.loads(FRAME_EXCLUSIONS_PATH.read_text(encoding="utf-8"))


def frame_is_excluded(specimen, frame, exclusions):
    record = exclusions.get(specimen)
    return record is not None and int(frame) >= record["first_excluded_frame"]
