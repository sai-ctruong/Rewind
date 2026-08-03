from pathlib import Path

import cv2
import numpy as np

from aic2026.local_refinement import LocalFrameRefiner, RefinementConfig
from ingestion.build_index import KeyframeIndex
from ingestion.schemas import RawKeyframe
from retrieval.video_engine import VideoIndexEntry


def entry_for(raws):
    index = KeyframeIndex(
        ids=list(raws), video_ids=[raws[k].video_id for k in raws],
        timestamps=[raws[k].timestamp for k in raws], objects=[[] for _ in raws],
    )
    return VideoIndexEntry("dataset", index, raws, len(raws), len(raws))


def test_missing_video_uses_keyframe_only_neighbors() -> None:
    raws = {
        f"V/{frame}": RawKeyframe(f"V/{frame}", "V", float(i), frame_idx=frame)
        for i, frame in enumerate((10, 40, 90))
    }
    result = LocalFrameRefiner().refine(entry_for(raws), "V/40")
    assert result.refinement_mode == "keyframe_only"
    assert result.evidence_frame_ids == ("10", "40", "90")
    assert result.warning


def test_mp4_refinement_is_bounded_and_selects_bright_frame(tmp_path: Path) -> None:
    video = tmp_path / "V.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    for i in range(15):
        writer.write(np.full((24, 32, 3), 255 if i == 8 else 10, dtype=np.uint8))
    writer.release()
    raw = RawKeyframe("V/5", "V", 1.0, source_video=str(video), frame_idx=5)
    config = RefinementConfig(window_before_s=1, window_after_s=2, fine_fps=5, max_frames=10)
    result = LocalFrameRefiner(config).refine(
        entry_for({raw.id: raw}), raw.id,
        frame_scorer=lambda frames: [float(frame.mean()) for frame in frames],
    )
    assert result.refinement_mode == "mp4"
    assert result.decoded_frame_count <= 10
    assert int(result.selected_frame_id) == 8
