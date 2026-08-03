from dataclasses import dataclass

from aic2026.ranking import RankingConfig, video_aware_top100


@dataclass(frozen=True)
class Item:
    video: str
    frame: int
    score: float


def test_top100_is_unique_diverse_bounded_and_keeps_rank_one() -> None:
    candidates = [Item("A", i * 30, 100 - i) for i in range(120)]
    candidates += [Item("B", i * 30, 50 - i) for i in range(20)]
    ranked = video_aware_top100(
        candidates,
        video_id=lambda x: x.video,
        frame_id=lambda x: x.frame,
        score=lambda x: x.score,
        config=RankingConfig(top_k=100, max_frames_per_video=60, min_frame_gap=20),
    )
    assert ranked[0] == candidates[0]
    assert len(ranked) <= 100
    assert len({(x.video, x.frame) for x in ranked}) == len(ranked)
    assert {x.video for x in ranked} == {"A", "B"}
    assert sum(x.video == "A" for x in ranked) <= 60


def test_neighbor_expansion_adds_expected_frame() -> None:
    anchor = Item("V", 100, 1.0)
    ranked = video_aware_top100(
        [anchor], video_id=lambda x: x.video, frame_id=lambda x: x.frame, score=lambda x: x.score,
        neighbors=lambda item, offsets: [Item(item.video, item.frame + offsets[0], 0.9)],
        config=RankingConfig(top_k=2, max_frames_per_video=3, neighbor_offsets=(10,), min_frame_gap=30),
    )
    assert [item.frame for item in ranked] == [100, 110]
