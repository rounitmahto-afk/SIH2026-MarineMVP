from ml.tracking import (
    DeterministicTracker,
    TrackDetection,
)


def detection(
    frame_index: int,
    detection_index: int,
    confidence: float,
    bbox: tuple[float, float, float, float],
) -> TrackDetection:
    return TrackDetection(
        frame_index=frame_index,
        detection_index=detection_index,
        class_id=0,
        class_name="Crab-Pot",
        confidence=confidence,
        bbox_xyxy=bbox,
    )


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    tracker = DeterministicTracker(
        max_center_distance=80.0,
        min_iou=0.05,
        max_frame_gap=1,
    )

    print("[1] Frame 0: create first track")

    tracks = tracker.update(
        0,
        [
            detection(
                0,
                0,
                0.70,
                (100.0, 100.0, 140.0, 140.0),
            )
        ],
    )

    assert_true(
        len(tracks) == 1,
        "Expected exactly one track.",
    )

    assert_true(
        tracks[0].track_id == 1,
        "Expected track ID 1.",
    )

    assert_true(
        tracks[0].detection_count == 1,
        "Expected one detection.",
    )

    print("[PASS] Track 1 created.")

    print("")
    print("[2] Frame 1: same object moves slightly")

    tracks = tracker.update(
        1,
        [
            detection(
                1,
                0,
                0.80,
                (106.0, 103.0, 146.0, 143.0),
            )
        ],
    )

    assert_true(
        len(tracks) == 1,
        "Expected one continuing track.",
    )

    assert_true(
        tracks[0].track_id == 1,
        "Object should remain on track 1.",
    )

    assert_true(
        tracks[0].detection_count == 2,
        "Expected two detections on track 1.",
    )

    print("[PASS] Same object associated correctly.")

    print("")
    print("[3] Frame 2: new distant object appears")

    tracks = tracker.update(
        2,
        [
            detection(
                2,
                0,
                0.82,
                (112.0, 106.0, 152.0, 146.0),
            ),
            detection(
                2,
                1,
                0.91,
                (500.0, 500.0, 540.0, 540.0),
            ),
        ],
    )

    assert_true(
        len(tracks) == 2,
        "Expected two tracks.",
    )

    track_ids = {
        track.track_id
        for track in tracks
    }

    assert_true(
        track_ids == {1, 2},
        "Expected track IDs 1 and 2.",
    )

    print("[PASS] New distant object created as Track 2.")

    print("")
    print("[4] Frame 3: first track persists")

    tracks = tracker.update(
        3,
        [
            detection(
                3,
                0,
                0.85,
                (118.0, 109.0, 158.0, 149.0),
            )
        ],
    )

    track_1 = next(
        track
        for track in tracks
        if track.track_id == 1
    )

    assert_true(
        track_1.detection_count == 4,
        "Track 1 should contain four detections.",
    )

    assert_true(
        track_1.persistence_score == 0.8,
        "Unexpected persistence score.",
    )

    print("[PASS] Persistence updated correctly.")

    print("")
    print("[5] Checking serialized track")

    serialized = track_1.to_dict()

    assert_true(
        serialized["track_id"] == 1,
        "Serialized track ID mismatch.",
    )

    assert_true(
        serialized["dominant_class"] == "Crab-Pot",
        "Serialized dominant class mismatch.",
    )

    print("[PASS] Serialization works.")

    print("")
    print("============================================================")
    print("TRACKER UNIT TEST PASSED")
    print("============================================================")

    for track in tracker.tracks():
        print(track.to_dict())


if __name__ == "__main__":
    main()
