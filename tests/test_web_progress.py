from app.web import channel_progress_payload, progress_item, progress_payload


def test_progress_payload_maps_known_stage() -> None:
    payload = progress_payload("fetching_captions", "Checking captions")

    assert payload["percent"] == 30
    assert payload["label"] == "Fetching Captions"
    assert payload["detail"] == "Checking captions"


def test_channel_progress_averages_video_items() -> None:
    items = [
        progress_item("One", "complete", "Done", "succeeded", percent=100),
        progress_item("Two", "fetching_captions", "Checking captions", "running"),
    ]

    payload = channel_progress_payload(items)

    assert payload["percent"] == 65
    assert payload["stage"] == "fetching_captions"
    assert "One" not in payload["detail"]
    assert "Two" in payload["detail"]


def test_channel_progress_empty_is_complete() -> None:
    payload = channel_progress_payload([])

    assert payload["percent"] == 100
    assert payload["stage"] == "complete"
