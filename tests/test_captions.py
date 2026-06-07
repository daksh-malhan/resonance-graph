from app.captions import parse_vtt_transcript, select_caption_track


def test_parse_vtt_transcript() -> None:
    vtt = """WEBVTT

00:00:01.000 --> 00:00:03.500
<c>Hello</c> &amp; welcome

00:00:04.000 --> 00:00:06.000 align:start
to the transcript.
"""

    transcript = parse_vtt_transcript(vtt, "vid")

    assert transcript.source == "youtube_caption"
    assert len(transcript.segments) == 2
    assert transcript.segments[0].segment_id == "vid:ytcap:000000"
    assert transcript.segments[0].start_time == 1.0
    assert transcript.segments[0].end_time == 3.5
    assert transcript.segments[0].text == "Hello & welcome"
    assert transcript.segments[0].source == "youtube_caption"


def test_select_caption_track_prefers_manual_english_vtt() -> None:
    info = {
        "subtitles": {
            "en": [
                {"ext": "srv3", "url": "https://example.com/manual.srv3"},
                {"ext": "vtt", "url": "https://example.com/manual.vtt"},
            ]
        },
        "automatic_captions": {
            "en": [{"ext": "vtt", "url": "https://example.com/auto.vtt"}],
        },
    }

    track = select_caption_track(info)

    assert track == {
        "url": "https://example.com/manual.vtt",
        "language": "en",
        "kind": "manual",
    }


def test_select_caption_track_falls_back_to_english_auto() -> None:
    info = {
        "subtitles": {},
        "automatic_captions": {
            "en-US": [{"ext": "vtt", "url": "https://example.com/auto.vtt"}],
        },
    }

    track = select_caption_track(info)

    assert track["url"] == "https://example.com/auto.vtt"
    assert track["kind"] == "auto"
