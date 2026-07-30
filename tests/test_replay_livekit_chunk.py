from tools.replay_livekit_chunk import SUBSCRIBER_SETTLE_SECONDS


def test_replay_waits_for_remote_subscribers_before_sending_audio() -> None:
    assert SUBSCRIBER_SETTLE_SECONDS >= 1.5
