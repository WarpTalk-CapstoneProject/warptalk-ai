from livekit_ingress_worker.worker import _parse_track_published_event


def test_parse_track_published_event_accepts_versioned_envelope() -> None:
    parsed = _parse_track_published_event(
        {
            "event_id": "019fa3b8-e332-7dd6-a9d8-6c6a1faf32fd",
            "event_type": "meeting.track_published",
            "schema_version": 1,
            "occurred_at": "2026-07-27T16:00:00Z",
            "producer": "meeting-service",
            "correlation_id": None,
            "causation_id": None,
            "workspace_id": None,
            "payload": {
                "room_name": "room-123",
                "participant_identity": "user-123",
                "track_id": "TR_123",
                "published_at": "2026-07-27T16:00:00Z",
            },
        }
    )

    assert parsed == ("room-123", "user-123", "TR_123")


def test_parse_track_published_event_rejects_legacy_payload() -> None:
    assert (
        _parse_track_published_event(
            {
                "RoomName": "room-123",
                "ParticipantIdentity": "user-123",
                "TrackId": "TR_123",
            }
        )
        is None
    )
