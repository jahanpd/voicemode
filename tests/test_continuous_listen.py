"""
Tests for continuous listen mode and session management.
"""

import pytest
import time
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock
import queue
import threading


class TestEndPhraseDetection:
    """Test end phrase matching logic."""

    def test_exact_end_phrase(self):
        from voice_mode.tools.converse import _check_end_phrase
        matched, cleaned = _check_end_phrase("I think that's everything over and out", ["over and out", "I'm done"])
        assert matched is True
        assert "over and out" not in cleaned.lower()
        assert "everything" in cleaned.lower()

    def test_end_phrase_with_punctuation(self):
        from voice_mode.tools.converse import _check_end_phrase
        matched, cleaned = _check_end_phrase("That's all.", ["that's all"])
        assert matched is True

    def test_end_phrase_case_insensitive(self):
        from voice_mode.tools.converse import _check_end_phrase
        matched, cleaned = _check_end_phrase("I'M DONE", ["i'm done"])
        assert matched is True

    def test_no_end_phrase(self):
        from voice_mode.tools.converse import _check_end_phrase
        matched, cleaned = _check_end_phrase("I want to keep talking about this", ["over and out", "I'm done"])
        assert matched is False
        assert cleaned == "I want to keep talking about this"

    def test_empty_text(self):
        from voice_mode.tools.converse import _check_end_phrase
        matched, cleaned = _check_end_phrase("", ["over and out"])
        assert matched is False

    def test_end_phrase_is_entire_text(self):
        from voice_mode.tools.converse import _check_end_phrase
        matched, cleaned = _check_end_phrase("Over and out", ["over and out"])
        assert matched is True
        assert cleaned == ""

    def test_end_phrase_with_trailing_punctuation_variations(self):
        from voice_mode.tools.converse import _check_end_phrase
        matched, cleaned = _check_end_phrase("That is all!", ["that is all"])
        assert matched is True

        matched, cleaned = _check_end_phrase("I'm done...", ["i'm done"])
        assert matched is True


class TestSessionManagement:
    """Test session creation, retrieval, and expiry."""

    def setup_method(self):
        """Clear sessions before each test."""
        from voice_mode.tools import converse as converse_mod
        converse_mod._active_sessions.clear()

    def test_create_session(self):
        from voice_mode.tools.converse import _create_session, _get_session
        sid = _create_session(
            listen_mode="continuous",
            end_phrases=["over and out"],
            vad_aggressiveness=3,
            listen_duration_max=120.0,
            listen_duration_min=2.0,
        )
        assert sid is not None
        assert len(sid) == 12

        session = _get_session(sid)
        assert session is not None
        assert session["listen_mode"] == "continuous"
        assert session["end_phrases"] == ["over and out"]
        assert session["transcript_segments"] == []
        assert session["total_recording_time"] == 0.0

    def test_get_missing_session(self):
        from voice_mode.tools.converse import _get_session
        assert _get_session("nonexistent") is None

    def test_close_session(self):
        from voice_mode.tools.converse import _create_session, _close_session, _get_session
        sid = _create_session(
            listen_mode="continuous",
            end_phrases=[],
            vad_aggressiveness=None,
            listen_duration_max=120.0,
            listen_duration_min=2.0,
        )
        closed = _close_session(sid)
        assert closed is not None
        assert _get_session(sid) is None

    def test_session_expiry(self):
        from voice_mode.tools import converse as converse_mod
        from voice_mode.tools.converse import _create_session, _cleanup_expired_sessions, _get_session

        sid = _create_session(
            listen_mode="continuous",
            end_phrases=[],
            vad_aggressiveness=None,
            listen_duration_max=120.0,
            listen_duration_min=2.0,
        )
        # Manually set last_active to the past
        converse_mod._active_sessions[sid]["last_active"] = time.time() - 200
        _cleanup_expired_sessions()
        assert _get_session(sid) is None

    def test_session_stores_segments(self):
        from voice_mode.tools.converse import _create_session, _get_session
        sid = _create_session(
            listen_mode="continuous",
            end_phrases=["done"],
            vad_aggressiveness=3,
            listen_duration_max=120.0,
            listen_duration_min=2.0,
        )
        session = _get_session(sid)
        session["transcript_segments"].append("First segment.")
        session["transcript_segments"].append("Second segment.")
        session["total_recording_time"] = 15.0

        # Retrieve again and verify
        session2 = _get_session(sid)
        assert len(session2["transcript_segments"]) == 2
        assert session2["total_recording_time"] == 15.0


class TestContinuousRecordingWorker:
    """Test the continuous recording worker function."""

    def test_worker_sends_sentinel_on_completion(self):
        """Worker should always send None sentinel when done."""
        from voice_mode.tools.converse import _continuous_recording_worker

        seg_queue = queue.Queue()
        stop_event = threading.Event()
        stop_event.set()  # Immediately signal stop

        with patch('voice_mode.tools.converse.VAD_AVAILABLE', False):
            with patch('voice_mode.tools.converse.record_audio') as mock_record:
                mock_record.return_value = np.zeros(1000, dtype=np.int16)
                _continuous_recording_worker(
                    max_duration=5.0,
                    min_segment_duration=0.5,
                    vad_aggressiveness=3,
                    segment_queue=seg_queue,
                    stop_event=stop_event,
                    abort_time=time.time() + 60,
                )

        # Should have at least the sentinel
        items = []
        while not seg_queue.empty():
            items.append(seg_queue.get_nowait())
        assert items[-1] is None  # Sentinel

    def test_worker_respects_abort_time(self):
        """Worker should stop when abort_time is reached."""
        from voice_mode.tools.converse import _continuous_recording_worker

        seg_queue = queue.Queue()
        stop_event = threading.Event()

        with patch('voice_mode.tools.converse.VAD_AVAILABLE', False):
            with patch('voice_mode.tools.converse.record_audio') as mock_record:
                mock_record.return_value = np.zeros(1000, dtype=np.int16)
                # Set abort_time to now so it stops immediately
                _continuous_recording_worker(
                    max_duration=60.0,
                    min_segment_duration=0.5,
                    vad_aggressiveness=3,
                    segment_queue=seg_queue,
                    stop_event=stop_event,
                    abort_time=time.time() - 1,  # Already past
                )

        # Should get sentinel (worker completed)
        items = []
        while not seg_queue.empty():
            items.append(seg_queue.get_nowait())
        assert None in items


class TestRunContinuousListen:
    """Test the async orchestrator."""

    @pytest.mark.asyncio
    async def test_end_phrase_stops_listening(self):
        """When an end phrase is transcribed, listening should stop."""
        from voice_mode.tools.converse import _run_continuous_listen

        fake_audio = np.zeros(24000, dtype=np.int16)  # 1s of silence

        # Mock the recording worker to produce two segments
        def mock_worker(max_dur, min_seg, vad_agg, seg_queue, stop_event, abort_time):
            seg_queue.put((fake_audio, 5.0))
            # Wait for stop_event to be set by orchestrator after end phrase
            stop_event.wait(timeout=5.0)
            seg_queue.put(None)

        # Mock STT to return text with end phrase on first segment
        async def mock_stt(audio_data, save_audio, audio_dir, transport):
            return {"text": "Here is my explanation. I'm done."}

        with patch('voice_mode.tools.converse._continuous_recording_worker', side_effect=mock_worker):
            with patch('voice_mode.tools.converse.speech_to_text', side_effect=mock_stt):
                with patch('voice_mode.tools.converse.SAVE_AUDIO', False):
                    with patch('voice_mode.tools.converse.AUDIO_DIR', None):
                        result = await _run_continuous_listen(
                            timeout_deadline=time.time() + 30,
                            listen_duration_max=120.0,
                            listen_duration_min=0.5,
                            vad_aggressiveness=3,
                            end_phrases=["i'm done"],
                        )

        assert result["ended_by"] == "end_phrase"
        assert len(result["transcript_segments"]) >= 1
        # End phrase should be stripped from transcript
        full_text = " ".join(result["transcript_segments"])
        assert "i'm done" not in full_text.lower()

    @pytest.mark.asyncio
    async def test_timeout_creates_partial_result(self):
        """When timeout approaches, should return partial transcripts."""
        from voice_mode.tools.converse import _run_continuous_listen

        fake_audio = np.zeros(24000, dtype=np.int16)

        def mock_worker(max_dur, min_seg, vad_agg, seg_queue, stop_event, abort_time):
            seg_queue.put((fake_audio, 5.0))
            # Simulate timeout by waiting until stop_event
            stop_event.wait(timeout=5.0)
            seg_queue.put(None)

        call_count = 0
        async def mock_stt(audio_data, save_audio, audio_dir, transport):
            nonlocal call_count
            call_count += 1
            return {"text": f"Segment {call_count} of my long speech."}

        with patch('voice_mode.tools.converse._continuous_recording_worker', side_effect=mock_worker):
            with patch('voice_mode.tools.converse.speech_to_text', side_effect=mock_stt):
                with patch('voice_mode.tools.converse.SAVE_AUDIO', False):
                    with patch('voice_mode.tools.converse.AUDIO_DIR', None):
                        result = await _run_continuous_listen(
                            timeout_deadline=time.time() + 2,  # Very short timeout
                            listen_duration_max=120.0,
                            listen_duration_min=0.5,
                            vad_aggressiveness=3,
                            end_phrases=["over and out"],
                        )

        assert result["ended_by"] in ("timeout", "silence")
        assert len(result["transcript_segments"]) >= 1

    @pytest.mark.asyncio
    async def test_session_resume_preserves_prior_segments(self):
        """Resumed sessions should include prior transcript segments."""
        from voice_mode.tools.converse import _run_continuous_listen

        fake_audio = np.zeros(24000, dtype=np.int16)

        def mock_worker(max_dur, min_seg, vad_agg, seg_queue, stop_event, abort_time):
            seg_queue.put((fake_audio, 3.0))
            stop_event.wait(timeout=2.0)
            seg_queue.put(None)

        async def mock_stt(audio_data, save_audio, audio_dir, transport):
            return {"text": "New segment. I'm done."}

        prior_session = {
            "transcript_segments": ["First part from before."],
            "total_recording_time": 10.0,
            "listen_mode": "continuous",
            "end_phrases": ["i'm done"],
            "vad_aggressiveness": 3,
            "listen_duration_max": 120.0,
            "listen_duration_min": 0.5,
        }

        with patch('voice_mode.tools.converse._continuous_recording_worker', side_effect=mock_worker):
            with patch('voice_mode.tools.converse.speech_to_text', side_effect=mock_stt):
                with patch('voice_mode.tools.converse.SAVE_AUDIO', False):
                    with patch('voice_mode.tools.converse.AUDIO_DIR', None):
                        result = await _run_continuous_listen(
                            timeout_deadline=time.time() + 30,
                            listen_duration_max=120.0,
                            listen_duration_min=0.5,
                            vad_aggressiveness=3,
                            end_phrases=["i'm done"],
                            session=prior_session,
                        )

        assert result["ended_by"] == "end_phrase"
        # Should have prior segment + new segment
        assert len(result["transcript_segments"]) >= 2
        assert "First part from before." in result["transcript_segments"]


class TestConfigDefaults:
    """Test that config constants are properly defined."""

    def test_end_phrases_config(self):
        from voice_mode.config import END_PHRASES
        assert isinstance(END_PHRASES, list)
        assert len(END_PHRASES) > 0
        assert "over and out" in [p.lower() for p in END_PHRASES]

    def test_timeout_safety_margin_config(self):
        from voice_mode.config import TIMEOUT_SAFETY_MARGIN
        assert isinstance(TIMEOUT_SAFETY_MARGIN, float)
        assert TIMEOUT_SAFETY_MARGIN > 0
        assert TIMEOUT_SAFETY_MARGIN < 30  # Sanity check
