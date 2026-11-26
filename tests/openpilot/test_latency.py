"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The frame-level latency record: the aggregation, the join with the server's
own stamps, and what a phone that sends none still gives the comma.

The numbers are synthetic and the clock is a list, so every stage is asserted
against arithmetic rather than against a machine's speed.
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np

from jetlink import protocol as P
from jetlink.client import JetlinkClient
from jetlink.openpilot import latency
from jetlink.spec import ModelSpec
from jetlink.transport.base import LinkTimeout

SPEC = ModelSpec(sha256='a' * 64, nbytes=1, frame_skip=4,
                 input_shapes={'new_img': (2, 6, 128, 256), 'desire': (8,),
                               'traffic_convention': (1, 2), 'action_t': (1, 2)},
                 output_shapes={'outputs': (1, 16)}, output_slices={'out': slice(0, 16)},
                 checkpoint=None)


class Clock:
  """A clock the test moves by hand; Latency takes one, perf_counter by default."""

  def __init__(self, at: float = 100.0):
    self.at = at

  def __call__(self) -> float:
    return self.at

  def advance(self, seconds: float) -> float:
    self.at += seconds
    return self.at


def one_frame(lat: latency.Latency, frame_id: int, rtt_ms: float, *, stamps=None, held_ms: float | None = None) -> None:
  """A frame with every stage known: warp 1 ms, send 0.5, publish 2, half its
  round trip blocked in the wait, and the server's stamps when given."""
  t0 = 1000.0 + frame_id * 0.05
  t1, t2, sent = t0 + 0.0005, t0 + 0.001, t0 + 0.0015
  reply_at = sent + rtt_ms / 1e3
  lat.sent(frame_id, t0, t1, t2, sent)
  if held_ms is not None:
    lat.held(frame_id, t0 + held_ms / 1e3)
  lat.replied(frame_id, reply_at, waited_from=reply_at - rtt_ms / 2e3)
  lat.published(frame_id, reply_at + 0.002)
  if stamps is not None:
    lat.stamps(frame_id, *stamps)


class TestAggregation(unittest.TestCase):
  """p50/p95/max over a window, and the window clearing when it is read."""

  def test_every_stage_is_its_percentiles(self):
    lat = latency.Latency()
    for i in range(1, 101):
      one_frame(lat, i, rtt_ms=float(i))
    report = lat.report()
    self.assertEqual(report['lat_frames'], 100)
    self.assertEqual(report['lat_stamps'], 'none', 'no stamps: the round trip is all the comma has')
    self.assertEqual((report['warp_p50_ms'], report['warp_p95_ms'], report['warp_max_ms']), (1.0, 1.0, 1.0))
    self.assertEqual((report['send_p50_ms'], report['send_max_ms']), (0.5, 0.5))
    self.assertEqual((report['publish_p50_ms'], report['publish_max_ms']), (2.0, 2.0))
    # frames of 1 to 100 ms: nearest rank, as the server's own FrameStats takes it
    self.assertEqual((report['rtt_p50_ms'], report['rtt_p95_ms'], report['rtt_max_ms']), (51.0, 95.0, 100.0))
    self.assertEqual((report['wait_p50_ms'], report['wait_p95_ms'], report['wait_max_ms']), (25.5, 47.5, 50.0))

  def test_reading_the_window_clears_it(self):
    lat = latency.Latency()
    one_frame(lat, 1, rtt_ms=10)
    self.assertEqual(lat.report()['lat_frames'], 1)
    self.assertEqual(lat.report(), {}, 'nothing happened since')

  def test_an_empty_window_is_no_line_at_all(self):
    self.assertEqual(latency.Latency().report(), {})

  def test_the_window_says_how_long_it_was(self):
    clock = Clock()
    lat = latency.Latency(now=clock)
    one_frame(lat, 1, rtt_ms=10)
    clock.advance(1.25)
    self.assertEqual(lat.report()['lat_window_s'], 1.25)

  def test_a_frame_never_answered_is_dropped_not_kept(self):
    clock = Clock()
    lat = latency.Latency(now=clock)
    lat.sent(1, 100.0, 100.001, 100.002, 100.0025)
    clock.advance(latency.PENDING_S + 1)
    report = lat.report()
    self.assertEqual((report['lat_dropped'], report['lat_frames']), (1, 0))

  def test_a_frame_answered_at_the_last_moment_is_kept(self):
    clock = Clock()
    lat = latency.Latency(now=clock)
    lat.sent(1, 100.0, 100.001, 100.002, 100.0025)
    clock.advance(latency.PENDING_S - 0.1)
    lat.replied(1, clock())
    report = lat.report()
    self.assertEqual((report['lat_dropped'], report['lat_frames']), (0, 1))


class TestTheJoin(unittest.TestCase):
  """The server's stamps, keyed by frame id, split the round trip."""

  def test_the_wire_is_the_round_trip_less_the_phones_own_time(self):
    lat = latency.Latency()
    # queue 1.5 ms, inference 30, reply written in 0.5: 32 ms on the phone
    one_frame(lat, 1, rtt_ms=40.0, stamps=(1500, 30000, 500))
    report = lat.report()
    self.assertEqual(report['lat_stamps'], 'all')
    self.assertEqual(report['lat_stamped'], 1)
    self.assertEqual(report['queue_max_ms'], 1.5)
    self.assertEqual(report['phone_residence_max_ms'], 32.0)
    self.assertEqual(report['wire_max_ms'], 8.0)

  def test_a_frame_the_phone_stamped_before_the_comma_answered_still_joins(self):
    """The stamps come off the stream after the reply, but a drain can read
    them in either order around the report."""
    lat = latency.Latency()
    lat.sent(7, 100.0, 100.001, 100.002, 100.0025)
    lat.stamps(7, 1000, 28000, 1000)
    lat.replied(7, 100.0025 + 0.040)
    report = lat.report()
    self.assertEqual((report['phone_residence_max_ms'], report['wire_max_ms']), (30.0, 10.0))

  def test_a_stamp_that_lost_its_race_with_the_report_is_not_a_second_frame(self):
    lat = latency.Latency()
    one_frame(lat, 1, rtt_ms=40.0)
    self.assertEqual(lat.report()['lat_frames'], 1)
    lat.stamps(1, 1000, 30000, 1000)
    self.assertEqual(lat.report(), {}, 'the frame was reported; the stamp has nothing to join')

  def test_a_held_frames_stamps_join_after_a_later_frame_was_reported(self):
    """A held frame's reply — and its stamps — arrive two frames late, after
    the frames since were reported: its wire is the one worth having."""
    clock = Clock(100.0)
    lat = latency.Latency(now=clock)
    lat.sent(1, 100.0, 100.001, 100.002, 100.0025)
    lat.held(1, 100.046)
    one_frame(lat, 2, rtt_ms=30.0)
    self.assertEqual(lat.report()['lat_frames'], 1)
    lat.replied(1, 100.0575)             # the late reply lands: 55 ms round trip
    lat.stamps(1, 1500, 40000, 500)      # 42 ms of it on the phone
    report = lat.report()
    self.assertEqual((report['lat_frames'], report['lat_held_late']), (1, 1))
    self.assertEqual((report['phone_residence_max_ms'], report['wire_max_ms']), (42.0, 13.0))
    self.assertEqual(report['late_reply_max_ms'], 11.5)

  def test_a_partly_stamped_window_says_so(self):
    lat = latency.Latency()
    one_frame(lat, 1, rtt_ms=40.0, stamps=(1000, 30000, 1000))
    one_frame(lat, 2, rtt_ms=40.0)
    report = lat.report()
    self.assertEqual((report['lat_stamped'], report['lat_frames'], report['lat_stamps']), (1, 2, 'some'))

  def test_a_broken_clock_is_clamped_not_a_negative_stage(self):
    """A phone whose stamps claim more time than the round trip is a phone
    with a clock fault, not a wire that arrived before it was sent."""
    lat = latency.Latency()
    one_frame(lat, 1, rtt_ms=10.0, stamps=(9000, 9000, 9000))
    self.assertEqual(lat.report()['wire_p50_ms'], 0.0)


class TestHeldFrames(unittest.TestCase):
  """The frames that miss HOLD_FRAME: counted, and their late reply measured."""

  def test_a_held_frames_late_reply_is_measured_when_it_lands(self):
    clock = Clock()
    lat = latency.Latency(now=clock)
    t0 = 100.0
    lat.sent(1, t0, t0 + 0.001, t0 + 0.002, t0 + 0.0025)
    lat.held(1, t0 + 0.046)
    # the hold expired with the reply still out: nothing to report yet
    first = lat.report()
    self.assertEqual((first['lat_held'], first['lat_frames']), (1, 0))
    self.assertNotIn('late_reply_max_ms', first)
    # 14 ms later it lands, and the frame is published from the one before it
    lat.replied(1, t0 + 0.060)
    lat.published(1, t0 + 0.061)
    second = lat.report()
    self.assertEqual((second['lat_frames'], second['lat_held_late']), (1, 1))
    self.assertEqual((second['late_reply_p50_ms'], second['late_reply_max_ms']), (14.0, 14.0))
    self.assertEqual(second['rtt_max_ms'], 57.5)

  def test_a_held_frame_that_never_late_replies_shows_only_as_held(self):
    clock = Clock()
    lat = latency.Latency(now=clock)
    lat.sent(1, 100.0, 100.001, 100.002, 100.0025)
    lat.held(1, 100.046)
    report = lat.report()
    self.assertEqual((report['lat_held'], report['lat_held_late']), (1, 0))

  def test_a_frame_the_link_could_not_take_is_held_without_a_reply_to_wait_for(self):
    clock = Clock()
    lat = latency.Latency(now=clock)
    lat.sent(1, 100.0, 100.001, 100.002, None)
    lat.not_sent(1)
    lat.held(1, 100.046)
    first = lat.report()
    self.assertEqual((first['lat_held'], first['lat_not_sent'], first['lat_frames']), (1, 1, 0))
    # and it is not still pending five seconds later: no reply is coming
    clock.advance(latency.PENDING_S + 1)
    self.assertEqual(lat.report(), {})


class Scripted:
  """A transport that hands back queued messages and records what was sent."""

  def __init__(self, messages=()):
    self.messages = list(messages)
    self.sent = []
    self.datagrams = False

  def send(self, msg_type, seq, parts=(), flags=0, timeout=None):
    self.sent.append((msg_type, seq, parts))

  def send_json(self, msg_type, seq, obj, flags=0):
    self.sent.append((msg_type, seq, (json.dumps(obj).encode(),)))

  def recv(self, timeout=None):
    if not self.messages:
      raise LinkTimeout('nothing to read')
    return self.messages.pop(0)


def response(seq: int, frame_id: int) -> SimpleNamespace:
  payload = P.pack_infer_resp(frame_id, P.Status.OK, 0, 0, 20_000) + np.zeros(SPEC.reply_nelem, np.float32).tobytes()
  return SimpleNamespace(msg_type=P.Msg.INFER_RESP, seq=seq, payload=memoryview(payload))


def stamps(frame_id: int, queue_us=1500, infer_us=30_000, send_us=500) -> SimpleNamespace:
  return SimpleNamespace(msg_type=P.Msg.FRAME_STAMPS, seq=0, payload=memoryview(P.pack_frame_stamps(frame_id, queue_us, infer_us, send_us)))


class TestTheClientsSide(unittest.TestCase):
  """What the client asks for, and what it does with what comes back."""

  def client(self, transport) -> JetlinkClient:
    client = JetlinkClient(transport, want_hidden=False)
    client.spec = SPEC
    return client

  def frame(self, client: JetlinkClient, frame_id: int) -> None:
    seq = client.infer_begin(np.zeros(SPEC.warped_shape, np.uint8), np.zeros(SPEC.packed_nelem, np.float32), frame_id=frame_id)
    assert seq is not None
    client.infer_end(seq)

  def test_the_flag_asks_the_server_for_its_stamps(self):
    transport = Scripted([response(1, 1), response(2, 2)])
    client = self.client(transport)
    self.frame(client, 1)
    _, flags = P.unpack_infer_req(transport.sent[0][2][0])
    self.assertTrue(flags & P.Flag.WANT_STAMPS)

  def test_the_flag_can_be_left_off_for_an_ab_run(self):
    transport = Scripted([response(1, 1)])
    client = self.client(transport)
    client.want_stamps = False
    self.frame(client, 1)
    _, flags = P.unpack_infer_req(transport.sent[0][2][0])
    self.assertFalse(flags & P.Flag.WANT_STAMPS)

  def test_a_stamp_for_the_last_frame_is_read_by_the_next_ones_wait(self):
    transport = Scripted([response(1, 1), stamps(1), response(2, 2)])
    client = self.client(transport)
    read, stamped = [], []
    client.on_reply = lambda frame_id, at: read.append(frame_id)
    client.on_stamps = lambda *spans: stamped.append(spans)
    self.frame(client, 1)
    self.assertEqual(read, [1])
    self.assertEqual(stamped, [], 'the reply was handed back before its stamp was read')
    self.frame(client, 2)
    self.assertEqual(stamped, [(1, 1500, 30_000, 500)])

  def test_a_phone_that_sends_no_stamps_still_answers(self):
    """Backward compatibility: no FRAME_STAMPS on the stream is an older
    server, and the frame is as good as it ever was."""
    transport = Scripted([response(1, 1), response(2, 2)])
    client = self.client(transport)
    read, stamped = [], []
    client.on_reply = lambda frame_id, at: read.append(frame_id)
    client.on_stamps = lambda *spans: stamped.append(spans)
    self.frame(client, 1)
    self.frame(client, 2)
    self.assertEqual(read, [1, 2])
    self.assertEqual(stamped, [])
    self.assertFalse(client.dead)
    self.assertEqual(client.last_timings, (0, 0, 20_000))

  def test_a_stamp_that_arrived_with_nothing_in_flight_is_still_read(self):
    """The frame's stamps follow its reply, and the frame loop drains before it
    sends the next frame: without reading with nothing in flight, every window
    would be one frame short of its stamps."""
    transport = Scripted([response(1, 1), stamps(1)])
    client = self.client(transport)
    stamped = []
    client.on_stamps = lambda *spans: stamped.append(spans)
    self.frame(client, 1)
    self.assertEqual(client.drain(), 0, 'nothing was answered: the reply was taken already')
    self.assertEqual(stamped, [(1, 1500, 30_000, 500)])

  def test_a_reply_drained_with_no_frame_out_is_dropped_not_a_failed_link(self):
    """A hello clears the flight; a reply that was already in the buffer
    answers nothing and must not kill a link that has just been greeted."""
    transport = Scripted([response(9, 9)])
    client = self.client(transport)
    with self.assertLogs('jetlink.client', level='WARNING') as logged:
      self.assertEqual(client.drain(), 0)
    self.assertFalse(client.dead)
    self.assertIn('discarding stale', logged.output[0])

  def test_a_truncated_stamp_is_dropped_not_a_failed_link(self):
    short = SimpleNamespace(msg_type=P.Msg.FRAME_STAMPS, seq=0, payload=memoryview(b'\x01\x00\x00\x00'))
    transport = Scripted([response(1, 1), short, response(2, 2)])
    client = self.client(transport)
    stamped = []
    client.on_stamps = lambda *spans: stamped.append(spans)
    with self.assertLogs('jetlink.client', level='WARNING') as logged:
      self.frame(client, 1)
      self.frame(client, 2)
    self.assertEqual(stamped, [])
    self.assertFalse(client.dead)
    self.assertIn('frame stamps', logged.output[0])

  def test_the_stamp_round_trips_through_the_wire_format(self):
    packed = P.pack_frame_stamps(4_000_000_000, 1500, 30_000, 500)
    self.assertEqual(len(packed), P.FRAME_STAMPS_SIZE)
    self.assertEqual(P.unpack_frame_stamps(packed), (4_000_000_000, 1500, 30_000, 500))


if __name__ == '__main__':
  unittest.main()
