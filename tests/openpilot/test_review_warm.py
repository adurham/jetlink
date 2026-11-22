"""Reviewer's own tests for jetlink/warm-handback (9554b8e + 8362ee0).

Run from the jetlink worktree:
  PYTHONPATH=<this dir> timeout 240 python3 -m unittest test_review_warm -q

1. OffFuzzTest: randomized differential, main (9be48b1, loaded from git) against this
   branch with the warm hand-back off (default and explicit): same model calls, same
   published frames, same log lines, same handovers, over 40 random drives (held runs,
   behind, end() and send() failures, modeld frame-drop spikes, swaps back in).
2. OnFuzzTest: randomized invariants with it on: a warm frame is never published, the
   small model never runs one camera frame twice, and at every hand-back its history is
   either [hand-back frame] (cold, reset) or a contiguous run of frames ending at the
   hand-back frame (no stale history from an earlier episode, nothing twice).
3. UnboundedWarmTest (EXPECTED TO FAIL on the branch as committed): a host that holds 1
   frame in 10, which never trips HOLDS_IN_A_ROW or HOLDS_ALLOWED, keeps the small model
   warming on every frame for as long as it lasts (the load that starved driver
   monitoring in d790583). The reviewer's WARM_MAX patch caps it.
"""
import importlib.util
import random
import subprocess
from pathlib import Path
from unittest import mock

from jetlink.openpilot import joining
from tests.openpilot.fakes import RecordingLog
from tests.openpilot.test_joining import HistoryModel, JoiningBase, LinkModel, plan_array, wait_for

ROOT = Path(joining.__file__).resolve().parents[2]


def load_main():
  src = subprocess.run(['git', 'show', '9be48b1:jetlink/openpilot/joining.py'], capture_output=True, text=True, cwd=ROOT)
  assert src.returncode == 0, 'main (9be48b1) must be in this clone'
  spec = importlib.util.spec_from_loader('joining_main_review', loader=None)
  main = importlib.util.module_from_spec(spec)
  main.__file__ = joining.__file__
  exec(compile(src.stdout, 'joining_main_review.py', 'exec'), main.__dict__)  # noqa: S102
  return main


class ScriptModel(LinkModel):
  """LinkModel whose k-th frame (0-based end() call) does what script[k] says:
  'held', 'behind' (held and behind), 'fail_end', 'fail_send', or '' (answered)."""

  def __init__(self, steps, script):
    super().__init__(steps)
    self.script = script

  def _what(self, k):
    return self.script[k] if k < len(self.script) else ''

  def send(self, want_telemetry=False):
    self.steps.append(f'big send {self.frame}')
    if self._what(self.calls) == 'fail_send':
      self.calls += 1
      raise RuntimeError('send failed')

  def end(self, after_enqueue=None):
    self.steps.append(f'big end {self.frame}')
    k = self.calls
    self.calls += 1
    what = self._what(k)
    if what == 'fail_end':
      raise RuntimeError('no answer; link abandoned')
    if after_enqueue is not None:
      after_enqueue()
    self.held = what in ('held', 'behind')
    self.behind = 'held 5 frames in a row' if what == 'behind' else None
    return {'from': 'big', 'plan': plan_array(7.0), 'frame': self.frame}


def random_script(rng, n):
  """A drive's worth of the large model's frames: mostly answered, held runs of 1..5
  (5 ends 'behind'), and now and then a failure that ends the stretch."""
  out = []
  while len(out) < n:
    r = rng.random()
    if r < 0.10:
      run = rng.randint(1, 5)
      out += ['held'] * (run - 1) + (['behind'] if run == 5 else ['held'])
    elif r < 0.13:
      out.append(rng.choice(['fail_end', 'fail_send']))
    else:
      out += [''] * rng.randint(1, 12)
  return out[:n]


def drive(module, seed, **kwargs):
  """One random drive through `module`'s JoiningModelState. Returns the trace."""
  rng = random.Random(seed)
  steps = []
  small = HistoryModel(steps)
  script = random_script(rng, 400)
  big = ScriptModel(steps, script)
  log = RecordingLog()
  outs, events = [], []
  with mock.patch.object(module, 'SMALL_WARMUP_FRAMES', 0), mock.patch.object(module, 'REJOIN_DELAY_QUICK', 0.005), \
       mock.patch.object(module, 'REJOIN_DELAY', 0.005), mock.patch.object(module, 'REJOIN_DELAY_MAX', 0.005):
    s = module.JoiningModelState(small, lambda should_stop=None: (mock.MagicMock(name='client'), 'spec'),
                                 lambda c, sp: big, reset_small=small.reset, progress=mock.Mock(), log=log, **kwargs)
    try:
      frame = 0
      drops = rng.random() < 0.3
      for _stretch in range(12):
        if big.calls >= len(script):
          break
        assert wait_for(lambda: s._joined is not None, 5), 'never joined'
        s.in_control = False
        swapped = False
        while big.calls < len(script):
          frame += 1
          s.frame_drop_ratio = 0.02 if (drops and swapped and rng.random() < 0.01) else 0.0
          was_big = s._big is not None
          resets, calls = small.resets, big.calls
          o = s.run({}, {}, {'frame': frame, 'vx': 5.0})
          outs.append((o['from'], o.get('frame')))
          events.append((frame, was_big, s._big is not None, o['from'], small.resets - resets, list(small.history)))
          if s._big is not None or big.calls > calls:
            # the large model took this frame up (it may have handed back on it)
            swapped = True
          if swapped and s._big is None:
            break
        # in control while the small model drives a few frames, so the swap back in
        # lands on a frame the test chooses and not when the join thread happens to
        s.in_control = True
        for _ in range(rng.randint(2, 6)):
          frame += 1
          o = s.run({}, {}, {'frame': frame, 'vx': 5.0})
          outs.append((o['from'], o.get('frame')))
          events.append((frame, False, s._big is not None, o['from'], 0, list(small.history)))
    finally:
      s.close()
      s._thread.join(5)
  lines = [x for x in log.lines('warning') + log.lines('exception') + log.lines('error')
           if 'next attempt in' not in x and 'built the large model state in' not in x and 'fallback frame' not in x]
  return dict(steps=steps, outs=outs, lines=lines, handovers=s.handovers, events=events, small=small, big=big)


class OffFuzzTest(JoiningBase):
  def test_off_is_main_call_for_call_over_random_drives(self):
    main = load_main()
    with_resets = 0
    for seed in range(40):
      with self.subTest(seed=seed):
        a = drive(main, seed)
        b = drive(joining, seed)
        c = drive(joining, seed, warm_handback=False)
        for k in ('steps', 'outs', 'lines', 'handovers'):
          self.assertEqual(a[k], b[k], f'{k} differ (default off)')
          self.assertEqual(a[k], c[k], f'{k} differ (explicit off)')
        with_resets += 'small reset' in a['steps']
        self.assertFalse(any('warm' in x for x in b['lines']))
    # the drives mean something: most of them handed back, cold, more than once
    self.assertGreaterEqual(with_resets, 30)


class OnFuzzTest(JoiningBase):
  def test_on_invariants_over_random_drives(self):
    warm_handbacks = cold = 0
    for seed in range(40):
      with self.subTest(seed=seed):
        t = drive(joining, seed, warm_handback=True)
        runs = [x for x in t['steps'] if x.startswith('small run ')]
        self.assertEqual(len(runs), len(set(runs)), 'a frame ran twice in the small model')
        for frame, was_big, is_big, src, reset_now, history in t['events']:
          if is_big:
            self.assertEqual(src, 'big', f'frame {frame}: a warm frame was published')
          if was_big and not is_big:
            self.assertEqual(src, 'small')
            self.assertEqual(history[-1], frame)
            self.assertEqual(history, list(range(history[0], frame + 1)), f'frame {frame}: history not contiguous')
            if reset_now:
              cold += 1
              self.assertEqual(history, [frame])
            else:
              warm_handbacks += 1
              self.assertGreaterEqual(len(history), 2, 'warm with no warm frame')
    self.assertGreater(warm_handbacks, 10)
    self.assertGreater(cold, 5)


class UnboundedWarmTest(JoiningBase):
  """EXPECTED TO FAIL on 9554b8e/8362ee0 (proves finding F1): a host that holds
  1 frame in 10 (a warm-up that never ends) or 1 in 25 (one warm-up of 20 after
  another) never hands back (never 5 in a row, never more than 20 in 10 s), and
  the small model runs unpublished on most frames of the stretch. The reviewer's
  WARM_MAX caps the warm frames in any 10 s (the frames here run in well under 10 s)."""

  def _warm_runs(self, every):
    steps = []
    small = HistoryModel(steps)
    script = ['held' if k % every == 5 else '' for k in range(400)]
    big = ScriptModel(steps, script)
    s = self._make(small, self._connect, lambda c, sp: big, reset_small=small.reset, warm_handback=True)
    self.addCleanup(self._close, s)
    self._wait_joined(s)
    s.in_control = False
    for frame in range(1, 401):
      s.run({}, {}, {'frame': frame, 'vx': 5.0})
    self.assertIsNotNone(s._big, 'it never handed back')
    return sum(1 for x in steps if x.startswith('small run ')), small

  def test_a_host_that_holds_one_frame_in_ten_does_not_warm_forever(self):
    runs, _ = self._warm_runs(10)
    self.assertLessEqual(runs, getattr(joining, 'WARM_MAX', 60), f'the small model ran {runs} of 400 frames unpublished')

  def test_a_host_that_holds_one_frame_in_25_does_not_warm_most_frames(self):
    runs, _ = self._warm_runs(25)
    self.assertLessEqual(runs, getattr(joining, 'WARM_MAX', 60), f'the small model ran {runs} of 400 frames unpublished')

  def test_capped_the_next_hand_back_is_cold_and_clean(self):
    # past the cap the warm-up has ended: a lag hand-back resets, history = [its frame]
    if not hasattr(joining, 'WARM_MAX'):
      runs, small = self._warm_runs(10)
      self.assertLessEqual(runs, 60, f'uncapped: {runs} warm frames')
    steps = []
    small = HistoryModel(steps)
    script = ['held' if k % 10 == 5 else '' for k in range(300)] + ['held'] * 4 + ['behind'] + [''] * 5
    big = ScriptModel(steps, script)
    s = self._make(small, self._connect, lambda c, sp: big, reset_small=small.reset, warm_handback=True)
    self.addCleanup(self._close, s)
    self._wait_joined(s)
    s.in_control = False
    frame = 0
    while s._big is not None or frame == 0:
      frame += 1
      out = s.run({}, {}, {'frame': frame, 'vx': 5.0})
    self.assertEqual(out['from'], 'small')
    self.assertEqual(small.history, [frame])
