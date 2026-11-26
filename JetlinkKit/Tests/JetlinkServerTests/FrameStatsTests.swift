import Foundation
import Testing

@testable import JetlinkServer

/// The phone-side timing breakdown that rides on the telemetry payload.
@Suite("Frame stats timing")
struct FrameStatsTests {
  final class Clock: @unchecked Sendable {
    var value: TimeInterval = 0
    var read: @Sendable () -> TimeInterval { { [unowned self] in self.value } }
  }

  @Test("telemetryTiming reports p50, p95, max and the count over 46 ms")
  func phoneTelemetryTiming() {
    let clock = Clock()
    let stats = FrameStats(now: clock.read)
    // 10 frames at 1 ms, then 10 frames climbing to 60 ms, all in the last 10 s
    for i in 0..<20 {
      clock.value = Double(i) * 0.5
      let total = UInt32(i < 10 ? 1_000 : 10_000 + i * 5_000)
      stats.record(totalUs: total, gpuUs: UInt32(Double(total) * 0.8), queueUs: 100, sendUs: 50)
    }
    clock.value = 10.0
    let timing = stats.telemetryTiming(window: 10)
    #expect(timing["count"] as? Int == 20)
    #expect(timing["total_p50_ms"] as? Double == 15.0)
    #expect(timing["total_p95_ms"] as? Double == 55.0)
    #expect(timing["total_max_ms"] as? Double == 60.0)
    #expect(timing["gpu_p50_ms"] as? Double == 12.0)
    #expect(timing["over_46ms"] as? Int == 3)
    #expect(timing["window_s"] as? Double == 10.0)
  }

  @Test("telemetryTiming is empty when no frames landed in the window")
  func emptyWindow() {
    let clock = Clock()
    let stats = FrameStats(now: clock.read)
    stats.record(totalUs: 30_000, gpuUs: 20_000, queueUs: 100, sendUs: 50)
    clock.value = 20.0
    let timing = stats.telemetryTiming(window: 10)
    #expect(timing.isEmpty)
  }
}
