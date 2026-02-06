"""Field Testing Utilities

Tools for testing in real police station environment:
- Background noise simulation
- Audio quality metrics
- Automated test scenarios
- Performance logging
"""

import asyncio
import numpy as np
import time
import json
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger
import wave


@dataclass
class TestScenario:
    """Test scenario definition."""

    name: str
    description: str
    utterances: List[str]
    background_noise: Optional[str] = None
    noise_volume: float = 0.3


@dataclass
class TestResult:
    """Results from a test run."""

    scenario: str
    timestamp: float
    latency_ms: float
    stt_accuracy: float
    tts_quality_score: float
    success: bool
    error_message: Optional[str] = None


class NoiseSimulator:
    """
    Simulates background noise for testing.
    """

    NOISE_TYPES = {
        "police_station": {
            "description": "Police station ambience (phones, talking, footsteps)",
            "frequency_range": (200, 4000),
            "modulation": 0.3,
        },
        "traffic": {
            "description": "Traffic noise from outside",
            "frequency_range": (100, 2000),
            "modulation": 0.2,
        },
        "crowd": {
            "description": "Crowd murmur",
            "frequency_range": (300, 3000),
            "modulation": 0.4,
        },
        "ac": {
            "description": "Air conditioning hum",
            "frequency_range": (50, 500),
            "modulation": 0.1,
        },
    }

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.current_noise = None
        self.noise_type = None

    def generate_noise(
        self, noise_type: str, duration_ms: int, volume: float = 0.3
    ) -> np.ndarray:
        """Generate synthetic background noise."""
        if noise_type not in self.NOISE_TYPES:
            raise ValueError(f"Unknown noise type: {noise_type}")

        spec = self.NOISE_TYPES[noise_type]
        num_samples = int(self.sample_rate * duration_ms / 1000)

        # Generate base noise
        if noise_type == "ac":
            # Sinusoidal hum with harmonics
            t = np.linspace(0, duration_ms / 1000, num_samples)
            base_freq = 60  # 60Hz hum
            noise = np.sin(2 * np.pi * base_freq * t)
            # Add harmonics
            noise += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
            noise += 0.1 * np.sin(2 * np.pi * base_freq * 3 * t)
        else:
            # White/pink noise
            noise = np.random.normal(0, 1, num_samples)

            # Apply frequency shaping based on type
            min_freq, max_freq = spec["frequency_range"]
            # Simple bandpass approximation using FFT
            noise_fft = np.fft.rfft(noise)
            freqs = np.fft.rfftfreq(num_samples, 1 / self.sample_rate)
            mask = (freqs >= min_freq) & (freqs <= max_freq)
            noise_fft[~mask] *= 0.1
            noise = np.fft.irfft(noise_fft)

        # Apply modulation
        modulation = spec["modulation"]
        if modulation > 0:
            t = np.linspace(0, duration_ms / 1000, num_samples)
            modulator = 1 + modulation * np.sin(2 * np.pi * 0.5 * t)
            noise = noise * modulator

        # Normalize and apply volume
        noise = noise / np.max(np.abs(noise)) * volume

        return noise.astype(np.float32)

    def mix_with_audio(
        self, audio: np.ndarray, noise_type: str, volume: float = 0.3
    ) -> np.ndarray:
        """Mix noise with audio signal."""
        duration_ms = len(audio) / self.sample_rate * 1000
        noise = self.generate_noise(noise_type, int(duration_ms), volume)

        # Match lengths
        min_len = min(len(audio), len(noise))
        return audio[:min_len] + noise[:min_len]

    def save_noise_sample(self, noise_type: str, duration_ms: int, path: Path):
        """Save noise sample to file for external use."""
        noise = self.generate_noise(noise_type, duration_ms)

        # Convert to int16
        noise_int16 = (noise * 32767).astype(np.int16)

        with wave.open(str(path), "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(noise_int16.tobytes())

        logger.info(f"Saved {noise_type} noise sample to {path}")


class AudioQualityMetrics:
    """
    Calculate audio quality metrics.
    """

    @staticmethod
    def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
        """Calculate signal-to-noise ratio in dB."""
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)

        if noise_power == 0:
            return 100.0  # No noise

        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db

    @staticmethod
    def calculate_clarity_score(audio: np.ndarray) -> float:
        """Calculate clarity score (0-100)."""
        # Measure dynamic range
        max_val = np.max(np.abs(audio))
        min_val = np.min(np.abs(audio))
        dynamic_range = max_val - min_val

        # Measure distortion (high frequency content)
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        total_energy = np.sum(magnitude)
        high_freq_energy = np.sum(magnitude[len(magnitude) // 2 :])

        distortion_ratio = high_freq_energy / total_energy if total_energy > 0 else 0

        # Combine metrics
        score = (dynamic_range * 50) + ((1 - distortion_ratio) * 50)
        return min(100, max(0, score))

    @staticmethod
    def measure_latency(start_time: float, end_time: float) -> float:
        """Measure latency in milliseconds."""
        return (end_time - start_time) * 1000


class AutomatedTester:
    """
    Run automated test scenarios.
    """

    # Test scenarios for Konkani voice agent
    TEST_SCENARIOS = [
        TestScenario(
            name="basic_greeting",
            description="Basic greeting and introduction",
            utterances=[
                "नमस्कार",
                "तुम्ही कोण आहात?",
                "तुम्ही काय करता?",
            ],
            background_noise="police_station",
        ),
        TestScenario(
            name="fir_inquiry",
            description="FIR filing inquiry",
            utterances=[
                "मला एफआयआर दाखल करायची आहे",
                "चोरी झाली आहे",
                "माझं पाकीट हरवलं आहे",
            ],
            background_noise="police_station",
        ),
        TestScenario(
            name="noise_stress",
            description="High background noise",
            utterances=[
                "नमस्कार",
                "मदत हवी आहे",
            ],
            background_noise="crowd",
            noise_volume=0.5,
        ),
        TestScenario(
            name="quiet_environment",
            description="Quiet environment baseline",
            utterances=[
                "नमस्कार",
                "कसं आहात?",
            ],
            background_noise=None,
        ),
    ]

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[TestResult] = []
        self.noise_simulator = NoiseSimulator()

    async def run_test(
        self,
        scenario: TestScenario,
        pipeline_callback: Callable,
        timeout: float = 30.0,
    ) -> TestResult:
        """Run a single test scenario."""
        logger.info(f"Running test: {scenario.name}")
        logger.info(f"  Description: {scenario.description}")

        start_time = time.time()

        try:
            # Run test through callback
            result = await asyncio.wait_for(
                pipeline_callback(scenario), timeout=timeout
            )

            elapsed = (time.time() - start_time) * 1000

            test_result = TestResult(
                scenario=scenario.name,
                timestamp=time.time(),
                latency_ms=elapsed,
                stt_accuracy=result.get("stt_accuracy", 0.0),
                tts_quality_score=result.get("tts_quality", 0.0),
                success=True,
            )

            logger.info(f"✓ Test {scenario.name} passed ({elapsed:.1f}ms)")

        except asyncio.TimeoutError:
            test_result = TestResult(
                scenario=scenario.name,
                timestamp=time.time(),
                latency_ms=timeout * 1000,
                stt_accuracy=0.0,
                tts_quality_score=0.0,
                success=False,
                error_message="Timeout",
            )
            logger.error(f"✗ Test {scenario.name} timed out")

        except Exception as e:
            test_result = TestResult(
                scenario=scenario.name,
                timestamp=time.time(),
                latency_ms=(time.time() - start_time) * 1000,
                stt_accuracy=0.0,
                tts_quality_score=0.0,
                success=False,
                error_message=str(e),
            )
            logger.error(f"✗ Test {scenario.name} failed: {e}")

        self.results.append(test_result)
        return test_result

    async def run_all_tests(self, pipeline_callback: Callable) -> Dict:
        """Run all test scenarios."""
        logger.info("=" * 60)
        logger.info("STARTING AUTOMATED TEST SUITE")
        logger.info("=" * 60)

        for scenario in self.TEST_SCENARIOS:
            await self.run_test(scenario, pipeline_callback)
            await asyncio.sleep(1)  # Brief pause between tests

        # Generate report
        report = self.generate_report()
        self.save_report(report)

        return report

    def generate_report(self) -> Dict:
        """Generate test report."""
        if not self.results:
            return {}

        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed

        latencies = [r.latency_ms for r in self.results]
        stt_scores = [r.stt_accuracy for r in self.results if r.success]
        tts_scores = [r.tts_quality_score for r in self.results if r.success]

        return {
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "success_rate": passed / total if total > 0 else 0,
            },
            "latency": {
                "avg_ms": sum(latencies) / len(latencies) if latencies else 0,
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0,
            },
            "quality": {
                "avg_stt_accuracy": sum(stt_scores) / len(stt_scores)
                if stt_scores
                else 0,
                "avg_tts_quality": sum(tts_scores) / len(tts_scores)
                if tts_scores
                else 0,
            },
            "results": [asdict(r) for r in self.results],
        }

    def save_report(self, report: Dict):
        """Save report to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"test_report_{timestamp}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Test report saved to {report_path}")

        # Also save human-readable summary
        summary_path = self.output_dir / f"test_summary_{timestamp}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("KONKANI VOICE AGENT - TEST SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            summary = report.get("summary", {})
            f.write(f"Total Tests: {summary.get('total_tests', 0)}\n")
            f.write(f"Passed: {summary.get('passed', 0)}\n")
            f.write(f"Failed: {summary.get('failed', 0)}\n")
            f.write(f"Success Rate: {summary.get('success_rate', 0) * 100:.1f}%\n\n")

            latency = report.get("latency", {})
            f.write("Latency:\n")
            f.write(f"  Average: {latency.get('avg_ms', 0):.1f}ms\n")
            f.write(f"  Min: {latency.get('min_ms', 0):.1f}ms\n")
            f.write(f"  Max: {latency.get('max_ms', 0):.1f}ms\n\n")

            quality = report.get("quality", {})
            f.write("Quality:\n")
            f.write(
                f"  Avg STT Accuracy: {quality.get('avg_stt_accuracy', 0) * 100:.1f}%\n"
            )
            f.write(
                f"  Avg TTS Quality: {quality.get('avg_tts_quality', 0) * 100:.1f}%\n\n"
            )

            f.write("=" * 60 + "\n")

        logger.info(f"Test summary saved to {summary_path}")


class PerformanceLogger:
    """
    Log detailed performance metrics.
    """

    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            "stt_latency": [],
            "llm_latency": [],
            "tts_latency": [],
            "total_latency": [],
            "memory_usage": [],
        }

    def log_metric(self, metric_type: str, value: float):
        """Log a single metric."""
        if metric_type in self.metrics:
            self.metrics[metric_type].append(
                {
                    "timestamp": time.time(),
                    "value": value,
                }
            )

    def save_metrics(self):
        """Save all metrics to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        metrics_path = self.log_dir / f"performance_{timestamp}.json"

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Performance metrics saved to {metrics_path}")

    def get_statistics(self) -> Dict:
        """Get statistical summary of metrics."""
        stats = {}

        for metric_type, values in self.metrics.items():
            if values:
                vals = [v["value"] for v in values]
                stats[metric_type] = {
                    "count": len(vals),
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                    "p95": np.percentile(vals, 95),
                }

        return stats
