"""Unit tests for the goodput metrics bench_serving reports under --goodput."""

import unittest

from sglang.benchmark.serving import (
    RequestFuncOutput,
    _goodput_result_fields,
    calculate_metrics,
    parse_goodput_slos,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def _output(ttft, latency, output_len, success=True):
    return RequestFuncOutput(
        generated_text=" ".join(["tok"] * output_len),
        success=success,
        latency=latency,
        ttft=ttft,
        output_len=output_len,
    )


# ttft / tpot / e2el in ms: 100 / 100 / 900, 500 / 100 / 900, 200 / 200 / 2200,
# 200 / 0 / 200 (single token), then one failed request.
OUTPUTS = [
    _output(ttft=0.1, latency=0.9, output_len=9),
    _output(ttft=0.5, latency=0.9, output_len=5),
    _output(ttft=0.2, latency=2.2, output_len=11),
    _output(ttft=0.2, latency=0.2, output_len=1),
    _output(ttft=0.0, latency=0.0, output_len=0, success=False),
]
DURATION_S = 10.0


def _metrics(goodput_slos):
    metrics, _ = calculate_metrics(
        input_requests=None,
        outputs=OUTPUTS,
        dur_s=DURATION_S,
        tokenizer=_WhitespaceTokenizer(),
        backend="sglang-oai-chat",
        goodput_slos=goodput_slos,
    )
    return metrics


class TestParseGoodputSlos(CustomTestCase):
    def test_unset_means_disabled(self):
        self.assertIsNone(parse_goodput_slos(None))
        self.assertIsNone(parse_goodput_slos([]))

    def test_pairs_are_parsed_as_milliseconds(self):
        self.assertEqual(
            parse_goodput_slos(["ttft:500", "tpot:50.5", "e2el:20000"]),
            {"ttft": 500.0, "tpot": 50.5, "e2el": 20000.0},
        )

    def test_invalid_entries_are_rejected(self):
        for pairs in (
            ["itl:10"],
            ["ttft"],
            ["ttft:"],
            ["ttft:fast"],
            ["ttft:0"],
            ["ttft:-5"],
            ["ttft:inf"],
            ["ttft:100", "ttft:200"],
        ):
            with self.subTest(pairs=pairs):
                with self.assertRaises(ValueError):
                    parse_goodput_slos(pairs)


class TestGoodputMetrics(CustomTestCase):
    def test_disabled_leaves_fields_unset(self):
        metrics = _metrics(None)
        self.assertIsNone(metrics.request_goodput)
        self.assertIsNone(metrics.slo_attainment)
        self.assertIsNone(metrics.slo_attainment_by_metric)

    def test_request_is_good_only_if_every_slo_is_met(self):
        metrics = _metrics({"ttft": 300.0, "tpot": 150.0, "e2el": 1000.0})
        # Good: the first and the single-token request; the failed one is a miss.
        self.assertAlmostEqual(metrics.request_goodput, 2 / DURATION_S)
        self.assertAlmostEqual(metrics.slo_attainment, 2 / 5)
        self.assertEqual(
            metrics.slo_attainment_by_metric,
            {"ttft": 3 / 5, "tpot": 3 / 5, "e2el": 3 / 5},
        )

    def test_only_listed_slos_are_checked(self):
        metrics = _metrics({"ttft": 300.0})
        self.assertAlmostEqual(metrics.request_goodput, 3 / DURATION_S)
        self.assertEqual(metrics.slo_attainment_by_metric, {"ttft": 3 / 5})


class TestGoodputReporting(CustomTestCase):
    def test_no_result_fields_without_slos(self):
        self.assertEqual(
            _goodput_result_fields(goodput_slos=None, metrics=_metrics(None)), {}
        )

    def test_result_fields(self):
        slos = {"ttft": 300.0, "tpot": 150.0}
        fields = _goodput_result_fields(goodput_slos=slos, metrics=_metrics(slos))
        self.assertEqual(
            fields,
            {
                "goodput_slos_ms": slos,
                "request_goodput": 2 / DURATION_S,
                "slo_attainment": 2 / 5,
                "slo_attainment_by_metric": {"ttft": 3 / 5, "tpot": 3 / 5},
            },
        )


if __name__ == "__main__":
    unittest.main()
