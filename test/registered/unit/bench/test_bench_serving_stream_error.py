"""Unit tests for how bench_serving records a request the server fails inside a 200 stream."""

import asyncio
import json
import threading
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    async_request_openai_completions,
    async_request_sglang_generate,
    set_global_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MESSAGE = "Request waiting timeout reached."
# The events an SGLang server streams when SGLANG_REQ_WAITING_TIMEOUT expires on
# a queued request: HTTP 200, then the error in-band.
OPENAI_ERROR_EVENT = {
    "error": {
        "object": "error",
        "message": MESSAGE,
        "type": "SERVICE_UNAVAILABLE",
        "param": None,
        "code": 503,
    }
}
NATIVE_ABORT_EVENT = {
    "text": "",
    "output_ids": [],
    "meta_info": {
        "id": "rid",
        "finish_reason": {"type": "abort", "status_code": 503, "message": MESSAGE},
        "completion_tokens": 0,
    },
}
REQUESTED_OUTPUT_LEN = 64


class _AbortingHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        event = NATIVE_ABORT_EVENT if self.path == "/generate" else OPENAI_ERROR_EVENT
        payload = f"data: {json.dumps(event)}\n\n".encode() + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


class TestBenchServingStreamError(CustomTestCase):
    """A request the server aborts after the 200 header is a failed request.

    Its stream carries an error event and no token. It used to be recorded as a
    completed request with the requested output length, which inflated the
    completed count and the output token throughput whenever a server shed load.
    """

    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _AbortingHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        set_global_args(
            Namespace(
                disable_stream=False,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
                cache_report=False,
                return_logprob=False,
                return_routed_experts=False,
                top_logprobs_num=0,
                token_ids_logprob=None,
                logprob_start_len=-1,
                temperature=0.0,
                top_p=1.0,
            )
        )

    def _run(self, request_func, path):
        request = RequestFuncInput(
            prompt="hello",
            api_url=f"http://127.0.0.1:{self.server.server_port}{path}",
            prompt_len=1,
            output_len=REQUESTED_OUTPUT_LEN,
            model="dummy-model",
            lora_name="",
            image_data=None,
            extra_request_body={},
        )
        return asyncio.run(request_func(request))

    def test_in_stream_error_fails_the_request(self):
        cases = [
            ("chat", async_request_openai_chat_completions, "/v1/chat/completions"),
            ("completions", async_request_openai_completions, "/v1/completions"),
            ("native", async_request_sglang_generate, "/generate"),
        ]
        for name, request_func, path in cases:
            with self.subTest(name):
                output = self._run(request_func, path)
                self.assertFalse(output.success)
                self.assertEqual(output.error, MESSAGE)
                self.assertEqual(output.output_len, 0)


if __name__ == "__main__":
    unittest.main()
