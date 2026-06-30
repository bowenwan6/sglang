# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/cuda_piecewise_backend.py

import dataclasses
import logging
import os
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import torch.fx as fx

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compilation_counter import compilation_counter
from sglang.srt.compilation.compile_phase import (
    get_pcg_capture_stream,
    is_in_torch_compile_warmup,
)
from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import print_warning_once

logger = logging.getLogger(__name__)
_is_hip = is_hip()

# Diagnostic gate for the PCG capture-stream root-cause investigation.
# When SGLANG_DEBUG_PCG_CALL_TRACE=1, every CUDAPiecewiseBackend.__call__
# entry logs its instance id, layer index, warmup state, runtime shape,
# entry.cudagraph state, and (when about to consult get_pcg_capture_stream)
# the capture-stream identity. Off by default; production behavior unchanged.
_DEBUG_PCG_CALL_TRACE = os.environ.get("SGLANG_DEBUG_PCG_CALL_TRACE") == "1"


def _pcg_dbg(msg: str) -> None:
    # Single sink so the trace lands in the server log alongside Dynamo
    # recompile lines. Uses print so it survives whatever logger config
    # the launcher set.
    if _DEBUG_PCG_CALL_TRACE:
        print("[PCG_DEBUG] " + msg, flush=True)


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    need_to_compile: bool  # the size is in compile_sizes
    use_cudagraph: bool  # the size is in cudagraph_capture_sizes

    compiled: bool = False
    runnable: Callable = None  # type: ignore
    num_finished_warmup: int = 0
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None


class CUDAPiecewiseBackend:

    def __init__(
        self,
        graph: fx.GraphModule,
        compile_config: CompilationConfig,
        inductor_config: dict[str, Any],
        graph_pool: Any,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        compiled_graph_for_general_shape: Callable,
        sglang_backend,
    ):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        Independently, we will capture cudagraph for different shapes.

        If a shape needs both compilation and cudagraph, we will
        compile it first, and then capture cudagraph.
        """
        self.graph = graph
        self.inductor_config = inductor_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.sglang_backend = sglang_backend

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.compile_sizes: set[int] = set([])
        self.compile_config = compile_config
        self.cudagraph_capture_sizes: set[int] = set(compile_config.get_capture_sizes())

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        # the entries for different shapes that we need to either
        # compile or capture cudagraph
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: set[int] = self.compile_sizes.copy()
        for shape in self.compile_sizes.union(self.cudagraph_capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_cudagraph=shape in self.cudagraph_capture_sizes,
            )

    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.sglang_backend.compiler_manager.save_to_file()

    def __call__(self, *args) -> Any:
        if _DEBUG_PCG_CALL_TRACE:
            _pcg_dbg(
                f"call enter id={id(self):#x} layer_idx={self.piecewise_compile_index}"
                f"/{self.total_piecewise_compiles} first_run_finished={self.first_run_finished} "
                f"in_warmup={is_in_torch_compile_warmup()} "
                f"sym_shape_indices={self.sym_shape_indices}"
            )

        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        if len(self.sym_shape_indices) == 0:
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            if _DEBUG_PCG_CALL_TRACE:
                _pcg_dbg(
                    f"call id={id(self):#x} layer_idx={self.piecewise_compile_index} "
                    f"runtime_shape={runtime_shape} NOT in concrete_size_entries "
                    f"(general-shape fallback)"
                )
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]
        if _DEBUG_PCG_CALL_TRACE:
            _pcg_dbg(
                f"call id={id(self):#x} layer_idx={self.piecewise_compile_index} "
                f"runtime_shape={runtime_shape} "
                f"entry.cudagraph={'set' if entry.cudagraph is not None else 'None'} "
                f"entry.num_finished_warmup={entry.num_finished_warmup} "
                f"entry.compiled={entry.compiled}"
            )

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = self.sglang_backend.compiler_manager.compile(
                self.graph,
                args,
                self.inductor_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape,
            )

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        if is_in_torch_compile_warmup():
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < 1:  # noqa
                entry.num_finished_warmup += 1
                if _DEBUG_PCG_CALL_TRACE:
                    _pcg_dbg(
                        f"call id={id(self):#x} layer_idx={self.piecewise_compile_index} "
                        f"runtime_shape={runtime_shape} "
                        f"first warmup pass (num_finished_warmup -> "
                        f"{entry.num_finished_warmup}) — returning without capture"
                    )
                return entry.runnable(*args)

            # During normal capture (PiecewiseCudaGraphRunner.capture()),
            # set_pcg_capture_stream() guarantees a valid stream. However,
            # Dynamo may silently recompile (e.g. multimodal models'
            # forward when input_deepstack_embeds toggles from None to a
            # tensor on the first image request, or HIP/MLA batches
            # whose token count exceeds the captured range). The
            # replacement backend has no capture stream; fall back to
            # the inductor-compiled general-shape graph instead of
            # crashing. Pre-existed for HIP; extended to CUDA so the
            # assertion can no longer be reached at inference time.
            stream = get_pcg_capture_stream()
            if _DEBUG_PCG_CALL_TRACE:
                _pcg_dbg(
                    f"call id={id(self):#x} layer_idx={self.piecewise_compile_index} "
                    f"runtime_shape={runtime_shape} "
                    f"about to capture; stream={'set' if stream is not None else 'None'} "
                    f"_is_hip={_is_hip}"
                )
            if stream is None:
                print_warning_once(
                    "PCG capture stream is not set; likely a Dynamo runtime "
                    "recompilation. Falling back to eager execution for this "
                    "subgraph."
                )
                return entry.runnable(*args)

            if self.compile_config.get_enable_debug_mode():
                input_addresses = [
                    x.data_ptr() for x in args if isinstance(x, torch.Tensor)
                ]
                entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    # during every model forward, we will capture
                    # many pieces of cudagraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.cuda.empty_cache", lambda: None))
                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(cudagraph, pool=self.graph_pool, stream=stream):
                    # `output` is managed by pytorch's cudagraph pool
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.compile_config.get_enable_debug_mode():
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )
        if _DEBUG_PCG_CALL_TRACE:
            _pcg_dbg(
                f"call id={id(self):#x} layer_idx={self.piecewise_compile_index} "
                f"runtime_shape={runtime_shape} replay captured cudagraph"
            )
        entry.cudagraph.replay()
        return entry.output
