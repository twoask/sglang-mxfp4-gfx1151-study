import json

from group_kernels import group_trace
from steps_from_log import parse_steps


def _decode_line(running, tok_s, queue):
    return (
        f"[2026-11-07 20:15:15] Decode batch, #running-req: {running}, #token: 18000, token usage: 0.10, "
        f"cuda graph: True, gen throughput (token/s): {tok_s}, #queue-req: {queue}\n"
    )


def test_parse_steps_ignores_queued_lines(tmp_path):
    log = tmp_path / "srv.log"
    # skip_first drops the first queue-0 line of each batch size (ramp-up); 999 must not leak in
    lines = [_decode_line(16, 999.0, 0)] + [_decode_line(16, 400.0, 0)] * 4 + [_decode_line(16, 100.0, 3)] * 2
    log.write_text("".join(lines))
    assert parse_steps(str(log)) == {16: 40.0}


def test_parse_steps_needs_three_lines(tmp_path):
    log = tmp_path / "srv.log"
    log.write_text(_decode_line(4, 200.0, 0) * 3)  # 3 lines minus skip_first = 2, too few
    assert 4 not in parse_steps(str(log))


def test_group_trace_reports_unmatched(tmp_path):
    trace = tmp_path / "t.json"
    events = [
        {"ph": "X", "cat": "kernel", "name": "_matmul_ogs_NNT_bf16xmxfp4", "dur": 300.0},
        {"ph": "X", "cat": "kernel", "name": "_fwd_kernel_stage1", "dur": 100.0},
        {"ph": "X", "cat": "kernel", "name": "mystery_kernel", "dur": 50.0},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "dur": 9999.0},  # not a GPU kernel
    ]
    trace.write_text(json.dumps({"traceEvents": events}))
    rules = [("moe", r"matmul_ogs|_matmul"), ("attn", r"_fwd_kernel")]
    got = group_trace(str(trace), rules, n_steps=2)
    # microseconds summed per group, per step, in ms
    assert got == {"moe": 0.15, "attn": 0.05, "unmatched": 0.025}
