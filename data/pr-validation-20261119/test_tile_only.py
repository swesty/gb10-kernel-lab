from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x as require_sm120


def _randn(shape, *, device, scale=0.25):
    return torch.randn(shape, dtype=torch.float32, device=device).mul_(scale).to(torch.bfloat16).contiguous()


@pytest.mark.parametrize(
    ("output_columns", "tile_n"),
    [(2560, 128), (2496, 64), (2512, 16)],
)
def test_prefill_gemm_tile_n_choice_matches_reference(
    output_columns: int, tile_n: int
) -> None:
    """Each output-column tile width (128, then 64, then 16) computes the same GEMM."""
    from b12x.sequence.mtp_feedback import _cute_prefill

    device = require_sm120()
    rows, width, streams = 512, 2560, 4
    kernel = _cute_prefill.MTPPrefillBf16GemmKernel(
        rows=rows,
        output_columns=output_columns,
        reduction_width=width,
        streams=streams,
        add_token_path=True,
    )
    assert kernel.tile_n == tile_n
    inputs = _randn((rows, width), device=device, scale=0.5)
    weight = _randn((output_columns, width), device=device, scale=width**-0.5)
    token_path = _randn((rows // streams, output_columns), device=device, scale=0.5)
    output = torch.empty(rows * output_columns, device=device, dtype=torch.bfloat16)
    launch = _cute_prefill.compile_mtp_prefill_bf16_gemm(
        rows,
        output_columns,
        width,
        device=device,
        streams=streams,
        add_token_path=True,
    )
    live_rows = rows - 64
    launch(inputs, weight, output, token_path=token_path.view(-1), live_rows=live_rows)
    torch.cuda.synchronize(device)

    expected = (inputs[:live_rows].float() @ weight.float().t()) + token_path.float()[
        torch.arange(live_rows, device=device) // streams
    ]
    actual = output.view(rows, output_columns)[:live_rows].float()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-2)
