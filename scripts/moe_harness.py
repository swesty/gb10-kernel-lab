"""b12x NVFP4 MoE prefill harness at Qwen3.8 Flash Next shapes on GB10.

Shapes: 512 experts, top-10, hidden 2560, expert width 640, NVFP4 weights (ModelOpt layout, W31),
FP4 activations (A4), 4,096-token calls - the same as each prefill chunk of the live engine.
Weights are random legal NVFP4 bytes (timing does not depend on values). Routes come from the live
dump (data/moe-routes/*.pt, captured from one 14.5K real-text prefill) or from b12x's own synthetic
tuning workloads. Each configuration is an explicit MoeDecodeConfig, so tile M, input sharing,
intermediate materialisation and deterministic output can be compared in one process.

Usage:
  python moe_harness.py --routes /path/moe-routes --configs t32,t16,t64,t32det --iters 3
Config tokens: t<tile_m>[s0][mat][det][mac<N>]  e.g. t32 (shared input on), t32s0 (off), t128mat, t32det
"""
import argparse
import glob
import os
import torch

from b12x.moe.fused_moe import api
from b12x.moe.fused_moe._tuning import MoeDecodeConfig
from b12x.moe.fused_moe.workloads import make_routing_ids

E, H, I, K = 512, 2560, 640, 10
dev = torch.device("cuda", 0)


def make_weights(seed=7919):
    g = torch.Generator(device=dev).manual_seed(seed)

    def blocks(n, k):
        size = ((n + 127) // 128 * 128) * ((k // 16 + 3) // 4 * 4)
        return torch.full((E, size), 0.5, dtype=torch.float8_e4m3fn, device=dev)

    source = api.PackedWeights(
        w13=torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g),
        w2=torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev, generator=g),
        w13_block_scales=blocks(2 * I, H),
        w2_block_scales=blocks(H, I),
        w13_global_scales=torch.linspace(0.008, 0.016, E, device=dev),
        w2_global_scales=torch.linspace(0.008, 0.016, E, device=dev),
        input_scale=torch.full((E,), 64.0, device=dev),
        intermediate_scale=torch.full((E,), 64.0, device=dev),
        immutable_input_scales=True,  # static checkpoint scales, as in serving
    )
    plan = api.plan_weights(
        source=api.PackedSource(format="modelopt_nvfp4", w13_layout="w31"),
        activation=api.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=api.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
    )
    experts = api.prepare_weights(plan=plan, weights=source)
    return experts._impl.plan, experts._impl


def parse(token):
    t = token
    tile = int(t[1:].split("s")[0].split("m")[0].split("d")[0])
    return dict(
        tile=tile,
        share="s0" not in t,
        mat="mat" in t,
        det="det" in t,
        mac=int(t.split("mac")[1]) if "mac" in t else None,
    )


def build(wplan, tokens, c):
    cfg = MoeDecodeConfig(
        backend="dynamic", route_planner="internal", max_active_clusters=c["mac"],
        dynamic_tile_m=c["tile"], dynamic_route_mode="grouped",
        nvfp4_share_input=c["share"], nvfp4_materialize_intermediate=c["mat"],
    )
    caps = api.Caps(
        max_tokens=tokens, num_topk=K, device=dev, weight_plan=wplan, quant_mode="nvfp4",
        decode_config=cfg, deterministic_output=c["det"],
    )
    return api.plan(caps)


def load_routes(path, tokens, limit):
    routes = []
    for f in sorted(glob.glob(os.path.join(path, "moe-l*-c*.pt"))):
        d = torch.load(f)
        if d["tokens"] == tokens:
            routes.append((os.path.basename(f), d["ids"].to(dev, torch.int32).contiguous(),
                           d["weights"].to(dev, torch.float32).contiguous()))
        if len(routes) >= limit:
            break
    return routes


def synthetic(tokens, name):
    ids = make_routing_ids(tokens, K, E, workload=name, device=dev).to(torch.int32).contiguous()
    w = torch.softmax(torch.randn(tokens, K, device=dev), -1).contiguous()
    return [(name, ids, w)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default="/root/.cache/vllm/nsys-flashnext/moe-dump")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--configs", default="t32,t16,t64")
    ap.add_argument("--live", type=int, default=24, help="number of live route chunks to cycle")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--profile", default="", help="config to run once more under cudaProfilerStart/Stop")
    ap.add_argument("--ref", default="", help="file: save outputs of the first config if missing, else compare to it")
    args = ap.parse_args()

    torch.manual_seed(0)
    wplan, experts = make_weights()
    T = args.tokens
    x = (torch.randn(T, H, device=dev) * 0.125).to(torch.bfloat16)
    sets = {"live": load_routes(args.routes, T, args.live),
            "shared_40": synthetic(T, "shared_40"), "disjoint": synthetic(T, "disjoint")}
    print(f"live route chunks: {len(sets['live'])}")
    ref_out = torch.load(args.ref) if args.ref and os.path.exists(args.ref) else {}
    save_ref = bool(args.ref) and not ref_out
    for token in args.configs.split(","):
        c = parse(token)
        try:
            plan = build(wplan, T, c)
        except Exception as exc:
            print(f"{token:12s} plan failed: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=dev) for s in plan.scratch_specs())
        out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
        line = [f"{token:12s}"]
        for sname, routes in sets.items():
            binds = []
            for _, ids, w in routes:
                binds.append(plan.bind(scratch=scratch, a=x, experts=experts, topk_weights=w,
                                       topk_ids=ids, output=out, input_scales_static=True))
            try:
                for b in binds[:2]:
                    api.run(binding=b)
                torch.cuda.synchronize()
            except Exception as exc:
                line.append(f"{sname}: run failed {type(exc).__name__}: {str(exc)[:120]}")
                continue
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(args.iters):
                for b in binds:
                    api.run(binding=b)
            e.record(); torch.cuda.synchronize()
            us = s.elapsed_time(e) * 1000 / (args.iters * len(binds))
            # equivalence vs the first config on the first route set
            api.run(binding=binds[0]); torch.cuda.synchronize()
            o = out.float().clone()
            key = sname
            if key not in ref_out:
                ref_out[key] = o
                err = "ref"
            else:
                r = ref_out[key].to(o.device)
                err = f"rel {((o - r).norm() / r.norm()).item():.2e}"
            line.append(f"{sname} {us:8.0f} us ({err})")
        print(" | ".join(line), flush=True)
        if save_ref and ref_out:
            torch.save({k: v.cpu() for k, v in ref_out.items()}, args.ref)
            save_ref = False
        if args.profile == token:
            torch.cuda.cudart().cudaProfilerStart()
            api.run(binding=binds[0])
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        del scratch, binds, plan
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
