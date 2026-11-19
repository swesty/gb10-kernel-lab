"""Turn `path:line` citations in REPORT.md into GitHub permalinks at the pinned commits.

Each citation is resolved against the local clone of the repository, so a link is
only written when the file exists at that commit and the cited line is in range.
Anything ambiguous or unresolvable is left as plain text and reported.
"""
import re, subprocess, sys, pathlib, collections

CLONE = pathlib.Path(sys.argv[1])
REPORT = pathlib.Path(sys.argv[2])

REPOS = {
    "vllm":   ("vllm-project/vllm", "vllm"),
    "sglang": ("sgl-project/sglang", "sglang"),
    "fi":     ("flashinfer-ai/flashinfer", "flashinfer"),
    "fa":     ("Dao-AILab/flash-attention", "flash-attention"),
    "fork":   ("vllm-project/flash-attention", "vllm-flash-attention"),
    "cutlass":("NVIDIA/cutlass", "cutlass"),
}
SHA, FILES, LINES = {}, {}, {}
for k, (_, d) in REPOS.items():
    p = CLONE / d
    SHA[k] = subprocess.check_output(["git", "-C", p, "rev-parse", "HEAD"], text=True).strip()
    FILES[k] = subprocess.check_output(["git", "-C", p, "ls-files"], text=True).split("\n")

def nlines(k, path):
    key = (k, path)
    if key not in LINES:
        try:
            LINES[key] = sum(1 for _ in open(CLONE / REPOS[k][1] / path, "rb"))
        except OSError:
            LINES[key] = 0
    return LINES[key]

# Per-section repository preference and default directories for short names.
SECTIONS = [  # (section-number prefix, repo order, default dirs); first match wins, most specific first
    ("2",   ["cutlass", "fi", "sglang"], ["", "include/cute/arch", "include/cutlass/arch", "include/cutlass"]),
    ("4.1", ["fa", "vllm"], ["csrc/flash_attn", "csrc/flash_attn/src", "vllm/v1/attention/backends"]),
    ("4.2", ["fi"], ["include/flashinfer/attention", "include/flashinfer", "flashinfer"]),
    ("4.3", ["fi", "vllm"], ["csrc/xqa", "flashinfer", "flashinfer/jit", "vllm/v1/attention/backends"]),
    ("4.4", ["fi"], ["flashinfer", "flashinfer/jit/attention"]),
    ("4.5v",["vllm"], ["vllm/v1/attention/backends", "vllm/utils"]),
    ("4.5", ["fi", "vllm"], ["flashinfer/experimental/b12x/attention/paged", "flashinfer/experimental/b12x/_lib", "flashinfer/experimental/b12x"]),
    ("4.6", ["fa"], ["flash_attn/cute"]),
    ("4.7", ["sglang"], ["python/sglang/kernels/ops/attention/fa4_sm120", "python/sglang/kernels/ops/attention/flash_attn/cute", "python/sglang/kernels/ops/attention", "python/sglang/srt/arg_groups", "python/sglang/srt/layers/attention"]),
    ("4.8", ["vllm", "sglang"], ["vllm/v1/attention/backends"]),
    ("5",   ["vllm", "fi"], ["vllm/v1/attention/backends", "vllm/v1/attention", "vllm/utils", "vllm", "flashinfer"]),
    ("6",   ["sglang"], ["python/sglang/srt/arg_groups", "python/sglang/srt/layers/attention", "python/sglang/srt/utils", "python/sglang/srt", "python/sglang/kernels/ops/attention/flash_attn/cute"]),
    ("7.1", ["vllm"], ["vllm/v1/attention/backends", "vllm/v1/attention/backends/mla", "vllm"]),
    ("7.2", ["sglang"], ["python/sglang/srt/arg_groups", "python/sglang/srt/layers/attention", "python/sglang/kernels/ops/attention", "python/sglang/srt"]),
    ("7.3", ["fi"], ["flashinfer", "csrc", "include"]),
    ("8",   ["sglang", "fi"], ["python/sglang/kernels/ops/attention", "python/sglang/srt/layers/attention"]),
    ("10",  ["fi", "sglang", "vllm"], ["flashinfer/experimental/b12x/attention/paged", "python/sglang", "python/sglang/kernels/ops/attention", "vllm"]),
    ("12",  ["vllm", "fi", "sglang"], ["vllm/v1/attention", "vllm/v1/attention/backends", "vllm", "flashinfer/experimental/b12x/attention/paged", "python/sglang/kernels/ops/attention/flash_attn/cute"]),
    ("13",  ["vllm", "cutlass", "sglang"], ["", "include/cutlass/arch"]),
]

def context(num, h4):
    key = "4.5v" if h4.startswith("#### How vLLM drives b12x") else num
    for pre, repos, dirs in SECTIONS:
        if key == pre or key.startswith(pre + "."):
            return repos, dirs
    return ["vllm", "sglang", "fi", "fa", "cutlass"], [""]

PREFIXES = [("cutlass/", "cutlass"), ("sglang/", "sglang"), ("flashinfer/experimental", None)]

def candidates(k, cited):
    c = cited
    if k == "cutlass" and c.startswith("cutlass/"): c = c[len("cutlass/"):]
    if k == "cutlass" and c.startswith("cute/"): c = "include/" + c
    if k == "sglang" and c.startswith("sglang/"): c = c[len("sglang/"):]
    if k == "sglang" and c.startswith("srt/"): c = "python/sglang/" + c
    if k == "sglang" and c.startswith("kernels/"): c = "python/sglang/" + c
    if k == "sglang" and c.startswith(("dsv4/", "fa4_sm120/")):
        c = c  # resolved by suffix
    return [f for f in FILES[k] if f == c or f.endswith("/" + c)]

def resolve(cited, prefs, dirs, forced=None):
    order = [forced] if forced else prefs
    if "..." in cited:                      # abbreviated path: match on the tail after "..."
        cited = cited.split("...")[-1].lstrip("/")
    # 1. exact path, or exact path under one of the section's default directories
    for k in order:
        files = set(FILES[k])
        norm = candidates(k, cited)
        for d in dirs:
            exact = f"{d}/{cited}" if d else cited
            if exact in files:
                return k, exact
        for f in norm:
            if f == cited or f in (cited[len("cutlass/"):],):
                return k, f
    # 2. unique suffix match, in the section's repository order
    for k in order:
        m = candidates(k, cited)
        if len(m) == 1:
            return k, m[0]
        if len(m) > 1:
            under = [f for f in m for d in dirs if d and f.startswith(d + "/")]
            if len(set(under)) == 1:
                return k, under[0]
    seen = {k: candidates(k, cited)[:3] for k in order if candidates(k, cited)}
    return None, f"ambiguous {seen}" if seen else "not found"

SPAN = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|cuh|cu|hpp|h|cpp|md|inl|txt|toml|yaml|mdx))(?::([0-9][0-9 ,\-]*))?`")

out, heading, num, h4 = [], "", "", ""
stats = collections.Counter()
problems = []
in_code = False
for line in REPORT.read_text().split("\n"):
    if line.startswith("```"):
        in_code = not in_code
    if in_code or line.startswith("```"):
        out.append(line); continue
    if line.startswith("#"):
        heading = line
        mnum = re.match(r"#{2,3} (\d+(?:\.\d+)?)", line)
        if line.startswith(("## ", "### ")):
            h4 = ""
            if mnum: num = mnum.group(1)
        elif line.startswith("#### "):
            h4 = line
        out.append(line); continue
    if heading.startswith("## 14"):
        out.append(line); continue
    prefs, dirs = context(num, h4)

    def sub(m):
        cited, lines_spec = m.group(1), m.group(2)
        before = line[max(0, m.start() - 6):m.start()]
        forced = "fork" if "fork " in before else None
        k, path = resolve(cited, prefs, dirs, forced)
        if k is None:
            stats["unresolved"] += 1
            problems.append(f"{heading[:40]} | {m.group(0)} | {path}")
            return m.group(0)
        anchor = ""
        if lines_spec:
            first = re.match(r"\s*(\d+)(?:-(\d+))?", lines_spec)
            assert first is not None
            a, b = int(first.group(1)), first.group(2)
            n = nlines(k, path)
            if a > n or (b and int(b) > n):
                stats["line_out_of_range"] += 1
                problems.append(f"{heading[:40]} | {m.group(0)} | {path} has {n} lines")
                return m.group(0)
            anchor = f"#L{a}" + (f"-L{b}" if b else "")
        stats["linked"] += 1
        stats[k] += 1
        url = f"https://github.com/{REPOS[k][0]}/blob/{SHA[k]}/{path}{anchor}"
        return f"[{m.group(0)}]({url})"

    out.append(SPAN.sub(sub, line))

REPORT.write_text("\n".join(out))
print(dict(stats))
print("\n".join(problems))
