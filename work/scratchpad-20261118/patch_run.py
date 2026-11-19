p = "/home/spark/ncu-side/run.py"; s = open(p).read()
s = s.replace('SNAP = "/root/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/6909a5bed089a48fa07e956d3915af2537de9368"',
              'SNAP = "/work/model"  # 4-layer config (3 GDN + 1 full attention) of the 6909a5be snapshot; dummy weights')
s = s.replace('load_format="b12x"', 'load_format="dummy"')
s = s.replace('os.environ.get("GMU", "0.75")', 'os.environ.get("GMU", "0.20")')
s = s.replace('the real Qwen3.8-Flash-Next-NVFP4 (48 layers, real weights, MTP k=4)', 'Qwen3.8-Flash-Next-NVFP4 cut to 4 layers, dummy weights, MTP k=4')
open(p, "w").write(s)
