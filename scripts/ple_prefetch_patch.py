"""PLE disk-row prefetch for chunked prefill (switch VLLM_PLE_PREFETCH=1; default off = stock behaviour).

While prefill chunk N runs on the GPU, read chunk N+1's PLE n-gram rows from disk in a background thread, so
chunk N+1 finds them already staged. Correctness does not depend on the prediction: every disk transaction
compares its real row IDs with the IDs whose rows are resident in the staging cache and reads from disk on any
mismatch.

Files (each backed up once as .orig; the patch applies to the current content, so earlier patches survive):
- b12x/sequence/_shared/disk_table.py   DiskRowCache: resident-ID tracking, read_rows_cached(), prefetch()
- b12x/sequence/ple_embedding/_disk.py  DiskTable._run uses read_rows_cached(); prefetch mode hashes + reads only
- vllm/.../qwen4_exp/nvidia/b12x_ple.py embedding.prefetch_disk()
- vllm/.../qwen4_exp/nvidia/model_state.py  predicts the next chunk (one prefill request in the batch)
Usage (in the container): python3 ple_prefetch_patch.py
"""
import os
import shutil

import b12x.sequence._shared.disk_table as dt
import b12x.sequence.ple_embedding._disk as dk
import vllm.models.qwen4_exp.nvidia.b12x_ple as bp
import vllm.models.qwen4_exp.nvidia.model_state as ms

MARK = "PLE_PREFETCH_PATCH"


def patch(mod, reps):
    p = mod.__file__
    if not os.path.exists(p + ".orig"):
        shutil.copy(p, p + ".orig")
    s = open(p).read()
    if MARK in s:
        print("already patched", p)
        return
    for old, new in reps:
        assert s.count(old) == 1, (p, old[:80])
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("patched", p)


# ---- b12x DiskRowCache ------------------------------------------------------------------------------------
patch(dt, [
    ("        self._cache_used = False\n        self._closed = False\n",
     "        self._cache_used = False\n        self._closed = False\n"
     "        # PLE_PREFETCH_PATCH: IDs whose rows are resident in the staging cache, and the prefetch worker.\n"
     "        self._resident_ids = None\n"
     "        self._prefetch_thread = None\n"
     "        self._prefetch_ids_host = torch.empty((self.max_lookups,), dtype=torch.int64, device=\"cpu\", pin_memory=True)\n"
     "        self.prefetch_stats = {\"hits\": 0, \"misses\": 0, \"prefetches\": 0, \"prefetch_errors\": 0}\n"),
    ("    def read_rows(self, ids: torch.Tensor, count: int) -> None:\n"
     "        self._stage_ids(ids, count)\n"
     "        self._read_staged(count)\n",
     "    def read_rows(self, ids: torch.Tensor, count: int) -> None:\n"
     "        self._stage_ids(ids, count)\n"
     "        self._read_staged(count)\n"
     "        self._resident_ids = None\n"
     "\n"
     "    def read_rows_cached(self, ids: torch.Tensor, count: int) -> None:\n"
     "        \"\"\"Like read_rows, but skip the disk read when the staged rows already belong to exactly these IDs.\"\"\"\n"
     "        self._stage_ids(ids, count)\n"
     "        with torch.cuda.device(self.device):\n"
     "            self._ids_ready.synchronize()\n"
     "        resident = self._resident_ids\n"
     "        if resident is not None and resident.shape[0] == count and count > 0:\n"
     "            import numpy as _np\n"
     "            if _np.array_equal(resident, self.ids_host[:count].numpy()):\n"
     "                self.prefetch_stats[\"hits\"] += 1\n"
     "                return\n"
     "        self.prefetch_stats[\"misses\"] += 1\n"
     "        self._resident_ids = None\n"
     "        self._read_staged(count)\n"
     "        self._resident_ids = self.ids_host[:count].numpy().copy()\n"
     "\n"
     "    def prefetch(self, ids: torch.Tensor, count: int) -> None:\n"
     "        \"\"\"Read the rows for device IDs ``ids[:count]`` in a background thread.\n"
     "\n"
     "        Called on the stream that produced ``ids``, after the previous lookup was enqueued on it: the worker\n"
     "        waits for an event recorded here (so every earlier reader of the staging cache has finished), then\n"
     "        takes the cache lock, reads the rows and records them as resident.\n"
     "        \"\"\"\n"
     "        import threading as _th\n"
     "        count = operator.index(count)\n"
     "        if not 0 < count <= self.max_lookups or self._gds is not None:\n"
     "            return\n"
     "        self.join_prefetch()\n"
     "        self._prefetch_ids_host[:count].copy_(ids.view(-1)[:count], non_blocking=True)\n"
     "        ready = torch.cuda.Event()\n"
     "        ready.record(torch.cuda.current_stream(self.device))\n"
     "\n"
     "        def work():\n"
     "            try:\n"
     "                ready.synchronize()\n"
     "                with self._lock:\n"
     "                    self._resident_ids = None\n"
     "                    self.ids_host[:count].copy_(self._prefetch_ids_host[:count])\n"
     "                    self._native.ple_reader_run(self._reader, self._ids_buffer, self._weight_buffer,\n"
     "                                                self._scale_buffer, count)\n"
     "                    self._resident_ids = self.ids_host[:count].numpy().copy()\n"
     "                    self.prefetch_stats[\"prefetches\"] += 1\n"
     "            except Exception:  # noqa: BLE001 - a failed prefetch only costs the normal read later\n"
     "                self._resident_ids = None\n"
     "                self.prefetch_stats[\"prefetch_errors\"] += 1\n"
     "\n"
     "        self._prefetch_thread = _th.Thread(target=work, name=\"ple-prefetch\", daemon=True)\n"
     "        self._prefetch_thread.start()\n"
     "\n"
     "    def join_prefetch(self) -> None:\n"
     "        thread = self._prefetch_thread\n"
     "        if thread is not None:\n"
     "            thread.join()\n"
     "            self._prefetch_thread = None\n"),
])

# ---- b12x DiskTable ---------------------------------------------------------------------------------------
patch(dk, [
    ("    def _run(self, binding: Binding, *, state, token_count: int) -> None:\n"
     "        layout = self.layout\n"
     "        with self._cache.transaction():\n"
     "            state.hash_state.run(binding._hash_binding, token_count=token_count)\n"
     "            self._cache.read_rows(binding._ids, token_count * layout.head_count)\n",
     "    def _run(self, binding: Binding, *, state, token_count: int) -> None:\n"
     "        layout = self.layout\n"
     "        # PLE_PREFETCH_PATCH: prefetch mode hashes the predicted batch and reads its rows in the background.\n"
     "        if getattr(self, \"_prefetch_next\", False):\n"
     "            self._prefetch_next = False\n"
     "            if token_count:\n"
     "                state.hash_state.run(binding._hash_binding, token_count=token_count)\n"
     "                self._cache.prefetch(binding._ids, token_count * layout.head_count)\n"
     "            return\n"
     "        with self._cache.transaction():\n"
     "            state.hash_state.run(binding._hash_binding, token_count=token_count)\n"
     "            self._cache.read_rows_cached(binding._ids, token_count * layout.head_count)\n"),
])

# ---- vLLM PLE embedding -----------------------------------------------------------------------------------
patch(bp, [
    ("    def prepare_dummy_output(self, num_tokens: int) -> None:\n",
     "    def prefetch_disk(\n"
     "        self,\n"
     "        input_ids: torch.Tensor,\n"
     "        query_start_loc: torch.Tensor,\n"
     "        ngram_context: torch.Tensor,\n"
     "    ) -> None:\n"
     "        \"\"\"PLE_PREFETCH_PATCH: stage the next batch's disk rows in the background (outside graphs).\"\"\"\n"
     "        disk_table = self.ngram_embedding.disk_table\n"
     "        if disk_table is None or not self.requires_disk_preparation:\n"
     "            return\n"
     "        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():\n"
     "            return\n"
     "        self._plan_for(input_ids.numel())\n"
     "        token_count = self._prepare_inputs(input_ids, query_start_loc, ngram_context)\n"
     "        disk_table._prefetch_next = True\n"
     "        _b12x_module(\"ple_embedding\").run(\n"
     "            self._bind_embedding(token_count), token_count=token_count\n"
     "        )\n"
     "\n"
     "    def prepare_dummy_output(self, num_tokens: int) -> None:\n"),
])

# ---- vLLM model state: predict the next prefill chunk ---------------------------------------------------
patch(ms, [
    ("        for embedding in self.disk_embeddings:\n"
     "            embedding.prepare_disk(\n"
     "                input_batch.input_ids, query_start_loc, ngram_context\n"
     "            )\n",
     "        for embedding in self.disk_embeddings:\n"
     "            embedding.prepare_disk(\n"
     "                input_batch.input_ids, query_start_loc, ngram_context\n"
     "            )\n"
     "        # PLE_PREFETCH_PATCH: with one request still prefilling, its next chunk is (almost always) the next\n"
     "        # prompt slice; stage that chunk's PLE rows while this step runs. Misses fall back to a normal read.\n"
     "        if self.disk_embeddings and _PLE_PREFETCH and input_batch.num_reqs == 1:\n"
     "            self._ple_prefetch_next_chunk(input_batch, req_states, query_start_loc, ngram_context)\n"),
    ("    def prepare_dummy_inputs(\n",
     "    def _ple_prefetch_next_chunk(self, input_batch, req_states, query_start_loc, ngram_context) -> None:\n"
     "        start = int(input_batch.num_computed_tokens_np[0]) + int(input_batch.num_scheduled_tokens[0])\n"
     "        prefill_len = int(input_batch.prefill_len_np[0])\n"
     "        length = min(prefill_len - start, int(req_states.max_num_batched_tokens))\n"
     "        if length <= 0 or start < self.ngram_context_len:\n"
     "            return\n"
     "        if getattr(self, \"_ple_pf_qsl\", None) is None:\n"
     "            self._ple_pf_qsl = torch.empty_like(query_start_loc)\n"
     "            self._ple_pf_ctx = torch.empty_like(ngram_context)\n"
     "        req = int(input_batch.idx_mapping_np[0])\n"
     "        tokens = req_states.all_token_ids.gpu[req]\n"
     "        qsl = self._ple_pf_qsl\n"
     "        qsl.fill_(length)\n"
     "        qsl[0] = 0\n"
     "        ctx = self._ple_pf_ctx\n"
     "        ctx.fill_(self.ngram_eos_token_id)\n"
     "        ctx[0].copy_(tokens[start - self.ngram_context_len : start])\n"
     "        for embedding in self.disk_embeddings:\n"
     "            embedding.prefetch_disk(tokens[start : start + length], qsl, ctx)\n"
     "\n"
     "    def prepare_dummy_inputs(\n"),
    ("from .backend import uses_b12x\n",
     "from .backend import uses_b12x\n"
     "import os as _os_pf\n_PLE_PREFETCH = _os_pf.environ.get(\"VLLM_PLE_PREFETCH\", \"0\") == \"1\"\n"),
])
