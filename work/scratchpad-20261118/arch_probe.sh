cd /usr/local/lib/python3.12/dist-packages
python3 - <<'P'
import cutlass.cute as cute, inspect
names = [n for n in dir(cute.arch) if any(k in n.lower() for k in ("mbarrier", "fence", "vote", "ballot", "popc", "elect", "sync_warp", "lane"))]
print(names)
for n in ("mbarrier_init", "mbarrier_arrive", "mbarrier_arrive_and_expect_tx", "mbarrier_expect_tx", "mbarrier_wait", "mbarrier_try_wait", "fence_proxy", "fence_view_async_shared", "mbarrier_init_fence", "vote_ballot_sync"):
    f = getattr(cute.arch, n, None)
    if f is not None:
        try: print(n, inspect.signature(f))
        except Exception as e: print(n, "sig?", e)
P
echo "--- fence_proxy usage in b12x"; grep -rn "fence_proxy\|fence_view_async_shared\|mbarrier_arrive_and_expect_tx" --include=*.py b12x | head -8
