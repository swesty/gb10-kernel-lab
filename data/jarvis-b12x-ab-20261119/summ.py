import json,sys
for a in sys.argv[1:]:
    try: d=json.load(open(f"{a}/summary.json"))
    except Exception as e: print(a,"no summary",e); continue
    print(a, "c1 tps/req %.1f acc %.3f | c4 tps/req %.1f agg %.1f acc %.3f | json %s tool %s"%(d["c1"]["mean_tps_per_req"],d["c1"]["accept_rate"] or 0,d["c4"]["mean_tps_per_req"],d["c4"]["agg_tps"],d["c4"]["accept_rate"] or 0,d["json_ok"],d["tool_ok"]))
