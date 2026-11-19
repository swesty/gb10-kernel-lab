"""Plain / strict JSON-schema / forced tool-call smoke checks against Flash Next :8112 (thinking off)."""
import json, urllib.request
URL = "http://127.0.0.1:8112/v1/chat/completions"
M = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
def post(body):
    body.update(model=M, temperature=0, seed=42, chat_template_kwargs={"enable_thinking": False})
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300)
    return json.load(r)["choices"][0]["message"]
ok = True
m = post({"messages": [{"role": "user", "content": "In one sentence, what does a GPU's L2 cache do?"}], "max_tokens": 80})
print("plain:", (m.get("content") or "").strip()[:160])
schema = {"type": "object", "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"},
          "landmarks": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
          "required": ["city", "population_millions", "landmarks"], "additionalProperties": False}
m = post({"messages": [{"role": "user", "content": "Give facts about Paris."}], "max_tokens": 200,
          "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": schema, "strict": True}}})
try:
    d = json.loads(m["content"]); assert set(d) == {"city", "population_millions", "landmarks"} and len(d["landmarks"]) >= 2
    print("json_schema: PASS", json.dumps(d)[:140])
except Exception as e:
    ok = False; print("json_schema: FAIL", e, (m.get("content") or "")[:200])
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                         "required": ["city", "unit"]}}}]
m = post({"messages": [{"role": "user", "content": "What's the weather in Tokyo in celsius?"}], "max_tokens": 200,
          "tools": tools, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}})
try:
    tc = m["tool_calls"][0]["function"]; args = json.loads(tc["arguments"])
    assert tc["name"] == "get_weather" and args.get("city") and args.get("unit") in ("c", "f")
    print("forced tool call: PASS", tc["name"], args)
except Exception as e:
    ok = False; print("forced tool call: FAIL", e, str(m)[:200])
m = post({"messages": [{"role": "user", "content": "What's the weather in Oslo? Use the tool."}], "max_tokens": 200, "tools": tools, "tool_choice": "auto"})
print("auto tool call:", "PASS" if m.get("tool_calls") else "no call", (m.get("tool_calls") or [{}])[0].get("function"))
print("ALL PASS" if ok else "FAILURES")
