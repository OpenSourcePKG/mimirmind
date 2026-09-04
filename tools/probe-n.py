import json, ssl, sys, urllib.request
# 8.19.12 `n` gates. args: KEY
KEY = sys.argv[1]
URL = "https://localhost:8080/v1/chat/completions"
ctx = ssl.create_default_context(); ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        method="POST", headers={"Authorization": "Bearer " + KEY,
                                "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=300) as r:
            return r.status, json.loads(r.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8", "ignore"))

msgs = [{"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": "Write one creative opening sentence for a novel about a lighthouse keeper."}]

# Gate 1: n=1 unchanged shape (1 choice).
st, j = post({"model": "qwen3.6", "messages": msgs, "max_tokens": 40,
              "temperature": 0})
print("G1 n_default:", st, "choices=", len(j.get("choices", [])),
      "usage=", j.get("usage"))
ref = j["choices"][0]["message"]["content"]

# Gate 2: n=3 sampled -> 3 distinct choices, usage sums.
st, j = post({"model": "qwen3.6", "messages": msgs, "max_tokens": 40,
              "temperature": 0.9, "n": 3})
texts = [c["message"]["content"] for c in j.get("choices", [])]
idx = [c["index"] for c in j.get("choices", [])]
print("G2 n3_sampled:", st, "choices=", len(texts), "indices=", idx,
      "distinct=", len(set(texts)), "usage=", j.get("usage"))
for t in texts:
    print("   -", t.replace("\n", " ")[:100])

# Gate 3: n=3 greedy -> 3 identical choices (deterministic path shared).
st, j = post({"model": "qwen3.6", "messages": msgs, "max_tokens": 40,
              "temperature": 0, "n": 3})
texts0 = [c["message"]["content"] for c in j.get("choices", [])]
print("G3 n3_greedy:", st, "choices=", len(texts0),
      "identical=", len(set(texts0)) == 1,
      "matches_ref=", all(t == ref for t in texts0))

# Gate 4: stream + n>1 -> 400.
st, j = post({"model": "qwen3.6", "messages": msgs, "max_tokens": 8,
              "temperature": 0, "n": 2, "stream": True})
print("G4 stream_n2:", st, j.get("error", {}).get("type"))

# Gate 5: n=9 -> 400.
st, j = post({"model": "qwen3.6", "messages": msgs, "max_tokens": 8, "n": 9})
print("G5 n9:", st, j.get("error", {}).get("type"))
