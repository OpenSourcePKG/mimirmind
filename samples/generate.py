#!/usr/bin/env python3
"""Deterministic sample-prompt generator for mimirmind bench scripts.

Emits `samples/prompts/{en,de}/{shape}-{len}.{txt,json}` and
`samples/chats/{en,de}/chat-multiturn-{len}.jsonl`, plus a top-level
`samples/manifest.json` that bench scripts consume.

Design goals:

- Reproducible. Fixed seed, no third-party deps, procedural filler
  paragraphs. Same input → identical bytes.
- Length-targeted. Prompt files aim at a target token count using a
  per-language chars-per-token ratio (EN ~4.0, DE ~3.0 for Llama /
  Gemma BPE). Actual token count on the server side will vary a bit,
  but each file's length grows monotonically with the target so a
  length sweep still makes sense.
- Factor matrix. shapes (instruct, rag, needle, code, summarize,
  chat-multiturn) x lengths (120..32k) x languages (en, de). Sparse
  where combinations don't make sense (32k instruct is nonsense).
- No real-world knowledge in the filler — templated fake facts, so a
  memorised passage can never accidentally boost a decode.

Regenerate:
    python3 samples/generate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

SEED = 20260720
CHARS_PER_TOKEN = {"en": 4.0, "de": 3.0}

# Length buckets. Values are approximate target token counts.
LENGTHS = {
    "120":  120,
    "250":  250,
    "512":  512,
    "1k":   1024,
    "2k":   2048,
    "4k":   4096,
    "8k":   8192,
    "16k": 16384,
    "32k": 32768,
}

# Which (shape, length) combinations exist. Sparse on purpose — a
# 120-token needle-in-haystack is nonsense, and a 32k instruct prompt
# is a stress test for the wrong axis.
SHAPE_LENGTHS = {
    "instruct":  ["120", "250", "512"],
    "rag":       ["512", "1k", "2k", "4k", "8k", "16k", "32k"],
    "needle":    ["1k", "2k", "4k", "8k", "16k", "32k"],
    "code":      ["250", "512", "1k", "2k"],
    "summarize": ["1k", "2k", "4k", "8k", "16k", "32k"],
    "chat-multiturn": ["512", "1k", "2k", "4k"],
}

# Decode-length matrix per shape. Bench scripts iterate these as
# max_new_tokens values. Kept coarse to avoid combinatorial blow-up.
DECODE_TARGETS = {
    "instruct":       [32, 128],
    "rag":            [128, 512],
    "needle":         [32],
    "code":           [512, 2048],
    "summarize":      [128, 512],
    "chat-multiturn": [128, 512],
}

# --- Filler templates. Deterministic, procedural, no real knowledge. ---

FILLER_EN = [
    ("In the region of {region}, the annual production of {product} "
     "reached {n} units in year {year}, according to the {agency} "
     "statistical bureau."),
    ("The {agency} concluded that {product} exports to {region} grew "
     "by {n} percent between {year} and {year2}, driven by policy "
     "reforms and the widening domestic supply chain."),
    ("A {year} survey by the {agency} found that {n} percent of "
     "{region} households own at least one {product}, up from a "
     "decade prior when adoption was largely urban."),
    ("During the {year} fiscal year, the {agency} allocated {n} "
     "million in grants to {product}-related research in {region}, "
     "prioritising small and medium enterprises over incumbents."),
    ("By {year2}, the projected demand for {product} in {region} is "
     "expected to exceed {n} thousand units, per the {agency}'s "
     "baseline scenario and holding tariffs constant."),
    ("Analysts at the {agency} note that {region} continues to "
     "consolidate its role as a hub for {product}, with logistics "
     "throughput rising {n} percent in {year} alone."),
    ("A joint report between the {agency} and the trade council "
     "attributes the {year} uptick in {product} shipments to {region} "
     "to a {n}-point reduction in cross-border customs friction."),
]

FILLER_DE = [
    ("In der Region {region} erreichte die jaehrliche Produktion von "
     "{product} im Jahr {year} laut dem Statistikbuero der {agency} "
     "eine Menge von {n} Einheiten."),
    ("Die {agency} berichtete, dass die Ausfuhren von {product} nach "
     "{region} zwischen {year} und {year2} um {n} Prozent zunahmen, "
     "bedingt durch politische Reformen und eine breitere heimische "
     "Lieferkette."),
    ("Eine Erhebung der {agency} aus dem Jahr {year} ergab, dass in "
     "{region} {n} Prozent der Haushalte mindestens ein {product} "
     "besitzen, deutlich mehr als noch vor zehn Jahren."),
    ("Im Haushaltsjahr {year} bewilligte die {agency} Foerdermittel in "
     "Hoehe von {n} Millionen fuer Forschung im Bereich {product} in "
     "{region}, mit Vorrang fuer kleine und mittlere Unternehmen."),
    ("Bis {year2} wird der Bedarf an {product} in {region} laut "
     "Basisszenario der {agency} voraussichtlich {n} Tausend "
     "Einheiten uebersteigen, sofern die Zoelle stabil bleiben."),
    ("Analysten der {agency} weisen darauf hin, dass sich {region} "
     "weiter als Drehscheibe fuer {product} etabliert, mit einer "
     "Steigerung des Umschlags um {n} Prozent allein im Jahr {year}."),
    ("Ein gemeinsamer Bericht der {agency} und des Handelsrats fuehrt "
     "den Anstieg der {product}-Lieferungen nach {region} im Jahr "
     "{year} auf einen Rueckgang der Zollreibung um {n} Punkte "
     "zurueck."),
]

REGIONS = ["Aquileia", "Belrion", "Cathenor", "Dornhelm", "Ellisar",
           "Faralind", "Gorwyn", "Halmuth", "Ithelion", "Jarvath",
           "Kernewic", "Lomoria", "Mardun", "Nevaris", "Ostmark",
           "Perendol", "Quorath", "Rhesundir", "Sarnwood", "Thelmara"]

PRODUCTS_EN = ["photonic sensors", "biosynthetic wool",
               "modular gearboxes", "carbon batteries",
               "ceramic bearings", "hyperspectral cameras",
               "hydrogel packaging", "nanoporous filters",
               "phase-change alloys", "shape-memory springs",
               "cryogenic pumps", "recyclable turbines"]

PRODUCTS_DE = ["Photonik-Sensoren", "biosynthetische Wolle",
               "modulare Getriebe", "Karbon-Batterien",
               "Keramiklager", "hyperspektrale Kameras",
               "Hydrogel-Verpackungen", "nanoporoese Filter",
               "Phasenwechsel-Legierungen", "Formgedaechtnis-Federn",
               "Kryopumpen", "recyclingfaehige Turbinen"]

AGENCIES_EN = ["Northern Trade Council",
               "Federated Metrics Office",
               "Continental Registry",
               "Alliance Bureau of Analysis",
               "Union of Regional Assemblies"]

AGENCIES_DE = ["Nordischen Handelsrats",
               "Foederierten Statistikamts",
               "Kontinentalen Register",
               "Allianz-Analysebueros",
               "Union der Regionalversammlungen"]

# --- Instruction / task heads (short prefix that anchors the task). ---

INSTRUCT_EN = [
    "Write a one-sentence summary of the color blue.",
    "Explain in three bullet points what a lookup table is.",
    "List two upsides and two downsides of async I/O.",
    "Compose a haiku about a slow database migration.",
    "Describe a Merkle tree using no more than 40 words.",
]

INSTRUCT_DE = [
    "Schreibe eine ein-saetzige Zusammenfassung der Farbe Blau.",
    "Erklaere in drei Stichpunkten, was eine Nachschlagetabelle ist.",
    "Nenne zwei Vorteile und zwei Nachteile von asynchronem I/O.",
    "Verfasse ein Haiku ueber eine langsame Datenbank-Migration.",
    "Beschreibe einen Merkle-Baum in hoechstens 40 Woertern.",
]

SUMMARIZE_HEAD = {
    "en": ("Below is a long article. Summarise the main argument in "
           "three paragraphs, then list two weaknesses in the "
           "author's reasoning.\n\n"),
    "de": ("Nachfolgend ein laengerer Artikel. Fasse das Hauptargument "
           "in drei Absaetzen zusammen und nenne anschliessend zwei "
           "Schwaechen der Argumentation.\n\n"),
}

RAG_HEAD = {
    "en": ("You are a helpful research assistant. Below are excerpts "
           "from internal documents. Answer the question using ONLY "
           "the excerpts. If the excerpts do not contain the answer, "
           "say so explicitly.\n\n=== EXCERPTS ===\n\n"),
    "de": ("Du bist ein hilfreicher Recherche-Assistent. Nachfolgend "
           "Auszuege aus internen Dokumenten. Beantworte die Frage "
           "AUSSCHLIESSLICH auf Basis dieser Auszuege. Falls die "
           "Antwort nicht enthalten ist, sage das ausdruecklich.\n\n"
           "=== AUSZUEGE ===\n\n"),
}

RAG_TAIL = {
    "en": ("\n\n=== QUESTION ===\n\nWhich product had the largest "
           "growth in the region of Aquileia between years 2049 and "
           "2054, and by roughly how much?"),
    "de": ("\n\n=== FRAGE ===\n\nWelches Produkt hatte in der Region "
           "Aquileia zwischen 2049 und 2054 das groesste Wachstum, "
           "und in welcher Groessenordnung?"),
}

NEEDLE_QUESTION = {
    "en": ("\n\n=== QUESTION ===\n\nSomewhere in the text above, a "
           "SECRET_CODE was placed. Reproduce the code exactly, and "
           "output nothing else."),
    "de": ("\n\n=== FRAGE ===\n\nIrgendwo im obigen Text wurde ein "
           "SECRET_CODE platziert. Gib den Code exakt wieder und "
           "sonst nichts."),
}

# Code shape uses a synthetic, deterministic Python-like body. Keeps
# the prompt inside a realistic domain (code review / refactor) so
# tokenizer behaviour is representative.

CODE_HEAD = {
    "en": ("Review the following Python module and propose a "
           "refactor that (a) removes duplicated dictionary lookups, "
           "(b) hoists the datetime.now() call out of the loop, and "
           "(c) adds type hints. Return the full rewritten module.\n\n"
           "```python\n"),
    "de": ("Ueberarbeite das folgende Python-Modul so, dass (a) "
           "doppelte Dictionary-Zugriffe entfernt werden, (b) der "
           "datetime.now()-Aufruf aus der Schleife gehoben wird und "
           "(c) Typannotationen ergaenzt werden. Gib das komplette "
           "ueberarbeitete Modul zurueck.\n\n"
           "```python\n"),
}

CODE_TAIL = "\n```"

CODE_BODY_LINES = [
    "def process(records, config, sink):",
    "    result = []",
    "    for r in records:",
    "        ts = datetime.now().isoformat()",
    "        if config['flags']['strict'] and not r.get('id'):",
    "            continue",
    "        row = {",
    "            'id':     r.get('id'),",
    "            'ts':     ts,",
    "            'name':   config['naming']['prefix'] + r.get('name', ''),",
    "            'weight': r.get('weight', 0) * config['weights']['scale'],",
    "        }",
    "        if config['flags']['dedupe']:",
    "            if row['id'] in seen:",
    "                continue",
    "            seen.add(row['id'])",
    "        sink.write(row)",
    "        result.append(row)",
    "    return result",
    "",
    "def audit(records, config):",
    "    for r in records:",
    "        if config['flags']['strict'] and r.get('weight', 0) < 0:",
    "            log.error('negative weight for %s', r.get('id'))",
    "    return True",
    "",
]

# Chat-multiturn prior-message templates. The final message is the
# actual query the assistant should answer.

CHAT_SYSTEM = {
    "en": ("You are a concise engineering assistant. Answer only what "
           "was asked, in plain prose, without headings."),
    "de": ("Du bist ein knapper Engineering-Assistent. Beantworte nur "
           "das, was gefragt wurde, in fliessendem Text ohne "
           "Ueberschriften."),
}

CHAT_TURNS_EN = [
    ("user", "What is the difference between prefill and decode in an "
             "LLM inference server?"),
    ("assistant", "Prefill processes the whole prompt in one large "
                  "attention pass and is compute-bound; decode "
                  "generates one token at a time and is dominated by "
                  "memory bandwidth for the weight fetch. Batching "
                  "helps decode more than prefill."),
    ("user", "And how does continuous batching change that?"),
    ("assistant", "Continuous batching lets in-flight requests share "
                  "GPU work at each decode step, so a slow decoder "
                  "can piggyback on a fast one and the collective "
                  "throughput approaches the memory-bandwidth "
                  "ceiling."),
    ("user", "OK, now walk me through what a paged KV cache is and "
             "why vLLM introduced it, in three or four sentences."),
]

CHAT_TURNS_DE = [
    ("user", "Was ist der Unterschied zwischen Prefill und Decode in "
             "einem LLM-Inferenzserver?"),
    ("assistant", "Prefill verarbeitet den gesamten Prompt in einem "
                  "grossen Attention-Pass und ist compute-limitiert; "
                  "Decode erzeugt Token fuer Token und wird von der "
                  "Speicherbandbreite fuer das Gewicht-Laden "
                  "dominiert. Batching hilft dem Decode mehr als dem "
                  "Prefill."),
    ("user", "Und was aendert Continuous Batching daran?"),
    ("assistant", "Continuous Batching laesst laufende Requests bei "
                  "jedem Decode-Schritt gemeinsam GPU-Arbeit nutzen, "
                  "sodass ein langsamer Decoder auf einem schnellen "
                  "mitfaehrt und der Gesamtdurchsatz an die "
                  "Bandbreiten-Obergrenze heranreicht."),
    ("user", "OK, erklaere jetzt in drei bis vier Saetzen, was ein "
             "paged KV-Cache ist und warum vLLM ihn eingefuehrt hat."),
]

# --- helpers ---


def rng_for(bucket: str) -> random.Random:
    """Per-bucket deterministic RNG. Same bucket → same bytes."""
    seed = int(hashlib.sha256(f"{SEED}:{bucket}".encode()).hexdigest(),
               16) & 0xFFFFFFFF
    return random.Random(seed)


def render_filler(rng: random.Random, lang: str) -> str:
    templates = FILLER_EN if lang == "en" else FILLER_DE
    products  = PRODUCTS_EN if lang == "en" else PRODUCTS_DE
    agencies  = AGENCIES_EN if lang == "en" else AGENCIES_DE
    t = rng.choice(templates)
    return t.format(
        region  = rng.choice(REGIONS),
        product = rng.choice(products),
        agency  = rng.choice(agencies),
        n       = rng.randint(3, 987),
        year    = rng.randint(2040, 2060),
        year2   = rng.randint(2061, 2075),
    )


def pad_to_target(body_head: str, body_tail: str, target_tokens: int,
                  lang: str, rng: random.Random) -> str:
    """Grow a prompt with filler paragraphs until the target token
    budget is spent. Uses per-language chars/token to estimate."""
    cpt = CHARS_PER_TOKEN[lang]
    target_chars = int(target_tokens * cpt)
    parts = [body_head]
    current = len(body_head) + len(body_tail)
    # Reserve tail budget.
    while current < target_chars:
        p = render_filler(rng, lang)
        parts.append(p + "\n\n")
        current += len(p) + 2
    parts.append(body_tail)
    return "".join(parts)


def gen_needle(rng: random.Random) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "CODE-" + "".join(rng.choices(alphabet, k=8))


def insert_needle(text: str, needle: str, position_pct: float,
                  lang: str) -> str:
    """Insert a NEEDLE marker line at roughly position_pct into text
    (rounded to the nearest paragraph boundary)."""
    marker = (f"\n\nIMPORTANT: The SECRET_CODE for this document is "
              f"{needle}. Do not lose it.\n\n") if lang == "en" else (
              f"\n\nWICHTIG: Der SECRET_CODE fuer dieses Dokument "
              f"lautet {needle}. Bitte nicht verlieren.\n\n")
    paras = text.split("\n\n")
    if not paras:
        return marker + text
    idx = max(1, min(len(paras) - 1, int(len(paras) * position_pct)))
    paras.insert(idx, marker.strip())
    return "\n\n".join(paras)


# --- shape emitters ---


def emit_instruct(lang: str, tok_target: int) -> str:
    rng = rng_for(f"instruct:{lang}:{tok_target}")
    head = rng.choice(INSTRUCT_EN if lang == "en" else INSTRUCT_DE)
    # Instruction shape stays short; we add short scene-setting so we
    # can hit 120 / 250 / 512 targets without repeating the head.
    return pad_to_target(head + "\n\nContext:\n\n" if lang == "en"
                         else head + "\n\nKontext:\n\n",
                         "", tok_target, lang, rng)


def emit_rag(lang: str, tok_target: int) -> str:
    rng = rng_for(f"rag:{lang}:{tok_target}")
    return pad_to_target(RAG_HEAD[lang], RAG_TAIL[lang],
                         tok_target, lang, rng)


def emit_needle(lang: str, tok_target: int) -> Tuple[str, str, float]:
    rng = rng_for(f"needle:{lang}:{tok_target}")
    needle = gen_needle(rng)
    position_pct = round(rng.uniform(0.15, 0.85), 3)
    body = pad_to_target(RAG_HEAD[lang], NEEDLE_QUESTION[lang],
                         tok_target, lang, rng)
    body = insert_needle(body, needle, position_pct, lang)
    return body, needle, position_pct


def emit_summarize(lang: str, tok_target: int) -> str:
    rng = rng_for(f"summarize:{lang}:{tok_target}")
    return pad_to_target(SUMMARIZE_HEAD[lang], "",
                         tok_target, lang, rng)


def emit_code(lang: str, tok_target: int) -> str:
    """Code prompt: fixed head + tail, body grows by concatenating
    numbered copies of CODE_BODY_LINES so token count scales
    predictably."""
    rng = rng_for(f"code:{lang}:{tok_target}")
    cpt = CHARS_PER_TOKEN[lang]
    target_chars = int(tok_target * cpt)
    head = CODE_HEAD[lang]
    tail = CODE_TAIL
    body_parts = []
    current = len(head) + len(tail)
    copy_idx = 0
    while current < target_chars:
        # Suffix each helper with the copy_idx so bodies aren't
        # collapsible by a smart summariser.
        renamed = [ln.replace("process", f"process_{copy_idx}")
                     .replace("audit",   f"audit_{copy_idx}")
                   for ln in CODE_BODY_LINES]
        chunk = "\n".join(renamed) + "\n\n"
        body_parts.append(chunk)
        current += len(chunk)
        copy_idx += 1
    return head + "".join(body_parts) + tail


def emit_chat(lang: str, tok_target: int) -> List[dict]:
    """Multi-turn chat as an OpenAI-style message list. History grows
    with additional filler-based user/assistant turns until the
    concatenated char budget matches target_tokens * cpt. Last
    message is a fresh user query, kept identical across lengths so
    the query itself is a constant while only history varies."""
    rng = rng_for(f"chat:{lang}:{tok_target}")
    cpt = CHARS_PER_TOKEN[lang]
    target_chars = int(tok_target * cpt)

    msgs: List[dict] = [{"role": "system", "content": CHAT_SYSTEM[lang]}]
    base_turns = CHAT_TURNS_EN if lang == "en" else CHAT_TURNS_DE
    # All except the final user query go in as prior history.
    for role, content in base_turns[:-1]:
        msgs.append({"role": role, "content": content})

    def total_chars() -> int:
        return sum(len(m["content"]) for m in msgs)

    # Grow history with filler-based Q/A pairs.
    while total_chars() < target_chars - 400:  # keep room for final Q
        q_topic = render_filler(rng, lang)
        q = (f"Given this excerpt from an internal report: \"{q_topic}\" "
             f"— what would you probe next?") if lang == "en" else (
             f"Ausgehend von diesem Ausschnitt aus einem internen "
             f"Bericht: \"{q_topic}\" — was wuerdest du als Naechstes "
             f"untersuchen?")
        a_paras = [render_filler(rng, lang) for _ in range(2)]
        a = " ".join(a_paras)
        msgs.append({"role": "user",      "content": q})
        msgs.append({"role": "assistant", "content": a})

    # Final user turn — the actual query to answer.
    msgs.append({"role": "user", "content": base_turns[-1][1]})
    return msgs


# --- IO ---


@dataclass
class ManifestEntry:
    id: str
    path: str
    shape: str
    lang: str
    length_bucket: str
    token_target: int
    char_count: int
    decode_targets: List[int]
    type: str
    extra: dict


def write_prompt(out: Path, text: str) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return len(text)


def write_needle(out: Path, prompt: str, needle: str,
                 position_pct: float) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(
        {"prompt": prompt, "needle": needle,
         "position_pct": position_pct},
        ensure_ascii=False, indent=2)
    out.write_text(blob + "\n", encoding="utf-8")
    return len(prompt)


def write_chat(out: Path, msgs: List[dict]) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(m, ensure_ascii=False) for m in msgs]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sum(len(m["content"]) for m in msgs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parent,
                    help="samples/ root (default: this file's dir)")
    args = ap.parse_args()

    root: Path = args.root
    entries: List[ManifestEntry] = []

    for shape, buckets in SHAPE_LENGTHS.items():
        for bucket in buckets:
            tok = LENGTHS[bucket]
            decode = DECODE_TARGETS[shape]
            for lang in ("en", "de"):
                if shape == "chat-multiturn":
                    out_rel = f"chats/{lang}/chat-multiturn-{bucket}.jsonl"
                    msgs = emit_chat(lang, tok)
                    n = write_chat(root / out_rel, msgs)
                    e = ManifestEntry(
                        id=f"{lang}/chat-multiturn-{bucket}",
                        path=out_rel, shape=shape, lang=lang,
                        length_bucket=bucket, token_target=tok,
                        char_count=n, decode_targets=decode,
                        type="chat",
                        extra={"n_messages": len(msgs)})
                elif shape == "needle":
                    out_rel = f"prompts/{lang}/needle-{bucket}.json"
                    prompt, needle, pos = emit_needle(lang, tok)
                    n = write_needle(root / out_rel, prompt, needle, pos)
                    e = ManifestEntry(
                        id=f"{lang}/needle-{bucket}",
                        path=out_rel, shape=shape, lang=lang,
                        length_bucket=bucket, token_target=tok,
                        char_count=n, decode_targets=decode,
                        type="needle",
                        extra={"needle_field": "needle",
                               "prompt_field": "prompt",
                               "position_pct": pos})
                else:
                    out_rel = f"prompts/{lang}/{shape}-{bucket}.txt"
                    if shape == "instruct":
                        text = emit_instruct(lang, tok)
                    elif shape == "rag":
                        text = emit_rag(lang, tok)
                    elif shape == "summarize":
                        text = emit_summarize(lang, tok)
                    elif shape == "code":
                        text = emit_code(lang, tok)
                    else:
                        raise RuntimeError(f"unknown shape {shape}")
                    n = write_prompt(root / out_rel, text)
                    e = ManifestEntry(
                        id=f"{lang}/{shape}-{bucket}",
                        path=out_rel, shape=shape, lang=lang,
                        length_bucket=bucket, token_target=tok,
                        char_count=n, decode_targets=decode,
                        type="prompt", extra={})
                entries.append(e)

    manifest = {
        "version":       1,
        "seed":          SEED,
        "chars_per_token": CHARS_PER_TOKEN,
        "shapes":        sorted(SHAPE_LENGTHS.keys()),
        "lengths":       list(LENGTHS.keys()),
        "languages":     ["en", "de"],
        "decode_targets": DECODE_TARGETS,
        "entries": [
            {
                "id":             e.id,
                "path":           e.path,
                "shape":          e.shape,
                "lang":           e.lang,
                "length_bucket":  e.length_bucket,
                "token_target":   e.token_target,
                "char_count":     e.char_count,
                "decode_targets": e.decode_targets,
                "type":           e.type,
                **e.extra,
            }
            for e in entries
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    print(f"wrote {len(entries)} sample files + manifest.json")
    print(f"total bytes: "
          f"{sum(e.char_count for e in entries):,}")


if __name__ == "__main__":
    main()