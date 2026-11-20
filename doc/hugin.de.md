# Hugin — Model-Surgery-Companion für MimirMind

*Ein eingefrorenes Modell per MCP untersuchen, Verhalten lokalisieren und
editieren; jede Änderung ist eine Rune.*

**Version:** hugin 0.5.0 (2026-11)
**Englische Fassung:** [`hugin.md`](hugin.md)
**Roadmap:** Track 7 (M-Hugin + Futhark) in der Synaipse-Roadmap

---

## 1. Was Hugin ist

Hugin ist ein eigener C++-Prozess neben `mimirmind`. Er lädt dasselbe
Basismodell mit demselben Engine-Code und stellt es als **MCP-Server über
stdio** bereit. Ein Agent (Claude, über Claude Code) steuert ihn mit Tools:
Gewichte und Aktivierungen ansehen, herausfinden, wo ein Verhalten sitzt,
einen Eingriff ausprobieren und das Ergebnis als **Rune** festschreiben.

Die Regeln, unter denen Hugin arbeitet:

- **Die Basismodell-Datei wird nie verändert.** Jede Änderung ist eine
  Rune-Datei, per Fingerprint an das Basismodell gebunden, von **Futhark**
  nach dem Laden angewendet und durch Entfernen der Rune aus der Config
  wieder rückgängig gemacht.
- **Preview gleich Rune.** Jedes Preview-Tool läuft über denselben
  Engine-Pfad, den die Rune im Serving nutzt. Was eine Preview zeigt, tut
  die gestagte Rune auch.
- **Hugin aktiviert nie etwas.** Es staged, gatet und committet Runes.
  Einschalten ist eine Config-Änderung (`models[].runes`) plus Neustart,
  durch einen Menschen in einem Wartungsfenster.
- **Kein Python, kein Trainings-Framework.** Hugin wird aus den
  Engine-Quellen gebaut; der Forward ist der Serving-Forward.

Der Name folgt dem nordischen Thema: Huginn (*Gedanke*) fliegt aus und
bringt zurück, was er gefunden hat; Muninn (*Gedächtnis*, unser
Modell-Residenz-Daemon) hält, was schon da ist.

Noch nicht gebaut (Roadmap 7): Adapter-Training mit eigenem
C++-Backward-Kern (LoRA-SFT, DAgger, GKD, Pruning + Healing) und der
Input-Compression-Encoder (Entwurf:
[`hugin-input-compression.de.md`](hugin-input-compression.de.md)).

---

## 2. Bauen und starten

Hugin ist ein CMake-Target des Haupt-Trees:

```bash
cmake --build build --target hugin
```

Ein CUDA-Build kann NVFP4-Checkpoints öffnen; ein CPU-Build deckt
GGUF-Modelle ab (praktisch für schnelle Validierung auf kleinen Modellen
wie Llama 3.2 1B).

```text
hugin [--model <file.gguf | nvfp4-checkpoint-dir>] [--tokenizer-gguf <file.gguf>]
      [--rune-dir <dir>]      staged/committed/rejected Runes + Audit-Log (Default ./runes)
      [--single-engine]       Basis ODER Rune-Variante halten, nie beide
```

- `--single-engine` ist für Modelle, die nicht zweimal in den Speicher
  passen (Kolibri-1 auf 128 GB). Vergleiche werten dann zuerst die Basis
  auf allen Sets aus und laden die Variante einmal.
- Unter CUDA Hugin mit der Serving-Umgebung des Modells starten
  (`MIMIRMIND_*`-Flags), damit der Forward dem Produktivbetrieb entspricht.
- Auf GB10 immer nur ein GPU-Prozess: vor einer GPU-Hugin-Session das
  Serving stoppen.

In Claude Code registrieren:

```bash
claude mcp add hugin -- /path/to/hugin --model /path/to/model.gguf
```

Ohne `--model` zuerst `open_model` aufrufen.

---

## 3. Runes und Futhark

Eine **Rune** ist eine safetensors-Datei. Das Manifest liegt im Header
unter `__metadata__` → `mimirmind.rune`; die Tensoren sind die Payloads.
Das Manifest hält Name, Basis-Fingerprint, Herkunft (Tool, Parameter) und
eine Liste von Ops.

| Op-Typ | Wirkung | Angewendet als |
|---|---|---|
| `write_slice` | Zeilen-/Spaltenbereich eines Tensors überschreiben | Gewichtsschreiben beim Laden |
| `scale` | Zeilen oder Spalten multiplizieren (z. B. O-Proj-Spalten eines Heads) | Gewichtsschreiben beim Laden |
| `mask` | MoE-Experten abschalten: Router-Logits → −inf vor Top-K | Lade-Maske (Overlay) |
| `steer` | x += v nach Block L, jede Zeile, jeder Pfad | Residual-Overlay |
| `steer_project` | x −= α (x·u) u, u Einheitsvektor, 0 < α ≤ 1 | Residual-Overlay |
| `steer_if` | add oder project, nur Zeilen mit cos(h_Lc, c) > θ | Residual-Overlay + Device-Gate |
| `lowrank` | LoRA-Adapter auf einer Projektion: y += α·B(A x), A [rank, in], B [out, rank] | Projektions-Overlay (F32-Matmuls neben dem eingefrorenen Gewicht); nur qwen2-/llama-Backend, andere Architekturen lehnen ihn beim Laden ab |
| `write_quant_blocks` | im Format definiert | beim Laden abgelehnt (noch nicht angewendet) |

Futhark, der Applier in `mimirmind`:

1. prüft für jede Rune in Stack-Reihenfolge Basis-Fingerprint,
   Anwendbarkeit der Op, Zieltyp und -form, die Tensor-Hash-Kette pro Op
   und Payload-Größen, **bevor** irgendetwas geschrieben wird;
2. schreibt Gewichts-Ops in die geladenen Gewichte (bevor abgeleitete
   Gewichtskopien entstehen) und installiert Overlays im Arch-Backend;
3. loggt eine Zeile pro Rune und pro Overlay (Log-Tag `futhark`);
4. bricht laut ab — überspringt nie still —, wenn einem Backend eine Op
   fehlt (z. B. `steer_if` auf Level Zero oder HIP).

Aktiviert wird per Config-Eintrag am Modell:

```json
{ "id": "kolibri1", "runes": ["/path/to/runes/committed/my_rune.rune"] }
```

Eintrag entfernen und neu starten ergibt wieder exakt das Basismodell.

---

## 4. Lebenszyklus: stage → gate → commit

```text
Preview-Tools ──► rune_stage_*  ──► staged/
                                       │ rune_gate (Basis vs. Rune, Futhark-Ladepfad)
                      rot ◄────────────┤
                 rejected/             │ grün
                                       ▼
                                  rune_commit ──► committed/ ──► Config + Neustart (Mensch)
                                       ▲
                                  rune_rollback (committed → staged → rejected)
```

- **`rune_stage_*`** schreibt die Rune und eine Sidecar-Datei mit
  Parametern und Historie. Leitplanken je Typ (relative Steer-Norm,
  Head-Anteil, Scale-Obergrenze, α-Bereich der Projektion). Für
  Steer-Typen zusätzlich (7.12):
  - kein Steering nach Block 0; eine Dosis über 0.5 der Residualnorm
    braucht einen `override_reason`;
  - **Held-out-Evidenz**: Die Richtung (`steer`, `steer_project`) bzw. die
    Bedingung (`steer_if`) muss mit `holdout_positive` /
    `holdout_negative` gefittet sein — mindestens 4 je Klasse,
    Held-out-AUC ≥ `min_holdout_auc` (0.8), und bei einer Bedingung
    höchstens `max_holdout_negative_fire` (5 %) aller gegateten Zeilen der
    Held-out-Negativen über der Schwelle. Held-out-Prompts, die einen
    Fitting-Prompt wiederholen, werden abgelehnt;
  - die Shuffled-Label-Null wird berichtet, aber nicht gegatet: Bei
    d_model ≫ Prompts erreichen In-Sample-AUC und Val-Accuracy der Probe
    auch mit Zufallslabels 1.0 und sagen allein nichts aus;
  - **Kausalität**: `causality_check` muss genau diesen Eingriff (Vektor bei
    diesem α bzw. diesen Satz Head-Edits) mindestens als `weak` bewertet
    haben — Effekt ≥ 1.5× das 95. Perzentil zufälliger Eingriffe gleicher
    Größe (Zufallsrichtungen gleicher Norm bzw. zufällige andere Heads);
    `causal` verlangt zusätzlich ≥ 3× und einen Effekt, der über 25 / 50 /
    100 % Dosis wächst. Head-Edits brauchen außerdem eine gemessene Dosis
    ≤ 0.5 (Residualänderung / Residualnorm am editierten Block);
    die Baseline für einen Vektor ist die Mittelwert-Differenz seiner
    eigenen Fitting-Prompts mit vertauschten Labels bei gleicher Norm —
    Zufallsrichtungen im R^d sind zu leicht zu schlagen. Wo möglich mit
    einem `target`-Token messen: Die KL zeigt nur, *dass* sich die Ausgabe
    bewegt; ein echter Französisch-Steer auf Llama 3.2 1B war am Ziel-Token
    `causal` (3.1×), über die KL aber nicht über der Baseline. Das Gate nutzt
    die jeweils letzte Prüfung eines Eingriffs; ihre Messgröße steht in der
    Provenienz;
  - jede gescheiterte Prüfung lässt sich mit einem `override_reason`
    übersteuern, der in die Provenienz geschrieben wird.
- **`rune_gate`** fährt ein `target`-Set (was besser werden soll) und ein
  `control`-Set (was nicht schlechter werden darf) auf der Basis und auf
  dem Modell mit Rune. Grün genau dann, wenn:
  - die Target-Accuracy nicht fällt und Accuracy oder mittlere Logprob
    steigt;
  - die Control-Accuracy höchstens um `max_control_accuracy_drop`
    (Default 0) und die mittlere Logprob höchstens um
    `max_control_logprob_drop` (Default 0.05) fällt;
  - keine neuen nicht-endlichen Items auftauchen;
  - bei Freitext-Checks höchstens `max_check_regressions` (Default 0)
    Checks pro Set mit der Rune scheitern, die auf der Basis bestehen.

  Rot verschiebt die Rune nach `rejected/`, außer `auto_reject` ist false.
- **`rune_commit`** nimmt nur eine grüne Rune an, deren Datei byte-gleich
  zu dem ist, was das Gate ausgewertet hat, und gibt die Config-Zeile
  zurück.
- Jeder Schritt wird an `audit.jsonl` im Rune-Verzeichnis angehängt.
  Abgelehnte Runes bleiben auf der Platte.

Ein grünes Gate ist notwendig, nicht hinreichend. Vor der Aktivierung die
Rune auf den echten Prompts des Einsatzes prüfen (System-Prompts, Tools,
RAG-Kontext) — Kolibris `de_heads_top64_x2` hatte ein grünes Gate und hat
englische RAG-Nutzer unter dem Produktiv-Identity-Prompt trotzdem auf
Deutsch umgestellt.

---

## 5. Goldsets

`eval`, `eval_preview`, `rune_gate` und `steer_if_preview` teilen ein
Item-Format (inline `items` oder eine `.json`-/`.jsonl`-Datei per `path`):

| Item | Korrekt genau dann, wenn |
|---|---|
| `{prompt, expect}` | jedes erwartete Token der Argmax ist (teacher-forced) |
| `{prompt, choices[], answer}` | die Antwort die höchste Gesamt-Logprob hat |
| `{prompt, generate:{max_tokens, lang, max_other_markers, match[], no_match[]}}` | die greedy, KV-gecachte Generierung jeden Check besteht: Antwortsprache (`de` / `en`), kein Code-Mixing, Fakten-Regexe |

Die Sprach-Checks zählen Funktionswörter und brauchen mindestens drei
Marker; sehr kurze Antworten bekommen kein Sprachurteil, deshalb die
Antwort-Diffs zusätzlich lesen.

Prompts dürfen die Spezial-Tokens des Modells enthalten (`<|im_start|>` …);
sie werden als Spezial-Tokens geparst, sodass Chat-formatierte Items den
Forward genau wie im Serving erreichen.

---

## 6. Tool-Referenz

### Modell und Gewichte

| Tool | Zweck |
|---|---|
| `open_model` | GGUF-Datei öffnen (Gewichte + Forward) oder NVFP4-Checkpoint (nur Forward, CUDA) |
| `model_info` | Architektur, Basis-Fingerprint, Tensor-Anzahl, Bytes je Typ |
| `list_tensors` | Tensoren mit Typ, Form, Größe und ob eine Rune sie patchen kann |
| `tensor_stats` | Mittelwert, Std, Min/Max, L2, NaN/Inf, Zeilennormen (Host-Dequant) |
| `tensor_hash` | fnv1a64-Inhaltshash — den Hash muss eine Rune-Op tragen |

### Forward und Inspektion

| Tool | Zweck |
|---|---|
| `tokenize` | Ids und Pieces mit dem Tokenizer des Modells |
| `next_token` | Top-k-Verteilung des nächsten Tokens und Entropie |
| `perplexity` | Teacher-forced Logprob je Token, NLL, Perplexity |
| `logit_lens` | Residual nach jedem Block durch Final-Norm + lm_head |
| `activations` | Residual-Statistik je Block (Norm, Update, Kosinus, Top-Dims) |
| `attention_pattern` | Attention-Gewichte eines Blocks je Head, aus dem Post-RoPE-Q/K der Engine rekonstruiert |
| `expert_usage` | Tatsächlich genommenes MoE-Routing (Stage-Tap an der Router-Auswahl) |
| `causal_trace` | Activation Patching clean → corrupt je (Block, Position) |
| `causality_check` | Bewirkt ein gefundener Eingriff (Steer-Vektor bei α oder Head-Edits) mehr als ein zufälliger gleicher Größe? Dosis-Sweep + Verdikt causal / weak / epiphenomenal / undetermined + gemessene Dosis (§4) |
| `edit_search` | Mehrziel-Suche nach Eingriffen (eigener TPE, kein Optuna): Head-Anzahl k / Scale / Block-Fenster oder Steer-α; Ziel-Goldset hoch, Kontroll-KL runter, keine Regression in freier Generierung; liefert die Pareto-Front und jeden Versuch in Kurzform. `scales` (z. B. `[2, 4]`) beschränkt die Scale auf eine diskrete Menge — auf FP8- / Q8_0-blockskalierten Gewichten Zweierpotenzen nehmen, damit die gestagte Rune exakt bleibt. Head-Versuche laufen als temporäre Runen über den Ladepfad (KV-gecacht, identisch zu einer gestagten Rune). Einen Front-Punkt mit `rune_stage_heads` / `rune_stage_steer` + `search` + `trial` stagen (Parameter und Seed in der Provenienz) |
| `eval` | Goldset-Eval, optional gegen Runes (Deltas) |
| `generate` | Greedy-Fortsetzung mit optionalem Steer / Head-Edits (langsam, zustandslos) |

### Eingriffe (nur Preview, nichts wird geschrieben)

| Tool | Zweck |
|---|---|
| `steer_vector` | Mittlere Residual-Differenz positiv − negativ an einem Block |
| `steer_list` | Steering-Vektoren dieser Session |
| `steer_preview` | Verschiebung des nächsten Tokens (und Text) mit α · Vektor |
| `ablate_preview` | Attention-Heads nullen oder skalieren; Scan je Head |
| `eval_preview` | Goldset-Eval mit Steer und/oder Head-Edits auf jedem Forward |
| `condition_vector` | Das Gate eines bedingten Steers fitten (siehe §7) |
| `steer_if_preview` | Projektions-/bedingter Steer über das echte Overlay (siehe §7) |

### Runes

| Tool | Zweck |
|---|---|
| `rune_stage_steer` | Steering-Vektor → `steer`-Rune |
| `rune_stage_steer_if` | → `steer_if`- (mit Bedingung) oder `steer_project`-Rune |
| `rune_stage_ablate` | Head-Ablation → 0/1-Spalten-`scale` auf `attn_output` |
| `rune_stage_heads` | Head-Skalierung → Spalten-`scale` (exakt auf Q8_0-/FP8-Block-Scales für 0 oder Zweierpotenzen) |
| `rune_stage_experts` | Experten abschalten → `mask` auf dem MoE-Router |
| `rune_gate` | Target-/Control-Gate, Basis vs. Rune (§4) |
| `rune_commit` | Grün + byte-gleich → `committed/`, liefert die Config-Zeile |
| `rune_rollback` | committed → staged → rejected |
| `rune_list` | Alle Runes mit Status, Typ und Gate-Zusammenfassung |
| `rune_inspect` | Rune-Manifest dekodieren |
| `rune_check` | Runes trocken gegen das geladene Modell prüfen, mit Futharks eigenen Checks |
| `rune_compare` | Gleiche Eingabe auf Basis und Modell mit Rune: KL, Top-1-Übereinstimmung, Perplexity |

### Trainingskern (7.4, in Arbeit)

| Tool | Zweck |
|---|---|
| `train_block_check` | Baut einen dichten Block (llama / qwen2) aus Hugins eigenen Train-Ops mit den entpackten Gewichten der Engine; vergleicht seinen Forward mit dem Residual der Engine nach diesem Block und prüft den Backward (Block-Input + LoRA auf q/k/v/o/gate/up/down) mit Richtungs-Differenzen über eine Leiter von Schrittweiten |
| `train_lora_sft` | LoRA-SFT des ganzen dichten Modells (llama / qwen2): Items `{prompt, response}` oder `{prompt, teacher_prompt}` (die Antwort ist die eigene Greedy-Antwort des Basismodells auf den Teacher-Prompt — Selbst-Destillation), CE nur auf Antwort-Tokens, optionaler KL-Anker zur Basis (`kl_lambda`), AdamW mit Gradient-Akkumulation und Global-Norm-Clipping; legt die Adapter als `lowrank`-Rune ab und vergleicht die Logits des Trainers mit der Engine, die diese Rune ausführt |

Die Train-Ops (`src/hugin/train/`) sind eine CPU-Referenz mit Forward und Backward für Linear + LoRA, RMSNorm, RoPE, GQA-Attention, SwiGLU, das Sigmoid-MoE-Combine, Cross-Entropy, eine Top-K-KL-Divergenz und AdamW; der Block-Backward rechnet den Forward aus dem Block-Input neu (nur der Input wird gehalten). Die Basisgewichte bekommen nie einen Gradienten. Der Modell-Forward hält nur den Input jedes Blocks, baut den lm_head nur für Zeilen mit Ziel und beendet den Backward am untersten Block mit Adapter. Lineare Schichten laufen auf der Hälfte der Hardware-Threads (`threads` überschreibt das), damit ein Server auf derselben Maschine Luft behält.

**KL-Anker (7.4.5).** Reines SFT auf einer schmalen Zielmenge verschiebt auch Antworten, die bleiben sollen. Mit `kl_lambda` > 0 addiert `train_lora_sft` den Term `kl_lambda · KL(Basis ‖ adaptiert)` auf den Antwortzeilen der Anker-Items (`kl_kinds`, Standard: jede Art außer `target`). Die Basisverteilung wird einmal vor dem Training gecacht, solange alle Adapter noch B = 0 haben: als ihre Top-`kl_topk`-Tokens (Standard 64) plus ein Tail-Eimer für die restliche Masse. Die KL ist über diese Partition exakt, ihr Gradient ist `q − p̃`, wobei p̃ die Tail-Masse so verteilt wie das Modell. Anker-Items trainieren dann nur auf der KL, mit `kl_ce` auf KL + CE. **Kontext-Destillation:** Mit `distill_lambda` > 0 lernen Items aus `distill_kinds` (Standard `target`), deren `teacher_prompt` sich vom Prompt unterscheidet, zusätzlich die volle Top-K-Verteilung der Basis *unter dem Teacher-Prompt* auf denselben Antwort-Tokens, also `KL(Teacher ‖ Schüler)`. Ihre CE bleibt, außer `distill_ce` ist false. So bekommt der Schüler pro Token die ganze Verteilung des Teachers statt nur seiner Greedy-Antwort.

Ein trainierter Adapter durchläuft denselben Lebenszyklus wie jeder andere Eingriff: staged → `rune_gate` → `rune_commit`, aktiv erst durch Eintrag in die `runes`-Config eines Modells.

---

## 7. Bedingtes und projizierendes Steering (7.11)

Ein normales `steer` addiert denselben Vektor auf jede Zeile. Das passt zu
einem unbedingten Ziel („immer etwas förmlicher“), nicht zu einem bedingten
(„auf Deutsch antworten, *wenn* der System-Prompt es verlangt“) — die
konstante Verschiebung bewegt auch die Fälle, die bleiben sollen, wie sie
sind.

`steer_if` fügt ein Gate hinzu. An einem Bedingungsblock Lc ≤ L bewertet
die Engine jede Zeile:

```text
s_r = cos(h_r an Lc, c)            c = Einheitsvektor der Bedingung
x_r += α · [s_r > θ] · v           (add)   oder
x_r −= α · [s_r > θ] · (x_r·u) u   (project), an Block L
```

Designentscheidungen:

- **Pro Zeile, nicht pro Sequenz.** CAST (IBM, ICLR 2025) entscheidet
  einmal pro Sequenz beim Prefill. Ein Gate pro Zeile braucht keine
  Zuordnung Zeile → Sequenz, läuft komplett auf dem Device (kein
  Host-Sync) und funktioniert unverändert auf jedem Forward-Pfad: Prefill,
  Decode, Continuous Batching mit ragged Rows und CUDA-Graphs. Folge: Die
  Bedingung wird auf den Zeilen gefittet, die die Antwort erzeugen, nicht
  nur am Ende des Prompts.
- **Neue Op-Typen statt neuer Felder.** Ein älteres Futhark würde ein
  unbekanntes Feld ignorieren und den Steer bedingungslos anwenden; einen
  unbekannten Op-Typ lehnt es ab.
- **Device-Ops** hinter `IRuneSteerOps` (`rowDot`, gegatetes Zeilen-Add)
  auf CPU und CUDA. Level Zero und HIP liefern keine Implementierung, dort
  scheitert eine solche Rune beim Laden.
- Das α der Projektion ist auf (0, 1] begrenzt: α > 1 würde die Komponente
  spiegeln und das gegenteilige Verhalten einbauen.

Ablauf:

1. `steer_vector` — die Richtung, die addiert (oder herausprojiziert) wird.
2. `condition_vector` — eine binäre lineare Probe (der Softmax-Trainer aus
   8.23) auf L2-normalisierten Residuals an Block Lc, positive gegen
   negative Prompts. `positions: "generated"` (Default) nimmt die Zeilen,
   die die ersten `max_tokens` Antwort-Tokens des Basismodells erzeugen.
   Liefert Richtung, Schwelle, Train-/Val-Accuracy, Score-Bereiche je
   Klasse und Feuerraten je Prompt. `calibrate_negatives: true` bewertet
   jede Zeile, die das Gate sehen wird, auf den Negativ-Prompts (alle
   Prompt-Zeilen plus generierte Antwortzeilen) und hebt die Schwelle über
   deren Maximum — nötig, wenn die Richtung auf `last`-Zeilen gefittet
   wird, denn das Gate läuft auf jeder Zeile.
3. `steer_if_preview` — installiert das Overlay genau wie die Rune
   (aufgelöst über den echten Rune-Applier) und zeigt Basis- gegen
   editierten Text plus Feuerraten über Prompt- und Antwortzeilen.
   Optionales Goldset.
4. `rune_stage_steer_if` → `rune_gate` → `rune_commit`.

Validierung auf Llama 3.2 1B (CPU): Eine „Kochen“-Bedingung an Block 4
trennte ungesehene Prompts (Val-Accuracy 0.93). Ein darauf gegateter
Französisch-Steer stellte Koch-Antworten auf Französisch um, alle anderen
Antworten blieben byte-gleich zur Basis. Die gestagten Runes
reproduzierten die Preview-Texte exakt (4/4 und 2/2).

---

## 8. Grenzen

- NVFP4-Checkpoints: nur Forward-Tools (keine Host-Inspektion der
  Gewichte), CUDA-Build nötig.
- `generate` ist zustandslos und langsam; `steer_if_preview` und
  Generate-Items in `eval` nutzen den KV-gecachten Serving-Decode.
- Auf GB10 immer nur ein GPU-Prozess; Hugin-Sessions auf einem großen
  Modell brauchen ein Wartungsfenster.
- Die Prompts, aus denen Steer- und Bedingungsvektoren gefittet werden,
  disjunkt zu den Gate-Goldsets halten, und in beiden Klassen einer
  Bedingung dieselben Fragen nutzen (nur die geprüfte Eigenschaft darf
  sich unterscheiden). Auf Kolibri-1 sah eine Bedingung „deutsche
  Instruktion vorhanden“ perfekt trennbar aus (Val 1.00), bis die
  Fitting-Fragen disjunkt waren; dann fiel sie auf Zufallsniveau (0.57).
- Steering und Head-Edits sind günstige Hebel. Fehler, die neues Wissen
  oder eine andere Policy brauchen, gehören ins Adapter-Training
  (Roadmap 7.4/7.5), nicht in eine Rune dieser Art.
