# Hugin — Input-Compression-Adapter für MimirMind auf Xe-LPG

*System-Integrations-Design für einen Level-Zero-/USM-Inference-Runtime*

**Autor:** Stefan Werfling
**Fassung:** 2026-07-10
**Status:** Engineering-Design (M-Hugin, ungeplant)
**Framing:** Das ist ein Engineering-Design-Dokument, kein Paper. Die
Compressor-Architektur, das Training-Objective und das Per-Chunk-
Kompressions-Muster werden aus publizierter Open-Source-Arbeit
portiert — COCOM (RAG-seitiger Blueprint), PISCO (Per-Chunk-Muster),
ICAE (Distillation-Rezept), LLMLingua (optionaler Vorfilter). Der
Beitrag dieser Arbeit ist die Integration dieser bekannten Muster auf
unseren spezifischen Hardware-Stack (Xe-LPG-iGPU, Level-Zero, USM,
GGUF, Munin, pegenaut-Chroma) und die Messung des Compound-Speed-
Stacks auf dieser Basis. Kein publiziertes System deckt diese
Integration ab.
**Companion-Dokumente:**
[`munin-persistent-model-daemon`](../../../synaipse-vault/Memory/mimirmind/research/munin-persistent-model-daemon.md) (Persistenz-Layer),
[`roadmap-speculative-decoding`](../../../synaipse-vault/Memory/mimirmind/research/roadmap-speculative-decoding.md) (Decode-seitiges Pendant),
[`mtp-diffusiongemma-status-2026-07`](../../../synaipse-vault/Memory/mimirmind/research/mtp-diffusiongemma-status-2026-07.md)
**Englische Fassung:** [`hugin.md`](hugin.md)

---

## Überblick

Retrieval-Augmented Generation (RAG) auf integrierten GPU-Systemen
ist prefill-gebunden, nicht compute-gebunden. Auf der Meteor-Lake-
Xe-LPG-Zielhardware von mimirmind verbringt ein 20 000-Token-RAG-
Kontext den Großteil seiner Wall-Clock-Zeit damit, Attention-Keys,
Values und dequantisierte Gewichte über einen gemeinsam mit der CPU
genutzten LPDDR5X-Bus zu streamen. Dieses Dokument spezifiziert
**Hugin** — einen Input-Compression-Adapter, der zwischen der
Embedding-Schicht und dem ersten Transformer-Block eines *frozen*
autoregressiven Base-Modells sitzt und eine lange Token-Sequenz auf
eine feste, kleine Menge von *Memory Tokens* — typisch 64 bis 256 —
komprimiert, die das Base-Modell wie gewöhnliche Input-Embeddings
konsumiert. Weil das Base-Modell nicht verändert wird, ist Hugin
eine reine Inference-Time-Optimierung: kein Retraining des Target-
Modells, kein Wechsel des GGUF-Formats, kein Eingriff in den
Sampling-Loop, keine Änderung am KV-Cache-Layout. Die algorithmische
Idee selbst ist nicht neu — eine breit publizierte 2023-2026-
Literatur deckt Input-seitige Kontext-Kompression bereits ab (COCOM,
PISCO, 500xCompressor, xRAG, ICAE, LLMLingua sind alle Open Source).
Der Beitrag dieses Dokuments liegt nicht in einer neuen Compressor-
Architektur, sondern im **System-Integrations-Pfad** für einen
solchen Compressor auf einem Level-Zero-/USM-Runtime, dessen
dominante Kosten Speicherdurchsatz sind — nicht Fließkomma-
Durchsatz — und wo eine Reduktion der effektiven Prefill-Länge um
zwei Größenordnungen sowohl den quadratischen Attention-Term als
auch den KV-Cache-Footprint pro Schicht kollabiert. Kein publiziertes
System implementiert Input-seitige Kompression auf diesem Hardware-
Stack (Xe-LPG + Level-Zero + GGUF-Adapter + persistenter USM-Daemon),
und genau diese Lücke ist das Ziel von Hugin.

Das Dokument behandelt Hugin als Element eines **Compound-Speed-
Stacks**: Input-seitige Kompression (Hugin, per-Chunk nach PISCO,
optional mit LLMLingua-Vorfilter), Output-seitige Beschleunigung
(Gemma-4-MTP-Drafter), KV-Bandbreiten-Reduktion (M10.2 F16-KV-
Cache) und Prefix-Caching auf dem komprimierten Memory-Prefix
(RadixAttention). Das Stapeln der Hebel ist der Punkt: der isolierte
Hugin-Hebel liefert bei typischen 20-k-Token-RAG-Kontexten etwa 4×,
aber der Stack liefert 15×–20× End-to-End-Wall-Clock und 28×–68×
Time-to-First-Token auf denselben Requests. Das Dokument skizziert
einen Aufbauplan in drei Meilensteinen (M-Hugin.1 bis M-Hugin.3,
Aufwand ~5–6 Wochen) inkl. Distillations-Trainings-Regime (nach
COCOM/ICAE), GGUF-kompatiblem Adapter-Format und
Integrations-Schema mit dem bestehenden Munin-Persistenz-Layer.

Der Name folgt dem im Projekt bereits etablierten Norse-Muster. Odins
zwei Raben sind **Huginn**, *Gedanke*, der ausfliegt um Informationen
aus der Welt zusammenzutragen, und **Muninn**, *Erinnerung*, die
festhält was gesammelt wurde. Munin ist unser persistenter
Model-Memory-Daemon, der die Gewichte über Restarts hinweg im RAM
hält. Hugin, komplementär, ist die Schicht, die *zum Dokument
ausfliegt*, Gedanken daraus extrahiert und mit einer kompakten
Repräsentation zurückkehrt, auf die der weise Ratgeber (Mimir)
dann antworten kann.

---

## 1. Einleitung

### 1.1 Der Prefill-Flaschenhals bei RAG

Der Produktions-Ziel-Workload von mimirmind ist Retrieval-Augmented
Generation, ausgeliefert an das Schwesterprojekt *pegenaut*. In einem
typischen RAG-Call gehen dem User-Turn 3 bis 15 abgerufene
Dokument-Chunks im Umfang von 3 000 bis 20 000 Tokens voraus. Das
autoregressive Base-Modell — Gemma 4 26B-A4B-it in Produktion, Gemma 4
E4B im Perf-Shelf — muss jedes einzelne dieser Tokens *einmal*
verarbeiten und einen KV-Cache-Tensor mit Key- und Value-Projektionen
pro Schicht füllen, bevor auch nur ein einziges Antwort-Token
gesampelt werden kann.

Auf unserer Zielhardware wird der Aufwand dieser Ingest-Phase
dominiert von:

- **KV-Cache-Writes.** Bei 20 000-Token-Kontext auf Gemma 4 26B-A4B
  mit 40 Schichten, 4 KV-Heads und Head-Dim 256 in F16 ergeben sich
  `20 000 · 40 · 2 · 4 · 256 · 2 B ≈ 3.2 GiB` KV-Traffic.
- **Attention-Key/Value-Reads.** Der Attention-Pass streamt den
  kompletten Key- und Value-Block pro Layer über den geteilten Bus,
  jedes Mal wenn eine neue Query projiziert wird. Während des
  Prefills ist das nicht quadratisch in Tokens (ein großer batched
  Matmul), aber sehr wohl *linear* in der Kontextlänge.
- **Weight-Streaming.** On-the-fly-Dequantisierung von Q6_K-, Q8_0-
  und Q4_K-block-quantisierten Gewichten konkurriert mit dem oben
  genannten KV-Traffic um LPDDR5X-Bandbreite.

Das Perf-Regression-Ledger im Synaipse-Vault zeigt einen Live-1331-
Token-Prefill auf L0_TARGET_HOST bei 5.9 s Wall-Clock und 141 J
Energie, mit einem Q8_0-A/B-Win von −13.7 % Wall-Clock gegenüber F32
(Session-Note `project_session_state.md`). Linear extrapoliert
landet ein 20 000-Token-Prefill selbst nach allen Kernel-
Optimierungen dieses Quartals bei mehreren zehn Sekunden.

Jeder Kernel-seitige Hebel, den wir gebaut haben — der GEMM-Prefill-
Rewrite (ADR `gemm-prefill-rewrite`), der FlashAttention-Q8_0-GQA-
Prefill-Kernel (Commits `10e28a4`, `f33ac05`, `cde6e22`, `c18d720`),
der Command-List-Replay-Decode-Pfad (ADR `2026-07-06-command-list-
replay`), das KV-dtype-Foundation-Package (`M10.2`) — greift jeweils
einen Konstantenfaktor derselben Gleichung an. Alle helfen. Keiner
reduziert die *Anzahl der Tokens*, die das Base-Modell verarbeiten
muss.

Hugin greift genau diese Zahl an. Wenn ein 20 000-Token-Dokument
verlust-tolerabel durch 256 Memory-Tokens repräsentiert werden kann,
schrumpft die Prefill-Arbeit um Faktor 78 auf der Token-Anzahl *und*
der entsprechende KV-Traffic verschwindet mit ihr. Dieses Paper
untersucht, ob das erreichbar ist, wie viel Qualität dabei auf dem
Spiel steht und wie man es innerhalb des mimirmind-Runtime baut.

### 1.2 Positionierung des Beitrags

Der Vorschlag ist bewusst eng gefasst. Hugin ist:

- **Kein neues Base-Modell.** Die Gemma-4-Zielgewichte sind frozen.
- **Kein KV-Cache-Prefetch oder Session-Persistenz-Schema.** Das sind
  eigene Hebel, die in `M9.11 speculative decoding` und in der
  Roadmap für Session-KV-Cache über Turns hinweg behandelt werden.
- **Kein Diffusion-Decoder.** DiffusionGemma ist ein paralleler
  Forschungszweig (dokumentiert in `mtp-diffusiongemma-status`); er
  greift die Decode-Seite an, Hugin die Prefill-Seite.
- **Kein Tokenizer-Wechsel im SentencePiece-Sinn.** Der User-seitige
  Token-Stream bleibt unverändert; Hugin operiert eine Ebene tiefer,
  auf Token-*Embeddings*.

Hugin ist ein **gelernter Input-Compressor**, strukturell am nächsten
verwandt mit dem Q-Former in BLIP-2, dem latenten Bottleneck in
Perceiver IO, den "Gist-Tokens" von Mu et al. (2024) und den
In-Context-Autoencoders (Ge et al., 2024). Die Neuigkeit für dieses
Projekt liegt nicht in der mathematischen Form — die ist in der
Vision-Language- und Long-Context-Literatur gut erforscht — sondern
in der *System-Integration* auf einem Level-Zero-/USM-iGPU-Stack,
dessen Bottleneck-Profil sich qualitativ vom Discrete-GPU-CUDA-
Kontext dieser Vorarbeiten unterscheidet.

### 1.3 Scope — was wir bauen, was wir übernehmen

**Wir erfinden keine neue Kompressions-Methode.** Compressor,
Trainings-Objective und Per-Chunk-Muster werden 1:1 aus publizierten
Open-Source-Arbeiten übernommen (Abschnitt 3). Was hier spezifiziert
wird, ist Engineering-Integration, keine algorithmische Neuheit.

**Was wir bauen:**

- **Runtime-Integration.** Eine `HuginAdapter`-Klasse in
  `InferenceEngine`, teilt sich `UsmAllocator` und `L0Context` mit
  dem Base-Modell; SPIR-V-Kernels für Cross-Attention und
  Projection werden wie die existierenden Q4_K/Q6_K/Q8_0-Kernels
  gebaut (ocloc zur Build-Zeit).
- **GGUF-Adapter-Format** — kleine Manifest-Erweiterung, damit der
  existierende Loader (M3) Hugin-Adapter neben dem Base-Modell
  konsumieren kann, inkl. strikter Tokenizer-Hash- + Residual-
  Dimension-Kompatibilitätsprüfung zur Load-Zeit.
- **`config.json`-Integration** — ein einziger `hugin`-Block für
  Opt-in, Adapter-Pfad, Memory-Token-Anzahl und Per-Chunk-Caching.
- **Munin-Kopplung** — Adapter-Weights sind IPC-shared wie die
  Base-Model-Weights, überleben mimirmind-Worker-Restarts.
- **Pegenaut-Chroma-Integration** — neuer
  `POST /v1/hugin/encode`-Endpoint plus Chroma-seitige Chunk-
  Ingest-Pipeline, die Per-Chunk-Memory-Tokens mit Adapter-Version-
  Tag speichert.
- **Compound-Speed-Stack-Orchestrierung** — Code-Pfade, damit Hugin
  sauber mit den bereits geshippten Speed-Hebeln komponiert
  (MTP-Drafter, KV-F16, FlashAttention Q8_0-GQA, RadixAttention-
  artiger Prefix-Cache).
- **Distillations-Trainings-Pipeline**, läuft auf einem Discrete-
  GPU-Host (nicht auf Ziel-HW), produziert GGUF-Adapter für den
  mimirmind-Runtime.
- **Benchmark-Harness + Perf-Ledger-Einträge**, die jeden Schritt
  des Stacks gegen die unkomprimierte Baseline auf
  L0_TARGET_HOST validieren.

**Was wir übernehmen (nicht neu ableiten):**

- Encoder-Architektur: **COCOM-artig**, Cross-Attention-Read-out
  mit lernbaren Latent-Queries; Sizing folgt COCOM's kleiner
  Konfiguration.
- Per-Chunk-cache-freundliches Kompressions-Muster: **PISCO** —
  eine feste Memory-Token-Menge pro RAG-Chunk, zur Query-Zeit
  kombiniert.
- Training-Objective: **ICAE**-artige Self-Distillation mit dem
  frozen Base als eigener Teacher; keine Gold-Antworten nötig.
- Optionaler Vorfilter für extreme Kontexte: **LLMLingua**-
  Token-Pruning als Vorstufe vor dem Hugin-Encode.
- Positional-Encoding, Attention-Masking und Layer-1-Injection
  folgen direkt dem BLIP-2- / Compressed-Context-Memory-Muster.

**Aufwand-Implikation.** Weil die algorithmischen Komponenten
portiert und nicht designed werden, ist die Trainings-Pipeline
eine Distillation-Loop gegen den frozen Base — kein Forschungs-
Programm. Geschätzter Gesamtaufwand: **5–6 Wochen** inkl.
Ziel-HW-Benchmark, nicht die 9 Wochen, die ein From-Scratch-
Adapter-Design brauchen würde.

---

## 2. Hintergrund

### 2.1 Xe-LPG-Speicherhierarchie auf Meteor Lake

Die Zielplattform, dokumentiert in der Synaipse-Note
`Meteor Lake — Dual-IMC Memory Layout (L0_TARGET_HOST)`, ist eine
integrierte Xe-LPG-GPU, die sich LPDDR5X-7467 mit der CPU teilt. Die
zwei unabhängigen 64-Bit-IMCs liefern rund 89 GB/s aggregierte
Bandbreite an einen einzelnen Client, der sie saturieren kann, aber
dieses Budget teilt sich mit dem Co-Tenant-Pegenaut-TypeScript-Stack
auf demselben Host. In der Praxis liegt die für Inferenz nutzbare
sustained Bandbreite eher bei 55–70 GB/s; der Xe-LPG-Dispatch-Overhead
pro Level-Zero-Command wurde empirisch auf 10–15 μs gemessen (Lesson
`xelpg_dispatch_overhead`), nicht auf die ursprünglich angenommenen
40 μs.

Die Konsequenz für den Prefill: die arithmetische Intensität der
Transformer-Layer, in Operationen pro Byte, ist nicht die limitierende
Ressource. Q8_0-quantisierte Gewichte streamen bereits nahe an der
Peak-Sustained-Bandbreite. Jede Workload, die proportional zur
Token-Anzahl ist, skaliert linear mit der Wall-Clock; die einzige
Möglichkeit, diese Kurve zu verschieben, ist die Token-Anzahl zu
reduzieren.

### 2.2 Aktueller Prefill-Aufwand, gemessen

Zahlen aus dem Perf-Ledger und den Session-Notes auf dem Prod-Host:

| Modell                    | Ctx-Tokens | Prefill-Wall | Prefill / Tok | Energie |
|---------------------------|-----------:|-------------:|--------------:|--------:|
| Gemma 4 E4B Q4_K_M        |        306 |     ~0.5 s   |     ~1.6 ms   |   ~10 J |
| Gemma 4 26B-A4B Q8_0      |       1331 |      5.9 s   |     ~4.4 ms   |   141 J |
| Gemma 4 26B-A4B Q6_K      |       3400 |     ~14 s    |     ~4.1 ms   |  ~330 J |
| (extrapoliert, dasselbe)  |     20 000 |     ~80 s    |      ~4 ms    | ~1900 J |

Die E4B-Zeile ist Bench-Modus (rp0, CLR-ON) auf warmem System. Die
26B-A4B-Zeilen enthalten den Governor-Aufschlag durch den geteilten
Tenant. Die Prefill-Kosten pro Token sind über den mittleren
Kontextlängen-Bereich bemerkenswert stabil — der Fingerabdruck eines
bandbreitengebundenen Kernels: Bytes pro Token sind eine Konstante,
Wall-Clock skaliert mit Token-Anzahl.

Für einen pegenaut-RAG-Turn mit fünf 4 000-Token-Chunks dominieren
die extrapolierten 80 Sekunden Prefill die zehn Millisekunden pro
Decode-Token, die danach folgen. Der User erlebt den gesamten Request
als "langsam", auch wenn jedes einzelne Token nach dem ersten in
Echtzeit ausgeliefert wird. Genau das greift Hugin an.

### 2.3 Was bereits versucht wurde

- **Q8_0-Aktivierungs-Quantisierung** — live, −13.7 % Wall auf Prefill.
- **FlashAttention-Q8_0-GQA-Prefill-Kernel** — auf Head-Branch,
  K-Tile-Autotune-Reconstruction gerade gelandet (Commit `c18d720`).
- **GEMM statt matvec für Prefill** — als `perf/gemm-prefill` gelandet,
  ersetzt den matvec-Loop wenn M > 1 auf Q4_K/Q6_K-Pfaden.
- **Command-List-Replay für Decode** — als `M-CLR` gelandet, nur
  Decode-seitig (strukturell inkompatibel mit MoE-Prefill, siehe
  `lesson_moe_clr_incompatible`).
- **KV-dtype-F16-Foundation** — als `M10.2 Phase 1a` gelandet,
  halbiert den KV-Bandbreiten-Term.
- **Speculative Decoding (M9.11) und MTP-Drafter** — designed und
  vermessen; die Analyse in `roadmap-speculative-decoding` zeigt,
  dass diese Hebel beim Decode helfen, nicht beim Prefill.

Alle diese sind Kernel-Konstantenfaktor-Verbesserungen. Keiner
reduziert die Anzahl der Tokens, die das Base-Modell verarbeitet.
Hugin ist der erste Vorschlag auf dem Shelf, der die Token-Anzahl
selbst verändert.

---

## 3. Verwandte Arbeiten

Die Idee, einen langen Kontext in eine kleine Menge dichter Vektoren
zu komprimieren, die ein frozen Language-Modell konsumieren kann, ist
nicht neu; die Literatur seit 2023 hat mindestens vier deutliche
Familien hervorgebracht, an deren Kreuzung Hugin sitzt.

**Perceiver und Perceiver IO (Jaegle et al., 2021, 2022).** Ein
gelerntes Cross-Attention-Read-out von einem großen Input in ein
kleines latentes Array. Perceiver hat gezeigt, dass ein Latent-Array
fester Größe genug Information für nachgeordnete Aufgaben tragen kann,
solange das Read-out bidirektional und mehrschichtig ist. Perceiver IO
erweitert das Muster um ein symmetrisches Write-out — beliebige
Ausgabe-Strukturen aus dem latenten Zustand generierbar.

**BLIP-2 Q-Former (Li et al., 2023).** Ein trainierbares 32-Query-
Modul, das Patch-Tokens eines frozen ViT in 32 dichte Vektoren
verwandelt und sie über eine lineare Projektion in die Input-Seite
eines frozen LLM speist. Strukturell die publikationsseitig
naheliegendste Analogie zu Hugin: eine kleine, trainierbare Brücke,
die einen großen Frozen-Encoder-Output in eine kurze Sequenz umformt,
die vom Frozen-Decoder-LLM wie normale Input-Embeddings konsumiert wird.

**Prefix-Tuning und Prompt-Tuning (Li & Liang, 2021; Lester et al.,
2021).** Hat gezeigt, dass eine kleine Zahl trainierbarer "Virtual
Tokens" vor dem Input ein Frozen-LLM steuern kann. Hugin verallgemeinert
das von einem festen gelernten Prefix zu einem input-abhängigen
gelernten Prefix: die Memory-Tokens sind eine Funktion des Dokuments,
keine Konstante.

**Gisting (Mu, Li, Goodman, 2024).** Prompt-Kompression, die spezielle
"Gist"-Tokens während einer Fine-Tuning-Phase einfügt und über
Attention-Masks das Modell zwingt, alle Informationen des voranstehenden
Prompts in die Gist-Positionen zu quetschen. Zur Inference-Zeit werden
die Pre-Gist-Tokens verworfen. Erste Arbeit mit Kompressionsverhältnis
26× bei minimalem Downstream-Verlust auf einer Reihe von
Instruction-Following-Benchmarks.

**In-Context Autoencoders / AutoCompressors (Ge et al., 2024;
Chevalier et al., 2023).** Autoencoder-artige Methoden, die einen
kleinen Encoder trainieren, aus einem langen Dokument k dichte
"Summary-Vektoren" zu erzeugen, aus denen ein Frozen-Decoder das
Dokument rekonstruieren oder Fragen dazu beantworten kann.
Kompressionsverhältnisse 4×–15× ohne Downstream-Degradation auf QA.
ICAE's Reference-Impl und AutoCompressors-Code sind beide public;
AutoCompressors ist an Llama-2 / OPT-2.7B gebunden und seit 2024
nicht mehr gepflegt.

**Compressed Context Memory (Kim et al., ICLR 2024).** Trainiert ein
Conditional-LoRA, das während interaktiver Nutzung dynamisch
komprimierten Memory der Attention-KVs erzeugt. Public PyTorch-
Referenz unter [snu-mllab/context-memory](https://github.com/snu-mllab/context-memory).
Strukturell nah an Hugin (frozen Base + kleiner trainierbarer
Adapter), unterscheidet sich aber im Kompressions-Ziel: KV-Cache
selbst statt Layer-1-Memory-Token-Prefix.

**LLMLingua / LongLLMLingua (Jiang et al., Microsoft, 2023–2026).**
Kein Soft-Embedding-Compressor, sondern ein *Token-Selection*-
Compressor, der über ein kleines Language-Modell niedrig-
informative Tokens identifiziert und wegwirft. Erreicht rund 20×
Kompression bei ~1.5 pp Accuracy-Drop und ist Stand 2026 das am
breitesten in Prod eingesetzte Prompt-Compression-Tool. Komponiert
mit Hugin (erst LLMLingua, dann Memory-Token-Kompression auf der
reduzierten Sequenz), ersetzt es aber nicht — LLMLinguas
Kompressions-Ceiling liegt etwa eine Größenordnung unter Soft-
Embedding-Methoden.

**xRAG (Cheng et al., NeurIPS 2024).** Das extreme Ende: ein ganzes
Dokument in *einen einzigen* Soft-Token kollabiert, der in den
LLM-Input eingefügt wird. Zeigt, dass moderne Embedding-Modelle für
Short-Answer-QA bereits genug Information tragen, wenn das LLM auf
Konsum dieses einen Vektors fine-tuned wird. Die Obergrenze des
Machbaren, aber verlangt LLM-Fine-Tuning (verletzt den Frozen-Base-
Vertrag).

**COCOM (Rau et al., 2024).** Context Embeddings for Efficient
Answer Generation in RAG. Meldet einen *gemessenen* End-to-End-
Inference-Speedup von **5.69×** auf RAG-QA. Aktuell die publikations-
nächste Zahl zur Wall-Clock-Schätzung in §6.4 dieses Dokuments und
der nützlichste externe Kalibrierungs-Punkt. Direkt verwandt mit
unserer Variante A.

**500xCompressor (Li & Briscoe, ACL 2025).** Trainiert einen
Soft-Prompt-Compressor mit 0.3 % zusätzlichen Parametern, erreicht
Kompressionsverhältnisse bis **480×** bei erhaltenen 62–73 % der
LLM-Downstream-Capability. Belegt publikations-seitig, dass Hugins
Zielbereich (200×–300×) prinzipiell erreichbar ist, mit
dokumentiertem Qualitäts-Preis, der zwar nicht trivial, aber nicht
katastrophal ist.

**PISCO (2025).** Encoder-Decoder gemeinsam trainiert mit variabler
Memory-Token-Anzahl pro abgerufenem Chunk, spezifisch für RAG.
Antizipiert unsere Mode-2-pegenaut-Chroma-Integration (§5.6): pro
Chunk zur Ingest-Zeit berechnete Memory-Tokens, neben dem Chunk
gespeichert. PISCO's Beitrag ist der Nachweis, dass Per-Chunk-
Memory-Tokens zur Query-Zeit ohne Cross-Chunk-Retraining
kombinierbar sind — genau die Annahme, auf der Mode 2 beruht.

**Q-RAG (Sun et al., ICLR 2026, Oral).** Reinforcement-Learning-
Ansatz, trainiert einen leichten Embedder-Agenten für Multi-Step-
Retrieval im Latent-Space bei frozen LLM. Anderes Task-Framing —
Retrieval statt Kompression — aber das Frozen-LLM-Adapter-Muster
ist dasselbe. Public Code unter [griver/Q-RAG](https://github.com/griver/Q-RAG).

**Multi-Token-Prediction-Drafter und Speculative Decoding
(mimirmind-Notes).** Greift die Decode-Seite an. Vollständig
orthogonal zu Hugin — kompatibel kombinierbar.

**Prefix-Caching / RadixAttention (Zheng et al., 2024).** Wiederverwendung
des bereits berechneten KV-Zustands eines geteilten Prompt-Prefixes über
viele Requests. Storage-Optimierung, auch für unkomprimierte Prompts
anwendbar; komponiert sauber mit Hugin, weil Hugin-komprimierte Inputs
kürzer sind und damit billiger zu prefix-cachen. Hugin ersetzt
Prefix-Caching nicht, es multipliziert dessen Budget.

**T5, BART (Raffel et al., 2020; Lewis et al., 2020).** Encoder-Decoder
mit Cross-Attention. Wenn wir das Base-Modell wechseln dürften, wäre
das der Lehrbuch-Ansatz. Dürfen wir aber nicht.

Positionierung von Hugin in diesem Raum:

- Kompressionsverhältnis-Ziel: 50×–300×, aggressiver als Gisting
  (26×) oder ICAE (4×–15×), auf COCOM's Operating-Point ausgerichtet,
  gut innerhalb der Envelope, die 500xCompressor publiziert hat.
- Frozen Base: ja, wie bei BLIP-2, Prefix-Tuning, Compressed Context
  Memory und COCOM; nicht wie T5/BART, nicht wie xRAG (das Base-LLM-
  Fine-Tuning verlangt).
- Input-abhängig: ja, im Gegensatz zu Prefix-Tuning; wie Gisting,
  BLIP-2 und alle anderen Arbeiten im Survey.
- Konsum-Interface: normale Input-Embeddings, keine Cross-Attention-
  Modifikation; nicht wie Perceiver IOs separater Decoder, sondern
  wie BLIP-2, Gisting, ICAE und COCOM.
- Per-Chunk-cache-freundlich: ja, folgt PISCO's Muster.

**Ehrliches Positioning-Statement.** Der algorithmische Inhalt von
Hugin ist eine geradlinige Instanziierung des Input-seitigen Soft-
Embedding-Compressor-Musters, das 2023–2024 etabliert und 2024–2026
von COCOM, PISCO und 500xCompressor verfeinert wurde. Der Beitrag
dieses Dokuments liegt nicht in einer neuen Kompressions-Methode,
sondern in der **erstmaligen System-Integration eines solchen Musters
auf einem Level-Zero-/USM-iGPU-Inference-Stack mit GGUF-serialisiertem
Adapter, persistentem USM-Daemon (Munin) für Zero-Downtime-Adapter-
Reload und Chroma-seitiger Pre-Compression-Pipeline in Kopplung mit
dem pegenaut-Schwesterprojekt.** Keine publizierte Open-Source-
Implementierung zielt auf diese Hardware-/Runtime-Kombination, und
keine der existierenden Impls (ICAE, COCOM, PISCO, Compressed Context
Memory) ist über ihren gelieferten Code auf Xe-LPG lauffähig.

---

## 4. Methode

### 4.1 Gesamtarchitektur

Hugin fügt genau eine neue Komponente in den Forward-Graph des
Base-Modells ein:

```
Dokumente ── Tokenizer ── Embedding-Lookup ─┐
                                            ▼
                                    ┌─────────────┐
                                    │    Hugin    │   trainierbar
                                    │  Encoder E  │
                                    └──────┬──────┘
                                           │  m Memory-Tokens (m ≪ n)
                                           ▼
     User-Turn ── Tokenizer ── Embedding-Lookup ─┐
                                                 ▼
                                    [ Memory | User ]  concat auf
                                                       Sequenz-Achse
                                                 │
                                                 ▼
                                        Transformer-Layer 1
                                                 ▼
                                        Transformer-Layer 2
                                                 ▼
                                             ... (frozen) ...
                                                 ▼
                                        Transformer-Layer L
                                                 ▼
                                             LM-Head
                                                 ▼
                                             Logits
```

Formal: gegeben eine Dokument-Token-Sequenz $\mathbf{d} \in \mathcal V^n$
und ein User-Turn $\mathbf{u} \in \mathcal V^k$, mit
$W_E \in \mathbb R^{|\mathcal V| \times d}$ als frozen Embedding-Matrix
des Base-Modells:

$$
\mathbf X_d = W_E[\mathbf d] \in \mathbb R^{n \times d}, \quad
\mathbf X_u = W_E[\mathbf u] \in \mathbb R^{k \times d}
$$

Hugins Encoder $E_\theta$ mit trainierbaren Parametern $\theta$
erzeugt $m$ *Memory-Embeddings*:

$$
\mathbf M = E_\theta(\mathbf X_d) \in \mathbb R^{m \times d}
$$

wobei $m$ ein fixer Hyperparameter ist (Kandidaten: 64, 128, 256) und
$m \ll n$. Das Base-Modell $F_\phi$, frozen mit vortrainiertem
Gewicht $\phi$, wird auf der Konkatenation von Memory-Embeddings
und User-Token-Embeddings aufgerufen:

$$
\hat{\mathbf y} = F_\phi([\mathbf M; \mathbf X_u])
$$

Nur $\theta$ wird trainiert. $\phi$ und $W_E$ sind frozen.

Die Memory-Embeddings teilen die Residual-Stream-Dimension $d$ des
Base-Modells (2304 für Gemma 4 26B-A4B, 3072 für Gemma 4 31B Dense).
Genau das erlaubt es, sie ohne jede Änderung an Layer 1 dort einzuspeisen.

### 4.2 Design-Varianten und die gewählte Variante

Der ursprüngliche Chat, aus dem dieses Paper hervorging, hat drei
Varianten diskutiert; ich fasse sie hier zusammen und begründe die
Wahl.

**Variante A — Mehrere Dokument-Tokens, one-shot.** Das Dokument geht
durch einen kleinen Encoder, der $m$ Tokens ausgibt. Diese Tokens
werden dem User-Turn vorangestellt. Das ist das Design aus Abschnitt 4.1.
Einfachste Implementierung, günstigste Serve-Kosten, engste
strukturelle Übereinstimmung mit dem existierenden Input-Verhalten des
Base-Modells. **Gewählte Variante.**

**Variante B — Encoder / Cross-Attention-Decoder (T5-artig).** Das
Dokument wird zu $m$ Latents encodiert und das Base-Modell liest sie
über Cross-Attention-Layer, die zwischen bestehenden Self-Attention-
Blöcken eingeschoben werden. Architektonisch sauberer, weil er
Dokument- und Query-Stream trennt, verlangt aber die Einschiebung neuer
Attention-Layer in einen frozen Decoder — was entweder umfangreiches
Training der eingefügten Cross-Attention-Weights oder einen komplett
neuen Encoder-Decoder erfordert, was den Frozen-Base-Vertrag bricht.
Verworfen.

**Variante C — Multi-Stage-Adapter mit Re-Injection an tieferen Layers.**
Ein kleiner Encoder erzeugt $m$ Latents vor Layer 1; ein zweiter
Adapter re-injiziert eventuell andere Latents zwischen einer frühen
Layer-Gruppe (etwa Layer 1–8) und einer späten Gruppe (Layer 9–L).
Motivation: frühe Layer kodieren mehr Syntax, späte mehr Semantik;
jede Gruppe mit einer eigenen Projektion des Dokuments zu konditionieren
könnte mehr Input-Information erhalten als eine einstufige Kompression.
Variante C bleibt query-agnostisch und damit Chroma-cache-freundlich.

**Variante D — Query-conditioned Single-Stage-Kompression.** Dokument
*und* User-Frage werden gemeinsam in den Hugin-Encoder gespeist; die
Latent-Queries cross-attendieren in `[Doc-Embeddings ; Query-Embeddings]`
und produzieren Memory-Tokens, die schon reflektieren was die Frage
verlangt. Erwarteter Quality-Gewinn gegenüber Variante A bei
Extract-Verbatim-, Numeric-Lookup- und Multi-Hop-Reasoning-Fragen;
Quality-Parity ansonsten. Kosten: die Memory-Tokens werden zur
Funktion der Query — Chunk-Caching in Chroma (Mode 2, §5.6) gilt für
jede Query, die Variante D triggert, **nicht mehr**, und jeder Request
zahlt den vollen ~1.1 s Encoder-Schritt. Verwandte publizierte Arbeit:
LongLLMLingua Query-Aware-Modus, PISCO's variabler Per-Chunk-Modus,
[Query-Conditioned Selector for RAG Compression](https://arxiv.org/pdf/2602.15856).

**Varianten A, C und D koexistieren zur Runtime.** Die drei Varianten
attackieren unterschiedliche Punkte auf der Fläche
Qualität × Speed × Cacheability. Statt statisch einen zu wählen, lädt
der Runtime alle drei Adapter (soweit verfügbar) und wählt per
Request via **Autotune** (§4.7). Variante B ist unter dem
Frozen-Base-Vertrag nicht implementierbar und wird nicht gebaut.

**Trade-off-Zusammenfassung:**

| Variante | Query-aware | Chroma-cacheable | Encode/Req | Best bei             |
|----------|:-----------:|:----------------:|:----------:|----------------------|
| A        |    nein     |        ja        |   ~1.1 s   | General-Purpose      |
| C        |    nein     |        ja        |   ~1.5 s   | lange Docs, komplex  |
| D        |   **ja**    |    **nein**      |   ~1.1 s   | Extract/Numeric      |

### 4.3 Der Hugin-Encoder

Der Encoder $E_\theta$ ist ein kleiner Transformer-Encoder-artiger
Block mit Cross-Attention-Read-out in ein Latent-Array fester Größe,
enger am Perceiver-Ur-Design als an BLIP-2:

1. **Lernbare Latent-Queries.** $m$ Latent-Vektoren $\mathbf Q \in
   \mathbb R^{m \times d_h}$, zufällig initialisiert, sind der einzige
   sequenzlängen-unabhängige Zustand im Encoder.
2. **Cross-Attention-Read-out.** Ein Stack aus $L_E$ Blöcken wechselt
   zwischen (a) Cross-Attention von $\mathbf Q$ in $\mathbf X_d$ in
   Standard-Scaled-Dot-Product-Form und (b) Self-Attention innerhalb
   $\mathbf Q$.
3. **Projektion.** Ein finaler linearer Layer projiziert von der
   Encoder-Innen-Dimension $d_h$ hoch (oder runter) auf die Residual-
   Dimension $d$ des Base-Modells, was $\mathbf M$ ergibt.

Sizing-Kandidaten für den ersten Prototypen:

| Parameter          | Klein     | Mittel    | Groß       |
|--------------------|----------:|----------:|-----------:|
| $L_E$ (Blöcke)     | 4         | 6         | 8          |
| $d_h$              | 512       | 768       | 1024       |
| Heads              | 8         | 12        | 16         |
| $m$                | 128       | 256       | 256        |
| Parameter-Zahl     | ~15 M     | ~55 M     | ~140 M     |

Die "Klein"-Konfiguration ist klein genug, um den Encoder-Schritt auf
der iGPU günstig zu halten (siehe Performance-Modell in Abschnitt 6);
die "Groß"-Konfiguration ist das, was wir vermutlich brauchen, wenn
Kompression auf $m = 64$ bei sehr langem Kontext gefordert ist.

### 4.4 Trainings-Regime

Das Base-Modell ist frozen. Nur $\theta$ wird trainiert.

**Zielfunktion.** Self-Distillation vom frozen Base-Modell, in der
Linie von AutoCompressors und ICAE:

1. Sample ein Dokument $\mathbf d$ (bis 32 k Tokens) und ein zugehöriges
   Query-Answer-Paar $(\mathbf u, \mathbf y)$ aus einem RAG-artigen
   Corpus.
2. Berechne ein *Teacher*-Ziel: laufe das frozen Base-Modell auf
   $[\mathbf d; \mathbf u]$ und zeichne die Logits an den Answer-
   Positionen auf: $\ell_{\text{teacher}}(\mathbf y_t \mid \mathbf d,
   \mathbf u, \mathbf y_{<t})$.
3. Berechne eine *Student*-Prediction: laufe Hugin auf $\mathbf d$ um
   $\mathbf M$ zu erhalten, dann das frozen Base-Modell auf
   $[\mathbf M; \mathbf u]$, und zeichne die Logits an den Answer-
   Positionen auf.
4. Minimiere die Token-Level-KL-Divergenz zwischen Teacher- und
   Student-Logits an jeder Answer-Position, plus einen kleinen
   Standard-Cross-Entropy-Term gegen die Gold-Antwort.

$$
\mathcal L(\theta) = \sum_t \Big[
   \underbrace{\mathrm{KL}\big(
      p_{\text{teacher}}(\cdot \mid \mathbf d, \mathbf u, \mathbf y_{<t})
      \Vert
      p_{\text{student}}(\cdot \mid E_\theta(\mathbf d), \mathbf u,
                                   \mathbf y_{<t})
   \big)}_{\text{Distillation}}
   + \lambda \underbrace{\big(-\log p_{\text{student}}(\mathbf y_t)\big)}_{\text{Gold}}
\Big]
$$

mit $\lambda$ typisch 0.1–0.3.

**Warum Distillation.** Teacher und Student teilen sich denselben
frozen Decoder $F_\phi$; der einzige Unterschied ist die Input-Seite.
Der KL-Term reduziert sich damit zu: *"produziere Memory-Tokens, sodass
$F_\phi$'s Next-Token-Verteilung dem entspricht, was $F_\phi$ bei
Sicht auf das vollständige Dokument ausgeben würde."* Genau das ist
das Ziel zur Inference-Zeit. Es ist unabhängig vom exakten Wortlaut
der Gold-Antwort und funktioniert selbst dann, wenn der RAG-Corpus
keine Gold-Antworten enthält — ein unkurierter Dokumenten-Dump plus
generierte Queries reicht.

**Curriculum.** Start mit kurzen Dokumenten (256–1024 Tokens),
progressive Steigerung bis 32 k. Rationale entsprechend Abschnitt 3
des Gisting-Papers: der Compression-Head lernt auf kurzen Inputs
deutlich leichter, und die gelernten Latent-Query-Muster transferieren.

**Daten.** Für einen Erst-Prototypen genügt FineWeb-Edu plus
synthetische Query-Generierung durch das Base-Modell selbst, um Proof-
of-Concept auf Gemma 4 E4B zu erreichen. Für einen Gemma-4-26B-A4B-
Prod-Run ist der pegenaut-RAG-Corpus die verteilungsangemessene
Trainingsquelle.

**Wo Training läuft.** *Nicht auf der Zielhardware von mimirmind.*
Hugins Training ist ein GPU-Tage-Job auf einem Discrete-GPU-Host
(gemietete H100 oder eigene RTX) unter PyTorch. Nur die resultierenden
Adapter-Weights kommen zurück in den mimirmind-Runtime, konvertiert
zu GGUF und über den existierenden Weight-Loader-Pfad geladen. Damit
bleibt die Projektregel gewahrt, dass mimirmind nicht trainiert (siehe
CLAUDE.md).

### 4.5 Positional- und Vokabular-Betrachtungen

Zwei Feinheiten sind für Korrektheit innerhalb eines Decoder-only-
Base-Modells relevant.

**Positional-Encoding der Memory-Tokens.** Gemma 4 nutzt RoPE mit
per-Layer-Sliding-Window-Base-Frequenzen. Die Memory-Tokens belegen
Positionen $0, 1, \dots, m-1$ im Positions-Counter des Base-Modells,
der User-Turn belegt Positionen $m, m+1, \dots, m+k-1$. Das heißt, die
*effektive* positionale Distanz zwischen dem letzten Memory-Token und
dem ersten User-Token ist 1, nicht $n$. Genau das ist die gewünschte
Eigenschaft: das Modell soll Memory-Tokens wie einen unmittelbaren,
angrenzenden Kontext behandeln, nicht wie eine entfernte Präambel.
Es gibt kein Leaking der "originalen" Positionen $0, \dots, n-1$ in
die Memory-Tokens; diese Positionen werden vom Hugin-Encoder-internen
Positional-Encoding absorbiert, das unabhängig vom Base-Modell arbeitet.

**Vokabular.** Memory-Tokens entsprechen keinem Eintrag im Vokabular
des Base-Modells. Sie sind Embeddings in $\mathbb R^d$, die nie durch
die Vocab-Projektion laufen. Der LM-Head oben im Base-Modell liest
nur von den Final-Layer-Positionen des *User-Turns* (und während des
Decodes generierter Tokens), muss also nie Memory-Embeddings zurück
zu Vokabular-Items invertieren. Analog zu wie BLIP-2s Q-Former-
Outputs konsumiert werden.

**Attention-Mask.** Während des Prefills von $[\mathbf M; \mathbf X_u]$
gilt die Standard-Kausalmaske: der User-Turn attendiert auf die
vorangehenden Memory-Tokens, umgekehrt nicht. Während des Decodes
attendieren generierte Tokens auf beides. Memory-Tokens attendieren
im Base-Model-Prefill nicht autoregressiv aufeinander — ihre Inhalte
sind durch Hugins eigenen internen Encoder-Pass bereits fixiert — der
Attention-Block behandelt sie also als vollbidirektionalen Prefix,
was wieder dasselbe Regime wie bei BLIP-2 ist.

### 4.6 Mehrere Dokumente

Bei RAG mit $C$ abgerufenen Chunks sind zwei Strategien mit dem
Design kompatibel:

- **Concat-then-Compress.** Hugin sieht die Konkatenation aller Chunks
  und emittiert eine einzige Menge $m$ Memory-Tokens. Günstiger zur
  Inference-Zeit, aber der Encoder muss bis zu $Cn$ Input-Tokens
  handhaben.
- **Compress-then-Concat.** Hugin läuft einmal pro Chunk, produziert
  $m$ Memory-Tokens pro Chunk; das Base-Modell sieht $Cm$ Memory-
  Tokens gefolgt vom User-Turn. Etwas weniger komprimiert, aber
  massiv cache-freundlich: die Memory-Tokens jedes Chunks sind eine
  reine Funktion dieses Chunks, können also einmal vorberechnet und
  über jede Query, die diesen Chunk abruft, wiederverwendet werden.

Die **Compress-then-Concat**-Variante ist die natürliche Paarung mit
pegenauts ChromaDB-Retrieval-Store, der ohnehin per-Chunk indexiert.
Hugin-Memory-Tokens können *neben* dem Chunk in Chroma gespeichert,
zusammen mit ihm abgerufen und ohne weiteren Hugin-Aufruf zur
Request-Zeit an mimirmind ausgeliefert werden. Erhebliche Deploy-
Vereinfachung: zur Request-Zeit muss Hugin bei Cache-Hits gar nicht
laufen; er läuft nur für die (kleine) Menge Chunks, die seit dem
letzten Index-Update frisch ingested wurden.

### 4.7 Runtime-Varianten-Auswahl (Autotune)

Varianten A, C und D dominieren jeweils einen anderen Punkt auf der
Qualität × Speed × Cacheability-Fläche. Statt statisch einen zu
wählen, lädt der Runtime alle drei Adapter (soweit verfügbar) und
wählt pro Request über einen **Autotune**-Layer, folgt dabei dem
etablierten Projekt-Muster (`GpuMatmul::autotune`, K-Tile-Autotune
für den Q8_0-GQA-Prefill-Kernel).

**Entscheidungs-Signale zur Request-Zeit gesammelt:**

- `n_doc`: Total abgerufene Dokument-Token-Anzahl.
- `n_answer_est`: erwartete Antwort-Längen-Klasse (aus `max_tokens`
  oder kleinem Prior).
- `query_kind`: billige heuristische Klassifikation — Regex +
  Keyword-Rules — liefert eine von
  `{paraphrase, extract_verbatim, numeric, multi_hop}`. Budget:
  unter 500 μs pro Request.
- `chroma_cache_hit`: sind die abgerufenen Chunks bereits mit
  Varianten-A- oder -C-Memory-Tokens in Chroma pre-compressed?
- `session_quality_signal`: Rolling-Window über User-Reprompt-,
  Klärungs-Muster oder explizite Thumbs-Down; wenn über Schwelle,
  Eskalation zu Higher-Quality-Variante.

**Entscheidungs-Logik (Short-Circuit, in Reihenfolge):**

```
if n_doc < config.hugin.min_document_tokens:
    variant = NONE           # unkomprimiert, direkt an Base-Modell

elif chroma_cache_hit and query_kind in {paraphrase, multi_hop}:
    variant = cached_variant  # A oder C, was ingested wurde
                              # → 0 s Encode

elif query_kind in {extract_verbatim, numeric}
     and n_doc > config.hugin.query_conditioned_threshold:
    variant = D               # query-conditioned, live Encode
                              # tauscht Chroma-Cache gegen Quality

elif n_doc > config.hugin.long_context_threshold:   # z.B. 12 000
    variant = C               # multi-stage, beste bei sehr langen Docs

else:
    variant = A               # Default General-Purpose
```

**Autotune-Kalibrierung.** Ein Winner-Cache indexiert über
`(adapter_version, ctx_bucket, query_kind_bucket, hardware_id)`
speichert die Variante, die für ein gegebenes Quality-Gate den
niedrigsten Wall-Clock auf dem pegenaut-internen QA-Slice
produziert hat. Der Cache wird durch einen Startup-Bench (drei
Warm-ups, zehn Measurement-Iterationen pro Kandidat) analog zu
[`GpuMatmul::autotune`](../src/compute/GpuMatmul.hpp) befüllt und
in einem kleinen `hugin_autotune.json` neben `config.json`
persistiert. Munin hält das File über Worker-Restarts.

**Prod-Quality-Safeguards** — kodifiziert aus
`lesson_dp4a_autotune_prod_hazard`:

1. **Niemals nur auf Wall-Clock-Delta auto-picken.** Der Winner-
   Cache verlangt, dass der Kandidat innerhalb 3 pp Quality-Marge
   der Baseline auf einem kleinen Held-out-Set liegt; erst
   *dann* ist Wall-Clock der Tiebreaker. Wall-Clock-Wins mit
   unbestätigter Quality-Kosten waren genau der Failure-Modus
   des DP4A-Autotune.
2. **Prior-Gewichtung für Chroma-cacheable Varianten.** Varianten
   A und C erhalten den Compound-Stack (Mode 2) intakt; der
   Autotune muss eine fixe Cost-Penalty (z.B. +200 ms) auf den
   Score von Variante D addieren, um den Verlust des Chroma-
   Cache-Pfads zu reflektieren — außer wenn ein Query-Kind-
   Signal das overridet. Ohne diesen Prior würde kleines Bench-
   Rauschen den Winner auf D flippen und den Compound-Stack-
   TTFT-Gewinn stillschweigend killen.
3. **Kill-Switch.** `MIMIRMIND_HUGIN_VARIANT={auto|A|C|D|off}`
   Env-Override für Dev- und Prod-Triage. Ledger-Eintrag
   verpflichtend pro Switch-Flip.
4. **Windowed-Adaptive-Re-Selection.** Wenn eine Session's
   `session_quality_signal` degradiert, eine Varianten-Tier
   eskalieren (A → C, oder A/C → D) für den Rest der Session;
   Eskalation loggen.

**Encoder-Invocation-Strategie zur Runtime:**

| Gewählte Variante | Encoder zur Query-Zeit? | KV-Cache-Prefix reusable? |
|:-----------------:|:-----------------------:|:-------------------------:|
| A (aus Chroma)    |          nein           |            ja             |
| C (aus Chroma)    |          nein           |            ja             |
| A (live)          |            ja           |            ja             |
| C (live)          |            ja           |            ja             |
| D                 |     ja, immer           |          nein             |
| NONE              |          nein           |            n/a            |

Zwei Adapter (`A` und `C`) teilen sich die Chroma-Pre-Compression-
Pipeline, weil beide query-agnostisch sind; Variante D ist per
Konstruktion inkompatibel mit Chroma-Pre-Compression und lebt
ausschließlich auf dem Live-Encode-Pfad.

**Aufwand:** die Runtime-Selection-Schicht ist ein kleines Stück
Code (~2 Tage inkl. Query-Kind-Klassifikator und Safeguards). Das
Gewicht des Milestones kommt aus dem Trainieren dreier Adapter
statt eines — siehe M-Hugin.2-Revision in §9.

---

## 5. System-Integration in MimirMind

Hugin ist eine reine Runtime-Optimierung, die in einen Codebase
passen muss, dessen Architektur bereits festgeschrieben ist. Dieser
Abschnitt bildet das Design auf das konkrete Component-Layout aus
`doc/architecture.md` ab.

### 5.1 Komponenten-Platzierung

```
┌────────────────────────────────────────────────────────────────┐
│                          ApiServer                             │
│  ChatCompletionHandler (POST /v1/chat/completions, JSON+SSE)   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                     RequestDispatcher                          │
│  routet zu Engine, verdrahtet Spec-Dec wenn aktiv              │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                     InferenceEngine                            │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  HuginAdapter    (neu — optional, per-Engine)            │  │
│  │    encode(document_tokens) → memory_embeddings           │  │
│  │    besitzt eigene SPIR-V-Kernels für Cross-Attn / MHA    │  │
│  │    reused UsmAllocator + L0Context der Engine            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ▼                                 │
│  ArchBackend (Qwen2 / Gemma4 / Gemma4E4B / Gemma4Moe / Dense)  │
│    runBlock(...) — unverändert                                 │
└────────────────────────────────────────────────────────────────┘
```

Der HuginAdapter ist eine **neue Klasse**, Geschwister von
`ArchBackend`, die eine einzige Methode implementiert: gegeben einen
Batch von Dokument-Token-IDs liefert sie einen Tensor der Shape
`[m, d]`, der in USM liegt. Dieser Tensor wird dann direkt in den
Residual-Stream-Input-Puffer des Base-Modells auf Positionen
$0 \dots m-1$ geschrieben, und der `runBlock` des Base-Modells wird
auf der konkatenierten `[m + k]`-langen Sequenz aufgerufen.

Die `ArchBackend`-Implementierungen des Base-Modells müssen nicht
verändert werden: aus ihrer Sicht hat der Input-Residual-Stream
einfach $m + k$ Positionen statt $n + k$.

### 5.2 GGUF-kompatibles Adapter-Format

Hugins Weights liegen als ein einzelnes GGUF-File mit einem Manifest-
Tag, das es als Hugin-Adapter identifiziert und die Residual-
Dimension des Base-Modells sowie den Tokenizer-Hash referenziert. Der
existierende GGUF-Loader (aus M3) wird mit nur zwei Ergänzungen genutzt:

- Ein neuer `AdapterKind::HUGIN`-Enum-Wert.
- Eine neue `HuginManifest`-Struktur, die $m$, $L_E$, $d_h$, die
  Base-Model-Residual-Dimension $d$ und den SHA-256 des Base-Tokenizers
  für Kompatibilitätsprüfung festhält.

Adapter werden als kleine unabhängige Files ausgeliefert, größenmäßig
im Bereich 50–500 MiB je nach Encoder-Größe. Sie sind ohne Rebuild
oder Container-Restart austauschbar: `POST /v1/admin/reload`
(definiert im Munin-Vorschlag, Abschnitt "Hot-Reload") triggert einen
Reload des aktuell aktiven Adapter-Pfads aus `config.json`.

Einen inkompatiblen Adapter (falsches $d$, falscher Tokenizer-Hash)
schon zur Load-Zeit abzuweisen ist essentiell — Silent-Mismatches
würden die Qualität unsichtbar degradieren, exakt der Failure-Modus
aus `lesson_dp4a_autotune_prod_hazard`. Der Compatibility-Check
gehört an die Loader-Grenze, nicht in den Encode-Pfad.

### 5.3 Config-Schema

Nach der `config.json`-Migration in Commit `456bd2a` wird Hugin über
einen einzigen Config-Block aktiviert:

```jsonc
{
  "hugin": {
    "enabled": true,
    "min_document_tokens": 512,
    "long_context_threshold": 12000,
    "query_conditioned_threshold": 4000,
    "adapters": {
      "A": "/models/adapters/hugin-gemma4-26b-a4b-A-m128.gguf",
      "C": "/models/adapters/hugin-gemma4-26b-a4b-C-m128.gguf",
      "D": "/models/adapters/hugin-gemma4-26b-a4b-D-m128.gguf"
    },
    "memory_tokens": 128,
    "autotune": {
      "enabled": true,
      "winner_cache_path": "/etc/mimirmind/hugin_autotune.json",
      "quality_gate_pp": 3.0,
      "variant_D_cost_penalty_ms": 200,
      "startup_bench": true
    },
    "cache": {
      "enabled": true,
      "backend": "chroma",
      "chroma_url": "http://pegenaut-chroma:8000",
      "collection": "hugin-memory-v1",
      "ingest_variants": ["A", "C"]
    }
  }
}
```

`min_document_tokens` ist die Schwelle, unter der Hugin übersprungen
wird und das Base-Modell den Rohtext direkt bekommt. Bei einem
200-Token-System-Prompt oder einer zweizeiligen Rückfrage ist
Kompression strikt ein Verlust. `long_context_threshold` und
`query_conditioned_threshold` sind die Cutoffs, die der Autotune in
seiner Entscheidungs-Logik (§4.7) benutzt. `ingest_variants` listet,
welche Varianten die Chroma-Pre-Compression-Pipeline pro Chunk
vorberechnet und speichert — Variante D fehlt bewusst, weil sie
query-conditioned ist.

Für Dev/Triage kürzt `MIMIRMIND_HUGIN_VARIANT={auto|A|C|D|off}`
den Autotune ab und pinnt eine Variante.

### 5.4 Zusammenspiel mit dem Munin-Persistenz-Daemon

Munin (siehe `research/munin-persistent-model-daemon.md`) allokiert
die Model-Weights einmal zum Startup in USM und exportiert IPC-
Handles an die Worker-Prozesse. Die Hugin-Adapter-Weights sind mit
diesem Modell kompatibel: sie werden von Munin einmal geladen, als
zusätzlicher IPC-shared-Tensor-Block exportiert und vom Worker
gleichzeitig mit den Base-Model-Tensoren attached.

Das KV- / Aktivierungs-Scratch des Hugin-Adapters ist per-Engine und
wird vom Worker zum Startup allokiert, exakt wie das Base-Model-
Scratch. Kein Cross-Process-State.

### 5.5 Zusammenspiel mit KV-Cache und existierenden Kerneln

Der KV-Cache des Base-Modells sieht pro Request nur $m + k$ Positionen.
Die KV-Cache-Größe bei 20 000 Rohtokens, komprimiert zu 256 Memory-
Tokens plus einem 512-Token-User-Turn, schrumpft von 3.2 GiB auf
`(256 + 512) / 20000 · 3.2 GiB ≈ 123 MiB` — Faktor 26. Das befreit
KV-Budget für entweder längere Decodes, mehr concurrent-Sessions oder
KV-dtype-Upgrades auf F16 (M10.2) ohne Druck.

Der FlashAttention-Q8_0-GQA-Prefill-Kernel (Commits `10e28a4` und
Folge-Commits) wird mit einer kürzeren Sequenzlänge aufgerufen. Weil
der Kernel O(seq²) Shared-Memory-Tile-Traffic pro Head hat, verstärkt
der Speedup aus kürzeren Sequenzen den Konstantenfaktor des
FlashAttention selbst — die zwei Hebel heben sich nicht auf.

RoPE-Anwendung ist unverändert — Memory-Tokens bekommen Positionen
$0 \dots m-1$ und werden RoPE-encoded wie normale Tokens.

### 5.6 Zusammenspiel mit pegenaut

Pegenaut ist der beabsichtigte Consumer. Zwei Integrations-Modi:

**Mode 1 — transparent.** Pegenaut merkt nichts. MimirMind bekommt
den vollen RAG-Prompt, Hugin komprimiert die abgerufenen Chunks
intern, pegenaut sieht nichts davon. Kleinster Code-Change, nutzt
aber kein Per-Chunk-Caching in Chroma.

**Mode 2 — pre-compressed.** Pegenaut schickt jeden Chunk zur
Document-Ingest-Zeit an einen neuen mimirmind-Endpoint
`POST /v1/hugin/encode` und speichert die zurückgegebenen Memory-
Tokens neben dem Chunk in Chroma. Zur Query-Zeit schickt es einen
Request, der *Memory-Tokens* statt *Dokumententext* enthält, und
mimirmind überspringt den Hugin-Encode für diesen Chunk komplett.

Mode 2 ist eine kleine Erweiterung des OpenAI-kompatiblen Endpoints
(neues Feld `content_kind: "hugin_memory"` auf Message-Parts),
liefert aber den größten Wall-Clock-Win, weil Hugins Encoder-Schritt
über viele Queries auf denselben Chunk amortisiert. Empfohlen als
M-Hugin.3-Ziel.

---

## 6. Performance-Modell

Ich leite die erwarteten Wall-Clock-Verbesserungen aus dem gemessenen
bandbreitengebundenen Cost-Profil des aktuellen Runtimes ab, im
selben Stil wie die Analysen in
`research/roadmap-speculative-decoding` und
`research/roadmap-gemm-xelpg-native-rewrite`.

### 6.1 Die drei Cost-Terme des Prefills

Für eine Token-Sequenz der Länge $T$, die einmal durch das Base-Modell
im Prefill läuft, dekomponiert sich der Wall-Clock:

$$
t_{\text{prefill}}(T) \approx
    \underbrace{L \cdot 2 T d^2 / B}_{\text{QKVO+FFN Weights}} +
    \underbrace{L \cdot 2 T^2 d / B}_{\text{Attention-KV-Traffic}} +
    \underbrace{L \cdot c_{\text{kernel}}}_{\text{Dispatch}}
$$

mit $L$ als Anzahl der Layer, $d$ als Residual-Dimension, $B$ als
sustained Bandbreite zur GPU, und $c_{\text{kernel}}$ als fixen
Dispatch-Cost pro Kernel (10–15 μs auf Xe-LPG per
`lesson_xelpg_dispatch_overhead`). Für Gemma 4 MoE ist der erste Term
*pro aktivem Experten*, nicht pro Layer, und seine effektive Konstante
ist kleiner als bei Dense.

Bei $T = 20\,000$ auf Gemma 4 26B-A4B dominiert der zweite Term
(Attention-KV-Traffic); der erste Term ist grob proportional zu $T$,
aber der zweite wächst quadratisch. Unterhalb von $T \approx 4\,000$
dominiert der erste Term, und der Prefill skaliert linear in $T$.

### 6.2 Effekt der Kompression auf $m$ Memory-Tokens

Hugin ersetzt $T$ durch $m + k$, wobei $k$ die User-Turn-Länge ist,
typisch 100–1000. Substitution für 20 000-Token-Dokument und
$m = 256$, $k = 512$:

- Erster Term schrumpft um Faktor $20000/(256+512) \approx 26$.
- Zweiter Term schrumpft um Faktor $(20000)^2/(256+512)^2 \approx 680$.
- Dritter Term unverändert (weiterhin $L$ Kernels, selbe Fixkosten).

Weil der zweite Term bei langen Kontexten dominant ist, ist der
Gesamtspeedup auf Prefill-Wall-Clock bei 20 000 Tokens nach oben
begrenzt durch ungefähr Faktor 300–500 *vor* Berücksichtigung der
Encoder-Kosten, und bei kurzen Kontexten (wo der zweite Term noch
nicht dominant ist) durch ungefähr Faktor 26.

### 6.3 Die Encoder-Kosten

Hugins eigener Encoder ist ein kleiner Transformer: $L_E$ Layer auf
$n$ Input-Tokens mit Residual-Dimension $d_h$. Seine Kosten sind
grob:

$$
t_{\text{encode}}(n) \approx L_E \cdot (2 n d_h^2 + 2 n^2 d_h) / B
$$

Für die "Klein"-Konfiguration ($L_E = 4$, $d_h = 512$) auf $n = 20\,000$:

- Weight-Term: $4 \cdot 2 \cdot 20000 \cdot 512^2 / (55 \text{ GB/s})
  \approx 750$ ms
- Attention-Term (unkomprimierte Self-Attention innerhalb des Encoders):
  $4 \cdot 2 \cdot 20000^2 \cdot 512 / (55 \text{ GB/s}) \approx 30$ s

Der Attention-Term innerhalb von Hugins eigenem Encoder ist der
Killer, wenn wir naive Full-Self-Attention auf dem Rohdokument
laufen lassen. Zwei Milderungen sind strukturell verfügbar:

1. **Nur Perceiver-artiges Read-out.** Der Encoder attendiert nicht
   zwischen den Dokument-Tokens selbst; er cross-attendiert nur *von*
   den $m$ Latent-Queries *in* die Dokument-Tokens. Attention-Kosten
   sind $O(m n)$, nicht $O(n^2)$. Für $m = 256$ und $n = 20\,000$
   Faktor 78 Reduktion — kritisch. Genau das Design aus Abschnitt 4.3.
2. **Gechunktes Dokument-Processing.** Encoder bekommt Chunks von
   $n_c = 2048$ Tokens und pooled Latents. Fallback, falls das
   Read-out-Design qualitätsseitig nicht konvergiert.

Mit Perceiver-artigem Read-out sind Hugins Encoder-Kosten grob:

$$
t_{\text{encode}}(n) \approx L_E \cdot (2 n d_h^2 + 2 m n d_h) / B
$$

Für die "Klein"-Konfiguration mit $L_E = 4$, $d_h = 512$, $m = 256$,
$n = 20\,000$ bei $B = 55$ GB/s:

- Weight-Term: 750 ms wie oben
- Cross-Attention-Term: $4 \cdot 2 \cdot 256 \cdot 20000 \cdot 512 /
  55\,\text{GB/s} \approx 380$ ms

Encoder-Gesamtkosten: **~1.1 s** für ein 20 000-Token-Dokument in
dieser Konfiguration.

### 6.4 Break-Even und Gesamt-Wall-Clock

Unkomprimierter 20 000-Token-Prefill auf 26B-A4B (Ledger-Extrapolation):
**~80 s**.

Hugin-komprimierter Prefill über $256 + 512 = 768$ Tokens auf demselben
Modell: linear extrapoliert aus der 1 331-Token-/5.9-s-Zeile ungefähr
**3.4 s**.

Encoder-Schritt: **~1.1 s** (siehe 6.3), komplett skippbar bei
Mode 2.

**Live-Mode-Total:** $1.1 + 3.4 = 4.5$ s pro Request.
**Pre-Compressed-Mode-Total:** $3.4$ s pro Request.

Speedup vs unkomprimierter Baseline: **~18×** (Live) bis **~24×**
(Pre-Compressed) bei 20 000-Token-Kontexten.

Unterhalb 4 000 Tokens schrumpfen die Wins deutlich, unterhalb 2 000
Tokens können Encoder-Kosten die Ersparnis übersteigen; der
`min_document_tokens`-Guard im Config (Abschnitt 5.3) existiert genau
für dieses Regime.

### 6.5 Prediction-Hazard-Notice

Das obige Performance-Modell beruht auf denselben Annahmen, die dem
Projekt schon einmal weh getan haben (siehe
`lesson_dp4a_autotune_prod_hazard` und
`lesson_xelpg_dp4a_compute_ratio`): nämlich dass Speicherbandbreite
der dominante Kostenfaktor ist und dass Xe-LPG-Dispatch-Overhead gut
charakterisiert ist. Beide Annahmen halten *auf den Pfaden, die wir
schon gemessen haben*, aber keine wurde für die konkrete Kernel-
Form, die ein Hugin-Encoder benutzt, vermessen. Bevor wir einen
Bau-Pfad committen, sollten wir einen kleinen SPIR-V-Microbench
laufen lassen, der den Inner-Loop des Encoders auf realistischen
Tensor-Shapes reproduziert und den 750-ms-Weight-Term validiert.
Dieselbe Disziplin, die `research/roadmap-speculative-decoding` in
ihrem "Attention-Share validieren vor Build"-Gate kodifiziert hat.

### 6.6 Der Compound-Speed-Stack

Hugin allein ist bei einem Long-Context-RAG-Request ein ~4×-Hebel auf
End-to-End-Wall-Clock. Das ist nicht der Grund es zu bauen. Der Grund
ist, dass Hugin mit bereits geshippten und geplanten Speed-Hebeln
komponiert, und das Compound-Verhalten deutlich besser ist als jeder
einzelne Hebel.

Der Stack in Kompositions-Reihenfolge, von Input zu Output:

1. **Optionale LLMLingua-Token-Pruning.** Für Kontexte über
   ~10 000 Tokens: erst niedrig-informative Tokens mit einem kleinen
   Selektor-Modell fallen lassen. Publizierte Ratio 15×–20× bei
   ~1.5 pp Accuracy-Verlust. Reduziert den Input zum Hugin-Encoder,
   der die Hugin-Live-Mode-Kosten dominiert.
2. **Per-Chunk-Hugin-Kompression.** Jeder abgerufene Chunk
   unabhängig auf $m$ Memory-Tokens komprimiert; in Chroma neben dem
   Chunk gecacht (§5.6, PISCO-Muster).
3. **RadixAttention-artiger Prefix-Cache** auf dem Memory-Token-
   Prefix. Auch wenn Hugins Memory-Tokens pro Request variieren,
   teilen sich viele RAG-Turns den System-Prompt und den adapter-
   skalierten BOS; diese Positionen bleiben gecacht.
4. **KV-Cache in F16** (M10.2 Phase 1a, live).
5. **FlashAttention-Q8_0-GQA-Prefill-Kernel** auf der verkürzten
   Sequenz (bereits live).
6. **MTP-Drafter** auf dem Decode (geplant als M-MTP nach M8.K).
7. **Model-Routing** — E4B für Short-Answer-Pfade, 26B-A4B für
   Long-Form. Anwendbar wenn `stream=true` und der Client einen
   Chat-Stil statt Analytik signalisiert.

**Compound-Projektionen** auf einem 20 000-Token- / 300-Token-Decode-
Request auf Gemma 4 26B-A4B Q8_0, Prod-Governor:

| Stage                                          | Prefill (s) | Decode (s) | Total (s) | Speedup |
|------------------------------------------------|------------:|-----------:|----------:|--------:|
| Baseline (heute)                               |          88 |         66 |       154 |    1.0× |
| + Hugin (Live-Mode, $m=256$)                   |         3.1 |         66 |      69.1 |    2.2× |
| + Hugin (pre-compressed, $m=128$)              |         2.0 |         66 |      68.0 |    2.3× |
| + LLMLingua vor Hugin (nur Live)               |         2.1 |         66 |      68.1 |    2.3× |
| + MTP-Drafter (2.5× Decode)                    |         2.0 |       26.4 |      28.4 |    5.4× |
| + LLMLingua + Hugin + MTP + F16 KV             |         1.9 |       24.0 |      25.9 |    5.9× |
| + Model-Routing auf E4B für kurze Antworten    |         1.2 |        7.5 |       8.7 | **18×** |

Die dominanten Terme verschieben sich, während wir stapeln. In der
Baseline-Zeile dominiert Prefill (57 %). Nach Hugin allein dominiert
Decode (96 %). Nach MTP sind Prefill und Decode ausgeglichen (~7 % /
~93 %). Genau deshalb ist der Stack wichtig: nur Prefill anzugreifen
trifft schnell diminishing Returns; nur Decode anzugreifen lässt die
massive 88-s-Prefill-Basis stehen.

**Time-to-First-Token** (TTFT), die subjektiv als "Responsiveness"
empfundene Metrik:

| Stage                                     | 8 k ctx TTFT | 20 k ctx TTFT |
|-------------------------------------------|-------------:|--------------:|
| Baseline                                  |         35 s |          88 s |
| + Hugin Live-Mode                         |        2.4 s |         3.1 s |
| + Hugin pre-compressed                    |        2.0 s |         2.0 s |
| + LLMLingua + Hugin pre-compressed        |        2.0 s |         2.0 s |
| + Alles auf E4B-Route                     |        1.3 s |         1.3 s |

Bei 20 000-Token-Kontexten schrumpft TTFT von 88 Sekunden auf ~1.3
Sekunden — **~68×** auf der Metrik, die Nutzer tatsächlich wahrnehmen.

**Wie stage-t man den Stack.** Die Compound-Zahlen materialisieren
sich nur, wenn die untenstehende Reihenfolge während M-Hugin
respektiert wird:

- M-Hugin.1 liefert Hugin allein → **~2.3× Total, ~28× TTFT** auf
  20-k-Kontexten.
- MTP-Drafter (unabhängiger Milestone M-MTP) → **~5.4× Total**.
- Pegenaut-Pre-Compression (M-Hugin.3) → **~5.9× Total, ~44× TTFT**.
- Model-Routing (reine Config-Änderung auf pegenaut-Seite) →
  **~18× Total, ~68× TTFT**.

Model-Routing ist der billigste Hebel dieser Liste und lohnt sich
schon vor Hugin — als Control-Experiment.

**LLMLingua's abnehmender Grenznutzen.** Sobald der Encoder
Perceiver-shape hat (nur Cross-Attention, keine Full-Self-Attn), sind
die Hugin-Encode-Kosten schon linear in $n$ und klein
(~55 μs/Token). LLMLingua würde 20 000 auf ~1 000 Tokens vor Hugin
droppen und Encode von 1.1 s auf ~55 ms senken — reale Ersparnis im
Live-Mode, aber null im Pre-Compressed-Mode, weil der Encode ohnehin
amortisiert ist. Empfehlung: **LLMLingua in M-Hugin.1 und M-Hugin.2
überspringen**, nur evaluieren wenn Live-Mode-TTFT bei sehr langen
Kontexten nach M-Hugin.3 noch relevant ist.

**xRAG's 1-Token-pro-Chunk-Regime.** Bei $m=1$ pro Chunk würden fünf
abgerufene Chunks fünf Prefix-Positionen belegen. Der theoretische
TTFT-Gain ist real, aber xRAG braucht Base-LLM-Fine-Tuning (Frozen-
Base-Vertrag gebrochen). Als Research-Follow-up unter M-Hugin.4
abgelegt, nur falls wir später die Option zum Fine-Tuning des Base-
Modells selbst gewinnen.

---

## 7. Evaluation

### 7.1 Korrektheits-Parity

Die primäre Korrektheitsfrage ist, ob $F_\phi([\mathbf M; \mathbf X_u])$
Antworten derselben Qualität produziert wie $F_\phi([\mathbf X_d;
\mathbf X_u])$. Zwei Metriken:

- **Token-Level-Distributional-Distance.** KL-Divergenz zwischen den
  Next-Token-Verteilungen des Base-Modells an jeder Answer-Position,
  gerechnet auf einem Held-out-RAG-QA-Set. Schwelle: Median-KL
  unterhalb der KL zwischen dem selben Base-Model bei zwei
  Temperatur-Noise-Runs auf demselben Input — natürliche Obergrenze
  für "praktisch gleich".
- **Task-Level-Answer-Quality.** Exact-Match und F1 auf:
  - NarrativeQA (Long-Form-Dokument-QA)
  - HotpotQA (Multi-Hop, retrieval-augmentiert)
  - einem Held-out-Slice des pegenaut-Prod-Corpus

Akzeptanz: Task-Level-Quality innerhalb von 2 Prozentpunkten der
unkomprimierten Baseline beim größten getesteten Kompressionsverhältnis.

### 7.2 Referenz-Orakel für Kernel-Parity

Der Hugin-Encoder ist neuer Code, der neue SPIR-V-Kernels produziert.
Die Projekt-Regel (CLAUDE.md, "Reference Oracle") ist, dass neue
Kernels elementweise gegen `llama-cli` validiert werden. Hugin hat
kein `llama-cli`-Pendant — kein Äquivalent im Upstream — also ist die
Correctness-Referenz während der Implementierung eine **CPU-only
float64-Implementierung desselben Encoder-Graphen** in einem
`gpu_tests`-Binary, auf denselben Weights ausgeführt. Abweichungs-
Toleranz: 1e-5 relativ pro Element auf Q8_0-quantisierten Weights,
1e-3 auf Post-Softmax-Aktivierungen.

### 7.3 Wall-Clock-Parity

Bench-Modus-Runs auf L0_TARGET_HOST mit Governor per
`operations.md` gepinnt, drei Warm-ups, zehn Measurement-Iterationen,
p50 und p95 melden. Ledger-Eintrag verpflichtend per Standing-Rule
`feedback_perf_ledger`.

Vergleichsziele:

- Baseline: Current-Tip, kein Hugin, kein Spec-Dec, Live-Prod-Kernels.
- Nur Hugin, Live-Mode.
- Nur Hugin, Pre-Compressed-Mode.
- Hugin + MTP-Drafter (Abschnitt 8 von `mtp-diffusiongemma-status`).

### 7.4 Quality-Loss-Stress-Cases

Kompression schadet bekannt bei *Precise-Quote-Retrieval*-Fragen und
*Exact-Numeric*-Fragen. Ein eigenes Stress-Subset der Evaluation deckt ab:

- Paraphrase-Fragen ("Was sagt das Dokument über X?")
- Extract-Precise-String-Fragen ("Kopiere die Section-3-Überschrift
  wörtlich")
- Numeric-Fragen ("Welche Deadline steht in Anhang B?")
- Multi-Hop-Reasoning über mehrere komprimierte Chunks

Wenn Hugin Paraphrase und Multi-Hop besteht, aber Extract-Verbatim
und Numeric verliert, ist das ein reales Deployment-Constraint, das
dokumentiert und in einer Routing-Heuristik reflektiert werden muss:
bei extraktiv anmutenden Fragen unkomprimiert servieren.

---

## 8. Risiken und Grenzen

### 8.1 Qualitäts-Risiko

Aggressive Kompression verliert Information. Bei 300× Kompression ist
das Informations-Budget pro Memory-Token weniger als ein Dutzend
Quell-Token-Entropien. Perceiver und BLIP-2 berichten beide einen
sanften graduellen Abfall mit Kompressionsverhältnis; Gisting
berichtet einen Knick um 26×; ICAE um 15×. Hugins Zielverhältnisse
(50×–300×) sind aggressiv gegenüber jeder publizierten Baseline, und
wir sollten damit rechnen, dass der erste Prototyp Moderation zu
$m \geq 512$ oder 8×–32×-Verhältnissen braucht, bevor die Qualität
prod-akzeptabel ist.

Die richtige Framings-Perspektive ist, dass Hugin das RAG-Betriebs-
Fenster *erweitert*: Kontexte, die wir heute nicht bedienen (weil
Prefill 80 s dauert), werden bedienbar, ggf. mit kleinem Qualitäts-
Trade-off. Kontexte, die wir schon gut bedienen, bleiben über den
`min_document_tokens`-Guard unkomprimiert.

### 8.2 Trainings-Komplexität

Adapter-Training für ein bestimmtes Base-Modell ist ein GPU-Tage-Job,
der neu gemacht werden muss, sobald das Base-Modell wechselt. Für ein
Projekt mit Gemma 4 als Prod-Ziel ist das begrenzt, aber die ADR-
Ebene-Entscheidung Hugin zu bauen muss die Wartungskosten mehrerer
trainierter Adapter über Gemma 4 26B-A4B, Gemma 4 E4B und ein
etwaiges Gemma 4 31B Dense berücksichtigen.

### 8.3 Deployment-Komplexität

Mode 2 (pre-compressed Chunks in Chroma) verlangt Koordination
zwischen mimirmind und pegenaut: eine Chunk-Ingest-Pipeline, ein
Adapter-Version-Tag auf jedem gespeicherten Memory-Token-Blob, eine
Invalidierungs-Strategie bei Adapter-Retrain. Handhabbar, aber
nicht null Engineering.

### 8.4 Prediction-Hazard

Abschnitt 6.5 flaggt das schon. Jede Perf-Zahl in diesem Paper ist ein
Modell, keine Messung. Der Projekt-Track-Record mit prediction-
getriebener Perf-Arbeit ist gemischt (DP4A, M5f.5). Der M-Hugin.1-
Milestone muss ein Bandwidth-Microbench-Gate *vor* dem Trainings-
Commitment enthalten.

### 8.5 Interaktion mit Speculative Decoding

Hugin komprimiert den Input; Speculative Decoding beschleunigt den
Output. Strukturell kompatibel — der Forward-Pass des Base-Modells
ist in beiden Fällen unverändert — aber die zwei Hebel zusammen
compoundieren das KV-Cache-Size-Delta, weil Spec-Dec KV-Storage
temporär für den Draft verdoppelt. Gutes Problem, aber gemessen,
nicht angenommen.

### 8.6 Multi-Turn-State

Sobald Memory-Tokens für ein Dokument erstellt sind, drängt sich die
Frage auf, ob sie über Turns einer Konversation persistieren sollen.
Die Antwort ist ja, und der M-Hugin.3-Milestone erweitert die
`session-kv-cache`-Roadmap-Note um genau das: Memory-Tokens werden
zusammen mit dem KV-Cache im Session-State abgelegt, und nur der neue
User-Turn wird pro Runde frisch prefilled.

### 8.7 Vocab- und Base-Model-Kompatibilität

Der Hugin-Adapter ist an die Embedding-Matrix und Residual-Dimension
eines bestimmten Base-Modells gebunden. Ein Base-Model-Wechsel
invalidiert den Adapter. Kein Bug — dasselbe Constraint wie bei
Prefix-Tuning — aber es heißt, dass das Adapter-GGUF einen forcierten
Tokenizer-Hash- und Residual-Dimension-Check tragen muss, den der
Loader *bevor* der Encoder je läuft honoriert. Bei Mismatch auf
unkomprimiert zurückfallen ist besser als Silent-Garbage.

---

## 9. Meilenstein-Zerlegung — M-Hugin

Phase: post-Mimir-1.0. Roadmap-Platzierung nach Munin (das das Zero-
Downtime-Deploy liefert, eine Voraussetzung dafür, Adapter-Versionen
in Prod zu iterieren).

### M-Hugin.1 — Runtime-Integration + Variante A auf Gemma 4 E4B (1.5–2 Wochen)

Aufwand als Runtime-Integrations-Milestone gescoped, weil Encoder-
Architektur und Trainings-Rezept aus COCOM portiert werden.

- Bandwidth-Microbench-Gate: reproduziere den 750-ms-Weight-Term aus
  §6.3 auf echten Xe-LPG-Kerneln. Blocker für den Rest des
  Milestones.
- `HuginAdapter`-Klasse + SPIR-V-Kernels für Cross-Attention und
  Projection.
- GGUF-Adapter-Format + Loader-Integration (Kompatibilitäts-Check
  über Tokenizer-Hash und Residual-Dimension).
- CPU-only-Reference-Oracle-Test-Rig in `gpu_tests`.
- Bereits trainierten COCOM-artigen Checkpoint laden (falls für ein
  kleines Modell mit Gemma-Tokenizer verfügbar), sonst einen ersten
  kleinen Variant-A-Adapter nach COCOM's publiziertem Trainings-
  Rezept auf FineWeb-Edu trainieren (1–3 GPU-Tage auf gemieteter
  H100).
- Wall-Clock-Parity-Report gegen unkomprimiertes E4B auf 4 000-,
  8 000- und 16 000-Token-Kontexten.
- Go-/No-Go-Entscheidung: erreicht der Adapter Qualitäts-Parity
  innerhalb 2 pp bei 8× Kompression?

### M-Hugin.2 — Variante A + C für Gemma 4 26B-A4B, Autotune-Framework (3–3.5 Wochen)

- Trainiere Variant-A-"mittel"-Adapter für Gemma 4 26B-A4B auf
  pegenaut-RAG-Corpus über ICAE-artige Self-Distillation. Objective
  und Curriculum aus ICAE-/AutoCompressors-Code übernommen, keine
  neue Trainings-Forschung.
- Trainiere Variant-C-Adapter (Multi-Stage, Re-Injection an Layer 8).
  Verwendet denselben Corpus und dasselbe Objective; addiert einen
  zweiten kleinen Adapter-Block, trainiert nachdem der erste
  konvergiert ist.
- **Autotune-Framework** (§4.7): Winner-Cache, Startup-Bench, Drei-
  Signal-Entscheidungs-Logik (zunächst nur A vs. C vs. unkomprimiert),
  `hugin_autotune.json`-File-Layout, Munin-shared-Persistenz.
- Query-Kind-Klassifikator-Stub (Regex + Keyword, Budget <500 μs);
  nur Paraphrase-vs-Long-Doc-Split wird in diesem Milestone
  exerziert.
- `config.json`-`hugin`-Block (§5.3) mit `adapters.{A,C}`-Slots.
- Live-Mode-A/B auf L0_TARGET_HOST gegen Prod-Baseline zunächst
  für A allein, dann A+C+Autotune zusammen; Ledger-Eintrag mit
  Wall-Clock-Delta, Energie-Delta, Quality-Delta auf pegenauts
  internem QA-Set.
- Roll-out-Politik: `MIMIRMIND_HUGIN_VARIANT=auto` in der ersten
  Ledger-Runde Default-off, dann nach Quality-Gate-Bestätigung
  flipped.

### M-Hugin.3 — Variante D + Query-Conditioned Autotune (2 Wochen)

- Trainiere Variant-D-Adapter (query-conditioned, Single-Stage) auf
  pegenaut-RAG-Corpus. Trainings-Daten enthalten
  (Dokument, Query, Answer)-Tripel, damit der Encoder lernt, welche
  Dokument-Inhalte für welche Frage wichtig sind.
- Query-Kind-Klassifikator erweitern um Unterscheidung
  `paraphrase`, `extract_verbatim`, `numeric`, `multi_hop`; kleines
  Held-out-Labeled-Set auf pegenaut-Corpus für Kalibrierung.
- Autotune erweitern um Variante D mit den Safeguards aus §4.7
  (Quality-Gate-Marge, Cost-Penalty-Prior, Kill-Switch).
- A/B auf den vier Query-Kind-Buckets vs. M-Hugin.2-Baseline;
  Ledger-Eintrag.

### M-Hugin.4 — Pre-Compressed-Mode + Compound-Stack-Integration (1.5–2 Wochen)

- `POST /v1/hugin/encode`-Endpoint auf mimirmind.
- Pegenaut-ChromaDB-Ingest-Erweiterung: bei Chunk-Insert
  `/v1/hugin/encode` aufrufen für `ingest_variants=[A, C]`,
  Memory-Tokens neben Chunk-Content speichern, mit Adapter-Version
  taggen.
- Query-Pfad: pegenaut ruft Memory-Tokens zusammen mit (oder statt)
  Text ab, schickt sie als `content_kind: "hugin_memory"`; Autotune's
  `chroma_cache_hit`-Signal wired dazu.
- Cache-Invalidierungs-Politik bei Adapter-Retrain.
- Full-Compound-Stack-A/B: Autotune-gewähltes Hugin + MTP + F16 KV +
  Model-Routing auf Short-Answer-Pfaden. Ledger-Eintrag mit den
  Full-Stack-Zahlen aus §6.6.

**Total geplanter Aufwand: ~8–9.5 Wochen.**

Der Schritt-Up vom früheren ~5-6-Wochen-Estimate kommt komplett vom
Zufügen der Varianten C und D plus Autotune-Framework, was der
explizite User-Wunsch war; die algorithmischen Komponenten werden
weiterhin portiert, nicht designed.

### M-Hugin.5 — Zwei-Stufen-Hybrid (contingent, nicht scheduled)

Nur wenn die pegenaut-Corpus-Quality-Analyse unter M-Hugin.3 eine
klare Lücke zwischen Variante A/C (cacheable aber query-agnostisch)
und Variante D (query-aware aber nicht cacheable) zeigt, die einen
**Zwei-Stufen-Hybrid** motiviert: eine große query-agnostische
Stage-1-Latent-Menge in Chroma pro Chunk gespeichert, und ein
kleiner query-conditioned Stage-2-Compressor zur Query-Zeit auf
kombinierten Stage-1-Latents + Query. Geschätzt 3–4 Wochen. Abgelegt
als Research, nicht scheduled.

---

## 10. Verworfene Alternativen

### 10.1 Chunked-Prefill mit Early-Exit

Nur die Top-K abgerufenen Tokens pro Query ins Base-Modell speisen,
gerankt von einem kleinen Classifier. Einfacher als Hugin, aber wirft
RAG-Signal außerhalb der Top-K weg; keine Vorstellung von
"komprimiertem Whole-Document-Kontext".

### 10.2 Retrieval auf jeder Layer (RETRO-artig)

Chunks abrufen und auf jeder Layer des Base-Modells cross-attendieren.
Setzt eine Base-Model-Modifikation voraus — verletzt den Frozen-Base-
Vertrag. Als Forschung für später abgelegt, falls wir jemals ein
eigenes Base-Modell kontrollieren.

### 10.3 Längeres-Kontext-Base-Modell + FlashAttention 3

Einfach längere Kontexte nativ mit besserem Attention-Kernel bedienen.
Konstantenfaktor-Angriff, nicht Token-Anzahl-Angriff. Der
FlashAttention-Q8_0-GQA-Prefill-Kernel ist schon gelandet (Commit
`10e28a4`) und sein Win liegt im 20 %-Bereich, nicht im 20×-Bereich,
den Hugin anpeilt.

### 10.4 KV-Cache-Wiederverwendung über Prefix-Caching (RadixAttention)

KV-Zustand über Requests mit geteiltem Prefix wiederverwenden.
Komponiert mit Hugin, ersetzt es nicht. Fügt erheblichen Storage-
Aufwand für lange geteilte Prefixe hinzu; Hugin reduziert diesen
Storage-Aufwand um den Kompressionsfaktor.

### 10.5 llama.cpp shippen

Keine ernsthafte Alternative angesichts der CLAUDE.md-Projekt-Charter,
aber der Vollständigkeit halber genannt: llama.cpp implementiert
heute nichts Hugin-Artiges, und seine RAG-Story auf Xe-LPG ist nicht
konkurrenzfähig.

### 10.6 Gar nichts tun jenseits vom bereits Geshippten

Sich komplett auf die Konstantenfaktor-Kernel-Arbeit verlassen
(FlashAttention, Q8_0, KV-F16, GEMM-Prefill, CLR). Prefill auf 20-k-
Token-Kontexten bleibt bei etwa einer Minute. RAG-Turns über 10 k
Tokens bleiben subjektiv unbrauchbar, und pegenaut kann für corpus-
lastige Queries keine interactive-feeling Antworten liefern. Das ist
der Ist-Zustand und die Null-Hypothese, gegen die alle M-Hugin-
Ledger-Einträge gemessen werden.

---

## 11. Summary

Retrieval-Augmented Generation auf integrierten GPUs ist ein
prefill-gebundener Workload, dessen dominanter Kostenfaktor
Speicherverkehr proportional zur Anzahl der Tokens ist, die ins
Base-Modell fließen. Jede Kernel-Level-Optimierung, die mimirmind
in diesem Quartal geshippt hat, greift eine Konstante dieser
Gleichung an; keine reduziert die Token-Anzahl selbst.

**Hugin** greift diese Token-Anzahl an, indem ein etabliertes
Input-seitiges Soft-Embedding-Kompressions-Muster (COCOM / PISCO /
ICAE / Compressed Context Memory) auf unseren spezifischen Runtime
portiert wird: Level-Zero, USM, GGUF-Loader, Munin-Persistenz,
pegenaut-Chroma. Der algorithmische Inhalt ist nicht neu — das Feld
publiziert seit 2023 Adapter mit Kompressionsverhältnissen, die
unsere Ziele erreichen oder übertreffen — aber keine Open-Source-
Implementierung zielt auf diese Hardware-/Runtime-Kombination.

Die geplante Arbeit ist Engineering-Integration in vier Meilensteinen
(~8–9.5 Wochen), plus ein Distillations-Trainings-Pass pro Adapter-
Variante off-target auf einer gemieteten Discrete-GPU. Drei Adapter-
Varianten koexistieren zur Runtime — Variante A (Single-Stage,
query-agnostisch, Chroma-cacheable), Variante C (Multi-Stage mit
Layer-8-Re-Injection, query-agnostisch, Chroma-cacheable) und
Variante D (Single-Stage, query-conditioned, nicht cacheable) — mit
Per-Request-Auswahl über einen Autotune-Layer nach dem existierenden
`GpuMatmul::autotune`-Muster. Jeder Milestone ist durch einen
gemessenen Vergleich gegen die unkomprimierte Baseline auf pegenaut-
skynet und einen Ledger-Eintrag entsprechend der Standing-Perf-Ledger-
Regel gegated.

Der isolierte Hugin-Hebel liefert **~2.3× End-to-End-Wall-Clock**
und **~28× TTFT** auf 20 000-Token-RAG-Kontexten. Der Grund Hugin
zu bauen ist nicht die isolierte Zahl, sondern der Compound-Stack,
den er ermöglicht: kombiniert mit der bereits geshippten KV-F16-
Foundation, FlashAttention Q8_0-GQA, RadixAttention-artigem Prefix-
Cache, plus den geplanten MTP-Drafter und pegenaut-seitigem Model-
Routing projiziert der Stack **~18× End-to-End** und **~68× TTFT**
auf denselben Requests. Genau diese Compound-Zahlen sind das Ziel
des Projekts.

Nichts an diesem Vorschlag ist publikationswürdig, und er ist nicht
so gedacht. Es ist ein Engineering-Pfad auf unseren existierenden
Runtime, der unbrauchbares Long-Context-RAG (80-s-Prefill) in eine
responsive Interaktion (~1–3-s-TTFT) verwandelt — durch Kombination
von Hebeln, die einzeln bereits existieren.

---

## 12. Empfohlener Rollout für ein Ein-Personen-Projekt

Der komplette M-Hugin.1–.4-Bau-Pfad ist ~8–9.5 Wochen fokussiertes
Engineering plus gemietete GPU-Zeit für drei Adapter-Retrains. Für
ein Ein-Personen-Projekt ist das ein großer Block, der committed
werden müsste, bevor die zugrundeliegenden Annahmen vermessen sind.
Der phased Rollout unten reduziert Risiko, indem er die Arbeit so
ordnet, dass jeder Schritt entweder user-sichtbaren Wert liefert
oder eine harte Messung produziert, die den nächsten Schritt gated.

### Phase 0 — Model-Routing (1–2 Tage, sofort)

Pegenaut-seitige Config-Änderung: Short-Answer-Requests auf Gemma 4
E4B routen, Long-Form auf Gemma 4 26B-A4B. Kein mimirmind-Code-
Change; kein Adapter; kein Training.

- Erwarteter Win: **3–4× auf Short-Answer-Wall-Clock**, keine
  Quality-Kosten auf Chat-Style-Traffic.
- Doppelfunktion als Control-Experiment: schließt Routing allein
  genug von der UX-Lücke, dass Hugin nicht mehr der größte Hebel
  ist?

**Exit-Bedingung:** Ledger-Eintrag mit Routing-on / Routing-off-
Delta auf dem tatsächlichen pegenaut-Traffic-Mix.

### Phase 1 — Traffic-Instrumentierung (2–3 Tage)

Log pro Request Kontextlänge und Query-Kind auf pegenaut für
1–2 Wochen. Verteilung sammeln.

**Entscheidungs-Gate:**

- Median-Kontext < 4 k Tokens → **Hugin nicht bauen.** Zeit in
  MTP-Drafter und Prefix-Caching investieren.
- Median 4–8 k → nur **Variante A + Chroma-Mode 2** bauen, C, D
  und Autotune überspringen. Spart ~4 Wochen.
- Median > 12 k → volle A + C + Autotune-Stack wie in §9
  spezifiziert bauen; die Compound-Stack-Zahlen rechtfertigen es.
- Query-Kind dominant Extract-Verbatim / Numeric → Variante D
  wird First-Class-Anforderung, kein M-Hugin.3-Add-on.

### Phase 2 — Bandwidth-Microbench-Gate (2–3 Tage)

Vor jedem Adapter-Training oder C++-Integration den §6.3-Encoder-
Inner-Loop-Cost auf echten Xe-LPG-Kerneln reproduzieren. ~50-Zeilen
SPIR-V-Microbench, der die zwei dominanten Kernel-Shapes
(Cross-Attention-Read-out, Weight-Term-Matmul) auf realistischen
Tensor-Shapes vermisst.

**Entscheidungs-Gate:**

- Weight-Term nahe der vorhergesagten 750 ms → das §6-Wall-Clock-
  Modell trägt, weiter.
- Weight-Term deutlich höher (etwa > 2 s) → die Compound-Stack-
  Zahlen tragen nicht. Neu rechnen, bevor Wochen committed werden.
  Exakt der Guardrail, den
  `lesson_dp4a_autotune_prod_hazard` und
  `lesson_xelpg_dp4a_compute_ratio` für frühere Perf-Arbeit
  kodifiziert haben.

### Phase 3 — Munin zuerst (Voraussetzung)

Adapter-Iteration in Prod ohne Zero-Downtime-Reload ist nicht
tragbar — jeder Retrain wird zu einem 90-Sekunden-Outage. Munin
(siehe `research/munin-persistent-model-daemon`) ist vor Hugin
nicht optional; es ist Voraussetzung. Baue mindestens die
Hot-Reload-Option (das kleinere Option B im Munin-Vorschlag),
bevor M-Hugin.2 startet.

### Phase 4 — Hugin minimal (~4–5 Wochen)

Nur Variante A + Chroma-Mode 2 shippen:

- M-Hugin.1 unverändert (~1.5–2 W)
- Komprimierter M-Hugin.2: nur Variante A auf 26B-A4B, kein
  Autotune, keine Variante C, keine `adapters.{C,D}`-Slots im
  Config (~2 W)
- Komprimierter M-Hugin.4: Pre-Compressed-Mode + pegenaut-Chroma-
  Integration + Compound-Stack-A/B (~1.5 W)

Erwartete Wins auf dem geshippten Pfad: **~2–3× Total-Wall-Clock,
~28× TTFT auf 20-k-Kontexten**. Alle Zahlen aus §6.6 Zeilen 1–2.
Nicht die 18×-Headline-Zahl, aber 80 % des Werts bei 50 % des
Aufwands.

### Phase 5 — Autotune + Variante C/D (nur wenn Phase 4 es rechtfertigt)

Sobald Variante A einige Wochen in Prod läuft und Ledger-Einträge
reale Quality-vs-Speed-Daten auf pegenaut-Traffic zeigen:

- Quality okay + Traffic-Kontexte werden länger → Variante C
  (Multi-Stage) für sehr lange Docs zufügen.
- Quality-Gaps bei Extract-Verbatim / Numeric → Variante D +
  Query-Conditioned Autotune zufügen.
- Quality durchgängig okay → der Compound-Stack ist schon so gut
  wie er ohne Base-Model-Changes wird; C/D nicht zufügen.

**Geschätzter Aufwand in dieser Stufe:** 3–4 Wochen inkrementell,
gescoped nach dem was C/D in den Daten rechtfertigt.

### Kill-Kriterien für den ganzen M-Hugin-Track

- Phase 1 zeigt Median-Kontext ≤ 4 k Tokens → Hugin verwerfen,
  MTP und Prefix-Caching decken den Rest ab.
- Phase-2-Microbench zeigt Xe-LPG-Encoder-Cost > 3× der Prediction
  → Hugin verwerfen, Base-Model-Long-Context-Arbeit ist der
  bessere Hebel.
- Phase-4-A-only-Ledger zeigt < 1.5× Total-Wall-Clock-Gain → nicht
  auf C/D weiter. Sunk-Cost als Dokumentation dessen bergen, was
  auf dieser HW-Klasse nicht funktionierte, und stoppen.

Der Sinn dieser Reihenfolge ist, dass keine Phase ein verlorener
Schritt ist, auch wenn die nächste gedroppt wird: Phase 0 liefert
direkt user-sichtbaren Wert; Phasen 1–2 produzieren Messungen, die
du sowieso haben willst; Phase 3 (Munin) ist unabhängig auf der
Roadmap; Phase 4 ist wo du erstmals substanzielle Engineering-
Stunden committest, und erst nachdem drei billige Gates ja gesagt
haben.

---

## Referenzen

Anmerkung: URLs, wenn öffentlich verfügbar. Interne Synaipse-Notes
sind oben verlinkt.

- Jaegle, A. et al. (2021). *Perceiver: General Perception with
  Iterative Attention.* ICML.
- Jaegle, A. et al. (2022). *Perceiver IO: A General Architecture
  for Structured Inputs and Outputs.* ICLR.
- Li, J., Li, D., Savarese, S., Hoi, S. (2023). *BLIP-2:
  Bootstrapping Language-Image Pre-training with Frozen Image
  Encoders and Large Language Models.* ICML.
- Li, X.L., Liang, P. (2021). *Prefix-Tuning: Optimizing
  Continuous Prompts for Generation.* ACL.
- Lester, B., Al-Rfou, R., Constant, N. (2021). *The Power of
  Scale for Parameter-Efficient Prompt Tuning.* EMNLP.
- Mu, J., Li, X.L., Goodman, N. (2024). *Learning to Compress
  Prompts with Gist Tokens.* NeurIPS 2023.
- Ge, T. et al. (2024). *In-Context Autoencoder for Context
  Compression in a Large Language Model.* ICLR.
- Chevalier, A., Wettig, A., Ajith, A., Chen, D. (2023).
  *Adapting Language Models to Compress Contexts.* EMNLP.
- Zheng, L. et al. (2024). *SGLang: Efficient Execution of
  Structured Language Model Programs / RadixAttention.*
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer
  Learning with a Unified Text-to-Text Transformer (T5).* JMLR.
- Lewis, M. et al. (2020). *BART: Denoising Sequence-to-Sequence
  Pre-training for Natural Language Generation.* ACL.
- Google AI (2026-05-05). *Accelerating Gemma 4: MTP Drafters.*
  <https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/>
- Google AI (2026-06-10). *DiffusionGemma: 4× Faster Text
  Generation.*
  <https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/>

Interne mimirmind- / Synaipse-Referenzen (Zugang siehe `.mcp.json`):

- `Memory/mimirmind/research/munin-persistent-model-daemon.md`
- `Memory/mimirmind/research/roadmap-speculative-decoding.md`
- `Memory/mimirmind/research/mtp-diffusiongemma-status-2026-07.md`
- `Memory/mimirmind/research/perf-regression-ledger.md`
- `Memory/mimirmind/decisions/target-model-quant.md`
- `Memory/mimirmind/decisions/config-json-migration.md`
- `Memory/mimirmind/decisions/2026-07-06-command-list-replay.md`
- `Memory/mimirmind/decisions/gemm-prefill-rewrite.md`
- `Memory/mimirmind/lessons/lesson_xelpg_dispatch_overhead.md`
- `Memory/mimirmind/lessons/lesson_dp4a_autotune_prod_hazard.md`

---

## Anhang A — Notations-Übersicht

| Symbol         | Bedeutung                                                  |
|----------------|------------------------------------------------------------|
| $n$            | Länge der Input-Dokument-Token-Sequenz                     |
| $k$            | Länge der User-Turn-Token-Sequenz                          |
| $m$            | Anzahl der Hugin-Memory-Tokens (fixer Hyperparameter)      |
| $d$            | Base-Model-Residual-Stream-Dimension                       |
| $d_h$          | Hugin-Encoder-Innen-Dimension                              |
| $L$            | Anzahl der Base-Model-Transformer-Layer                    |
| $L_E$          | Anzahl der Hugin-Encoder-Blöcke                            |
| $\phi$         | Frozen Base-Model-Weights                                  |
| $\theta$       | Trainierbare Hugin-Encoder-Weights                         |
| $F_\phi$       | Base-Model-Forward-Pass                                    |
| $E_\theta$     | Hugin-Encoder-Forward-Pass                                 |
| $W_E$          | Base-Model-Token-Embedding-Matrix                          |
| $\mathbf M$    | Memory-Token-Embeddings, Shape $[m, d]$                    |
| $B$            | Sustained-Memory-Bandbreite, ≈ 55 GB/s auf Ziel-HW         |

## Anhang B — Etymologie

Odins zwei Raben, *Huginn* (Gedanke) und *Muninn* (Erinnerung),
erscheinen zusammen in *Grímnismál*, Strophe 20 der *Poetischen Edda*:

> Huginn ok Muninn fljúga hverjan dag
> Jörmungrund yfir;
> óumk ek of Hugin, at hann aftr né komi-t,
> þó sjámk meirr um Munin.

Odin bangt, Huginn komme nicht zurück — der Gedanke, der in die Welt
hinausfliegt, kann verloren gehen. Noch mehr bangt er um Muninn —
Erinnerung, schwerer zu ersetzen wenn sie einmal fort ist. Die
Namensgebung dieses Vorschlags spiegelt dieselbe Asymmetrie: Munin
(Persistenz) ist der unersetzbare, langlebige Zustand; Hugin
(Kompression) ist bei Verlust neu rechenbar.