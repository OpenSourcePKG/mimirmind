## Project

**MimirMind** is a standalone C++ inference engine for GGUF language models,
targeting the integrated Intel Arc GPU on Meteor Lake via oneAPI Level Zero
with Unified Shared Memory. Loads pre-trained models; **does not train**.

Sister project: `pegenaut` (TypeScript RAG stack, separate repo). MimirMind
is intended to eventually serve as a drop-in OpenAI-compatible inference
backend for Pegenaut, once Phase **Mimir-1.0** lands.

## Codebase Conventions

- **Language:** C++20.
- **Build:** CMake ≥ 3.21, `cmake -B build && cmake --build build`.
- **Layout:** one class per file, `PascalCase` filenames, headers `.hpp`,
  sources `.cpp`. Namespaces follow folders (`mimirmind::runtime`, etc.).
- **Style:** 4-space indent, K&R braces, no tabs. Explicit `public`/
  `private` access. RAII over manual cleanup. `nullptr` / `enum class` /
  `std::span` over legacy idioms. No exceptions thrown across the
  Level-Zero boundary — return errors or use `std::expected` where it
  helps.
- **English only** in source, comments, log lines, error messages.
- **No `llama.cpp` / `ggml` as a runtime dependency.** The whole point of
  this project is to implement it ourselves. `llama-cli` is used during
  development only as a *reference oracle* for tensor / logits / token
  parity.

## Roadmap Anchor (Norse-Themed)

Always frame milestones with the phase they belong to:

- **Project Well** *(Alpha)* — read from disk into USM. M1-M4.
- **Project Envoy** *(Beta)* — GPU kernels, the head learns to speak. M5-M6.
- **Mimir-1.0** *(Release)* — OpenAI-compatible HTTP, Gemma 4 architecture.
  M7-M9.

Full milestone table lives in `README.md` and in the Synaipse note
`Memory/mimirmind/roadmap`.

## Target Model

**Gemma 4** is the strategic target. Gemma 3 (4B, 12B) is the *verification
baseline* — used for tensor parity tests against `llama.cpp/llama-cli`
before Gemma 4 work begins. Do **not** silently swap models without saying
which phase you are in.

## Available MCP Servers

### synaipse

Synaipse is the persistent long-term memory system for this project.

Synaipse contains:

- Architecture decisions
- Project knowledge
- Technical documentation
- Coding standards
- Known issues and solutions
- Research notes
- TODOs
- Lessons learned
- API knowledge
- Development history

### How Claude reaches Synaipse

Synaipse is accessed **exclusively through MCP tools** exposed by the
`synaipse` server in `.mcp.json`. Always use the tool — never touch the
synaipse repository or its vault on disk. The MCP layer handles project
scoping (the `X-Synaipse-Project: mimirmind` header pins every write
into `Memory/mimirmind/`), frontmatter, link rewriting, and git
autocommit. Direct file I/O bypasses all of that and corrupts the index.

**Cross-project caveat:** the header is enforced server-side. Writes from
a mimirmind-scoped session always land in `Memory/mimirmind/`, regardless
of the `path` parameter. If you need to update a pegenaut note, switch
to a pegenaut-scoped session.

Tool mapping for common user phrases:

- "search synaipse for X" / "was wissen wir über X" → `synaipse_search`
- "verwandte notes zu X" → `synaipse_related`, `synaipse_backlinks`
- "dokumentiere in synaipse" / "save to memory" → `synaipse_write_note`
- "update die note über X" → `synaipse_read_note` then `synaipse_update_note`
- "was steht an" / "recent" / "stale" → `synaipse_todos`, `synaipse_recent`, `synaipse_stale`
- "gib mir den projekt-kontext" / cold-start of a session → `synaipse_prime`

Do NOT:

- Open, read, or edit files under the synaipse codebase
- Read or grep the vault directory directly
- Reimplement search by walking the vault

### Memory First Policy

For every non-trivial task, Claude must follow this workflow:

SEARCH MEMORY
→ ANALYZE
→ IMPLEMENT
→ STORE KNOWLEDGE

Before starting work:

1. Search Synaipse for relevant knowledge.
2. Check existing architecture decisions.
3. Check known solutions.
4. Check known issues and workarounds.
5. Review related project documentation.

After completing work:

1. Store newly discovered knowledge.
2. Store important implementation details.
3. Store architecture decisions.
4. Store lessons learned.
5. Update outdated information.
6. Link related knowledge entries.

Knowledge stored in Synaipse takes precedence over assumptions.

If required information cannot be found:

1. Identify the knowledge gap.
2. Continue with best effort.
3. Suggest creating a new memory entry.

### Knowledge Categories

When storing information, classify it into one of the following categories:

- architecture
- decisions
- implementation
- bugs
- solutions
- infrastructure
- development
- documentation
- research
- api
- standards
- todos

### Architecture Decision Records

Important technical decisions must be documented.

Store:

- Problem
- Context
- Alternatives considered
- Final decision
- Consequences

### Lessons Learned

When solving a difficult problem, store:

- Root cause
- Investigation process
- Final solution
- Future recommendations

### Code Reuse

Before generating new implementations:

- Search for existing patterns.
- Search for similar implementations.
- Follow established project conventions.

Avoid creating duplicate solutions when an existing pattern already exists.