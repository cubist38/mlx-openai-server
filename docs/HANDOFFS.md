# Handoffs – mlx-openai-server-lab

> Tier 3: Compute Fabric / Engine  
> Role: OpenAI-compatible MLX server powering the local-first workspace.

---

## Protocol

- This file is the **single source of truth** for what was done and what’s next.
- At the end of every session, the agent MUST:
  - Append a new `### Session YYYY-MM-DD • Agent` section.
  - Fill in **What I did**, **What’s next**, **Risks / blockers**.
- Never rewrite previous sessions, only append.

---

## Current Status (short human summary)

- Repo forked as `mlx-openai-server-lab`.
- Goal: become the Tier 3 Engine for:
  - Chat (`/v1/chat/completions`)
  - Embeddings (`/v1/embeddings`)
  - Vision / image / audio as needed
  - Future RAG endpoints (`/rag_query`, `/rag_upsert`, …) wrapping mlx-rag-lab.

---

## Open Questions

- [ ] Finalize default Tier 0 (embeddings) and Tier 1 (chat) models.
- [ ] Decide naming + paths for future `/rag_*` endpoints.
- [ ] Confirm how this server is called from gen-idea-lab (env vars, base URL).

---

## Sessions

### Session 000 – Bootstrap • HUMAN

**What I did**

- Forked upstream as `mlx-openai-server-lab`.
- Created `fusion/tier3-engine-phase0` branch.
- Added `docs/HANDOFFS.md` & `docs/FUSION_PHASE0.md` skeleton.

**What’s next (for the agent)**

- Run Phase 0 scan of the repo structure.
- Fill out `docs/FUSION_PHASE0.md` with:
  - High-level map of main modules / entrypoints.
  - Where model loading & routing logic live.
  - List of config/env vars related to models & concurrency.

**Risks / blockers**

- None yet; design choices pending about default models and memory budget.
