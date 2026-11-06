# 000 — Constitution (non-negotiable)

1. **Spec before code.** No feature without `spec.md` + acceptance criteria (AC-NNN.x).
2. **Contracts are law.** Public signatures in each `plan.md` are the integration contract between
   workstreams. Change them only by editing the plan in the same change.
3. **Structured LLM output.** The generator returns a Pydantic `RiskReport`; temperature 0.0;
   model `gpt-4.1-mini`; embeddings `text-embedding-3-small`; similarity = numpy cosine.
4. **Offline-first tests.** Unit tests never touch the network or real infra. Integration tests are
   marked `integration`; e2e tests `e2e`.
5. **Transactions.** All multi-statement MySQL writes run inside `transaction()`; failure => rollback.
6. **Graceful degradation.** Redis down -> L1 cache only. ES down -> report still generated from
   MySQL-stored chunks with an explicit `data_gaps` note. OpenAI key missing -> fake provider.
7. **Privacy.** Only synthetic/anonymized vendor data. Never log secrets or full document bodies.
8. **Errors.** Domain errors subclass `AppError(code, message, status_code, details)`.
9. **Tooling.** uv, ruff, pytest, Docker Compose. Python 3.13.
