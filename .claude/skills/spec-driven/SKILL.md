---
name: spec-driven
description: Create or advance a feature using spec-driven development (spec.md -> plan.md -> tasks.md -> code -> tests). Use when starting any new feature or changing a contract.
---
1. Pick the next number: `ls specs/` -> `specs/NNN-short-name/`.
2. `spec.md`: Why, scope, out-of-scope, acceptance criteria `AC-NNN.x` (testable, one behaviour each).
3. `plan.md`: modules touched, exact public signatures (the contract), data model changes, failure modes.
4. `tasks.md`: ordered checklist; each task references the AC it satisfies.
5. Ask `architect-reviewer` to review the plan before coding.
6. Implement (`developer`), then tests (`qa-engineer`) with AC ids in docstrings.
7. Tick tasks, update `AGENTS.md` if layout/commands changed.
