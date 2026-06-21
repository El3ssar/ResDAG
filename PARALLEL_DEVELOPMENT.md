# Parallel Development Protocol

How sessions pick up and ship work on `resdag`. This is the **process** doc. The two other layers:

- **The live GitHub issues** (`gh issue list -R El3ssar/resdag --state open`) are the **per-ticket
  source of truth** — `status:ready`, priorities, each issue's `## Files`, and its dependency footer.
  Always the authority; the labels can lag, so re-check the body.
- **`ROADMAP.md`** (private, git-excluded) is the **strategy** layer — north star, pillars, and the
  shape of what's left. Read it for *what matters*; read the issues for *what to do*.

Work is done by **parallel sessions**, each taking one unblocked issue on its own git worktree +
branch, opening a PR into `main`.

> ⚠️ **Releases are frozen until 1.0.** Merging to `main` runs CI but **publishes nothing** (the
> `release` job is gated by the `RELEASE_FROZEN` repo variable). Breaking changes are expected; add a
> cheap deprecation shim only when it's nearly free. **Never rename `.github/workflows/release.yml`.**

---

## 1. The prompts

There are three ways to run work. Pick by the issue (see the **single-session list** below).

### (a) Single-issue worker — the default

Paste into a fresh session to autonomously ship the next ready issue:

```
Follow PARALLEL_DEVELOPMENT.md. Do ONE issue end to end:

1. Sync & clean: `git fetch --prune origin`; remove any worktree/branch already merged into
   origin/main or gone on the remote (see §3). Never touch main or a worktree with uncommitted changes.
2. Pick: from `gh issue list --state open --label status:ready` choose the UNASSIGNED issue with the
   highest priority (P0>P1>P2>P3), ties broken by lowest issue number. Re-read its body and confirm
   every `Blocked by` issue is CLOSED. If none qualify, stop and report.
3. Claim: `gh issue edit <n> --add-assignee @me` and comment that you are starting.
4. Worktree: `git worktree add .claude/worktrees/<type>-<slug> -b <type>/<slug> origin/main`
   (type ∈ fix|feat|perf|refactor|docs|test|chore; <slug> from the issue title / deps footer).
   Work ONLY inside that worktree.
5. Implement the full solution + EVERY acceptance criterion: tests, docs, and public-API/__all__
   updates. THE TRIAD: any new public symbol needs __all__ + a docs page + a runnable example + a
   test (see §5). House style: NumPy docstrings, black 100, ruff, mypy.
6. Verify (§5): pytest (affected), ruff, black --check, mypy. Report anything skipped and why.
7. PR into main: body has `Closes #<n>` + a checklist mapping each acceptance criterion to a change.
   DO NOT merge.
8. Comment on the issue listing the issues it BLOCKS (from its footer) so they can be flipped on merge.

Do not start a second issue. Keep scope to this one issue.
```

### (b) Conductor — dispatch a parallel batch

Paste into a session to fan out several background workers at once:

```
You are the conductor for the resdag roadmap. Read ROADMAP.md (strategy) and this file.

1. Sync: git fetch --prune origin; clean merged worktrees (§3);
   run `python .agent-planning/roadmap_build/maintenance.py --execute` to flip newly-unblocked issues.
2. Pick the next 5 highest-priority UNASSIGNED status:ready issues whose `## Files` DO NOT OVERLAP.
   EXCLUDE issues labelled `needs:dedicated-session` (those are off-limits to batch dispatch):
     gh issue list -R El3ssar/resdag --search \
       'is:open is:issue label:status:ready -label:needs:dedicated-session no:assignee'
   Read `## Files` from each ISSUE BODY (`gh issue view <n>`) — NOT from deploy_plan.json (it doesn't
   know the newest tickets). Never batch two issues touching the same file; the frequent collisions are
   src/resdag/__init__.py, layers/cells/esn_cell.py, layers/reservoirs/base_reservoir.py,
   core/model.py, hpo/run.py — at most ONE issue per shared file per batch.
   If you find fewer than 5 truly non-overlapping issues, dispatch fewer — never batch a collision.
3. Pre-create each worktree sequentially (avoids git-lock races), claim each issue, then launch one
   Opus BACKGROUND worker per issue with prompt (a), each in its own .claude/worktrees/<type>-<slug>.
4. Report each PR link + per-check result. DO NOT merge — leave them for the human.
```

### (c) Single dedicated session — for the hard issues

Some issues must NOT be fire-and-forget background workers. Use this prompt interactively:

```
Do issue #<n> end to end in THIS session, following PARALLEL_DEVELOPMENT.md. It needs a dedicated
session because it is <design-heavy / hardware-dependent / touches CI / defines public API>.

- Worktree .claude/worktrees/<type>-<slug> off origin/main.
- BEFORE coding: show me the design — the public surface (new symbols + signatures), how it composes
  with the existing API, and what could break usability. Wait for my OK.
- Implement the full acceptance criteria. THE TRIAD (§5) is mandatory for any public symbol.
- Run the four checks; if it needs the GPU/benchmarks, run them here (this machine has a GPU).
- PR into main with `Closes #<n>`; DO NOT merge — walk me through it first.
```

**Single-session list — do NOT background-batch these.** Every such issue carries the
**`needs:dedicated-session`** label, so the conductor query in §1b/§4 auto-excludes them. Categories:

| Category | Why | Examples |
|---|---|---|
| New public-API capabilities | A loose agent inventing API → rot; surface needs review first | #395 IP, #396 DeepESN, #397 classification, #399 hybrid, #400 metrics, #398/#268 sklearn, #265 BPTT, #264 diff-forecast, #270 export, #376/#377 online/FORCE |
| Hardware / perf | A headless worker can't measure speed (we learned this with #389) | #260 sparse, #261 vmap, #262 HPO-build, #263 bench-CI |
| CI / `.github/workflows` edits | Automating CI is risky; human-gated | #263, #299, #402, #303 |
| External env needed | Worker lacks Julia / multi-node / multi-GPU | #279 distributed HPO, RC.jl benchmarks |
| Trackers / umbrellas | Not shippable as one PR | #177, #404, the epics #117–#128 |
| Keystone shared-file refactors | Conflict with everything; gate many tickets | heavy edits to base_reservoir.py / model.py / esn_cell.py |

Everything else — small bugs, cleanups, docs, isolated tests in **disjoint files** — is ideal batch
fodder for prompt (b).

---

## 2. How the work is organized

Each issue carries the actionable detail; `ROADMAP.md` carries the strategy. Labels:

| Dimension | Encoding |
|---|---|
| **Pillar** | `pillar:*` — correctness, speed, api, pipeline, hpo, docs, infra |
| **Area** | `area:*` — foundation, core, readouts, reservoirs, init, transforms, training, models, hpo, data, ensemble, docs, tests |
| **Kind** | `type:*` — bug, feature, cleanup, refactor, perf, docs, test, chore |
| **Priority** | `priority:P0..P3` (P0 = unblocker / verified high-severity) |
| **Effort** | `size:S..XL` |
| **Readiness** | `status:ready` (all blockers closed) / `status:blocked` |

**Dependencies + the files a ticket touches** live in the issue body: human-readable `## Files`,
`Blocked by:` / `Blocks:`, plus a machine footer (ticket slugs are stable across renumbering):

```
<!-- resdag-deps: blocked_by=<slugs|-> blocks=<slugs|-> wave=N epic=X pillar=Y -->
```

`blocked_by` must all be **closed** before the ticket starts. `## Files` is what the conductor uses
for the non-overlap check — read it from the live issue body.

---

## 3. Cleaning merged worktrees (always step 1)

Worktrees live under `.claude/worktrees/` (git-excluded). Note: squash-merged branches are NOT
ancestors of `origin/main`, so confirm a worktree's PR is **merged** (`gh pr list --head <branch>
--state all`) before removing — the snippet below catches remote-deleted branches; check the rest by PR.

```bash
git fetch --prune origin
git worktree list --porcelain | awk '/^worktree /{print $2}' | while read -r wt; do
  [ "$wt" = "$(git rev-parse --show-toplevel)" ] && continue
  case "$wt" in *"/.claude/worktrees/"*) : ;; *) continue ;; esac
  br=$(git -C "$wt" symbolic-ref --quiet --short HEAD 2>/dev/null) || continue
  [ -n "$(git -C "$wt" status --porcelain)" ] && continue          # uncommitted? leave it.
  if git branch --merged origin/main | grep -qx "  $br" \
     || ! git show-ref --verify --quiet "refs/remotes/origin/$br"; then
    echo "removing merged/gone worktree: $wt ($br)"; git worktree remove "$wt"; git branch -D "$br" 2>/dev/null || true
  fi
done
git worktree prune
git branch -vv | awk '/: gone]/{print $1}' | grep -vx main | xargs -r git branch -D
```

The `commit-commands:clean_gone` skill does the `[gone]` cleanup as a one-shot.

---

## 4. Picking the next issue

```bash
# Batchable ready issues (excludes those needing a dedicated session), highest priority first:
gh issue list -R El3ssar/resdag --limit 400 --json number,title,assignees,labels \
  --search 'is:open is:issue label:status:ready -label:needs:dedicated-session no:assignee' \
  --jq 'sort_by( (.labels|map(.name)|map(select(startswith("priority:")))[0] // "priority:P9") )
        | .[] | "\(.number)\t\(.title)"'
# (Drop the `-label:needs:dedicated-session` filter to also see dedicated-session issues.)
```

Take the top one (or, for a conductor, the top non-overlapping batch). **Always re-read the body and
confirm every `blocked_by` issue is CLOSED** — `status:ready` can lag reality.

---

## 5. Branch, implement, verify, PR

- **Worktree + branch:** `git worktree add .claude/worktrees/<type>-<slug> -b <type>/<slug> origin/main`
  (`<type>` ∈ fix|feat|perf|refactor|docs|test|chore).
- **Scope:** exactly one issue. Adjacent work → file a follow-up issue, don't expand the PR.
- **THE TRIAD (mandatory for any public-API change):** a new public symbol lands with
  **`__all__` update + a docs page + a runnable example + a test**. This is the anti-rot rule — no
  exceptions. Public API also needs maintainer sign-off (run it as a dedicated session, §1c).
- **Checks (must pass):** use the affected-tests selector, not the whole suite (see CLAUDE.md):
  ```bash
  uv run python tools/affected_tests.py --explain
  uv run pytest --no-cov -q $(uv run python tools/affected_tests.py --format args)
  uv run ruff check src tests
  uv run black --check src tests
  uv run mypy src            # bar: no NEW errors on files you touched
  ```
- **PR into `main`:** body has `Closes #<n>` + a checklist mapping each acceptance criterion to a
  change. Merging publishes nothing (releases frozen). The `commit-commands:commit-push-pr` skill
  automates commit → push → PR.

---

## 6. Maintaining the dependency state (on merge)

Closing an issue does **not** auto-flip dependents. After a batch merges, run:

```bash
python .agent-planning/roadmap_build/maintenance.py --execute
```

It flips every `status:blocked` issue to `status:ready` once all its `blocked_by` are closed. If a
blocker was closed as *not planned* (won't-fix), edit the dependent's footer to drop that dead slug
so readiness reflects reality. Keep `status:ready` trustworthy.

---

## 7. Conventions recap

- Python ≥ 3.11, `src/` layout, NumPy docstrings, black line length **100**, ruff (E,F,I,N,W), mypy
  (`disallow_untyped_defs`). Tests mirror `src/` under `tests/`.
- Public-API changes update **both** the import and `__all__` in `src/resdag/__init__.py` — plus the
  triad (§5).
- Breaking changes are expected pre-1.0; cheap deprecation shim only when nearly free.
- Version lives only in `src/resdag/__init__.py`. **Releases frozen until 1.0** — never re-enable
  publishing or rename `release.yml`.
