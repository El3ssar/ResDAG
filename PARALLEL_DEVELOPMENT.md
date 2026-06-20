# Parallel Development Protocol

This repository's roadmap is a **dependency-ordered set of GitHub issues**, mirrored locally in the
private, gitignored **`ROADMAP.md`** (the living source of truth that planning sessions maintain).
The work is designed to be picked up by **multiple parallel Claude Code (or human) sessions**, each
taking one unblocked issue, doing it on its own git worktree + branch, and opening a PR into `main`.
This document is the contract those sessions follow.

> TL;DR for a new session: **clean merged worktrees → grab the highest-priority `status:ready`,
> unassigned issue → branch in a fresh worktree under `.claude/worktrees/` → implement with tests +
> docs → run checks → open a PR that closes the issue.** Then flip any newly-unblocked dependents to
> `status:ready`.

> ⚠️ **Releases are frozen until 1.0.** We are deliberately accumulating breaking changes toward a
> single major. Merging to `main` runs CI but **publishes nothing** (the `release` job is gated by the
> `RELEASE_FROZEN` repo variable — see `infra-freeze-releases`). Breaking changes are expected; add a
> cheap deprecation shim only when it's nearly free. Do **not** rename `.github/workflows/release.yml`.

---

## 1. The verbatim prompt to start a worker session

Paste this into a fresh session (in this repo) to have it pick up and complete the next available
piece of work autonomously:

```
Follow PARALLEL_DEVELOPMENT.md. Do ONE issue end to end:

1. Sync and clean: `git fetch --prune origin`, then remove any worktree/branch already merged into
   origin/main or gone on the remote (see §3). Never touch main or a worktree with uncommitted changes.
2. Pick work: from `gh issue list --state open --label status:ready` choose the UNASSIGNED issue with
   the highest priority (P0 > P1 > P2 > P3), breaking ties by lowest `wave`, then lowest issue number.
   Re-read its body and confirm every `Blocked by` issue is CLOSED. If none qualify, stop and report.
3. Claim it: `gh issue edit <n> --add-assignee @me` and comment that you are starting.
4. Branch + worktree: `git worktree add .claude/worktrees/<type>-<slug> -b <type>/<slug> origin/main`
   (type ∈ fix|feat|perf|refactor|docs|test|chore; <slug> is the ticket slug from the issue title /
   the deps footer). Work ONLY inside that worktree.
5. Implement the full "Proposed solution" and satisfy EVERY "Acceptance criteria" checkbox, including
   tests, docs, and __all__/public-API updates. Match house style (NumPy docstrings, black line
   length 100, ruff, mypy).
6. Verify: `uv run pytest -q`, `uv run ruff check src tests`, `uv run black --check src tests`,
   `uv run mypy src`. All must pass. Report any you skip and why.
7. PR: commit, push, and open a PR into `main` whose body contains `Closes #<n>` and a checklist
   mapping each acceptance criterion to the change that satisfies it.
8. Hand off: in a comment on the issue, list the issues this one BLOCKS (from its footer) so the next
   session can flip them to status:ready once this PR merges (see §6).

Do not start a second issue. Keep the change scoped to this one issue.
```

A new session can also be told simply: **"Follow PARALLEL_DEVELOPMENT.md and do the next ready
issue."**

---

## 2. How the work is organized

Every issue maps 1:1 to a **ticket slug** in `ROADMAP.md`. The ROADMAP carries the strategic framing
(five pillars, the wave plan, the critical path); the GitHub issues carry the actionable detail.

| Dimension | Encoding | Notes |
|---|---|---|
| **Pillar** | `pillar:*` label | `correctness`, `speed`, `api`, `pipeline`, `hpo`, `docs`, `infra` — which north-star goal the ticket advances. |
| **Epic / theme** | `type:epic` tracking issues | One per subsystem: foundation, core, reservoirs, readouts, init, transforms, models, data, speed, pipeline, hpo, docs-tests. Each lists its child issues. |
| **Area** | `area:*` label | foundation, core, readouts, reservoirs, init, transforms, training, models, hpo, data, ensemble, docs, tests. |
| **Kind** | `type:*` label | bug, feature, cleanup, refactor, perf, docs, test, chore. |
| **Priority** | `priority:P0..P3` | P0 = unblocker / verified high-severity bug. Reserve for the critical path. |
| **Effort** | `size:S..XL` | rough estimate. |
| **Readiness** | `status:ready` / `status:blocked` | `ready` = all `blocked_by` issues are closed; `blocked` = at least one is open. |

**Dependencies** live in each issue body as human-readable `Blocked by:` / `Blocks:` lines, mirrored
in a machine footer that references **ticket slugs** (stable across renumbering):

```
<!-- resdag-deps: blocked_by=<slugs|-> blocks=<slugs|-> wave=N epic=X pillar=Y -->
```

- `blocked_by` — tickets that must be **merged/closed before** this one starts (`-` if none).
- `blocks` — tickets this one unblocks (`-` if none).
- `wave` — longest dependency-path depth (wave 0 = no blockers). A scheduling hint; the source of
  truth is "are all `blocked_by` issues closed?".

The **critical path** (the handful of cross-cutting unblockers — e.g. `core-forward-stateless-refactor`,
`init-fast-spectral-radius`, `init-shared-scaling-contract`, `hpo-trial-runner-picklable-core`,
`models-shared-esn-builder`) is listed at the top of `ROADMAP.md`. Clear these early: they gate
disproportionately many tickets.

---

## 3. Cleaning merged worktrees (always step 1)

Worktrees live under `.claude/worktrees/` (git-excluded). Run this before grabbing new work so stale
worktrees from merged PRs don't pile up:

```bash
git fetch --prune origin

# Remove worktrees whose branch has been merged into origin/main or deleted on the remote.
# Skips the main checkout and anything with uncommitted changes.
git worktree list --porcelain | awk '/^worktree /{print $2}' | while read -r wt; do
  [ "$wt" = "$(git rev-parse --show-toplevel)" ] && continue
  case "$wt" in *"/.claude/worktrees/"*) : ;; *) continue ;; esac
  br=$(git -C "$wt" symbolic-ref --quiet --short HEAD 2>/dev/null) || continue
  [ -n "$(git -C "$wt" status --porcelain)" ] && continue   # uncommitted changes? leave it.
  if git branch --merged origin/main | grep -qx "  $br" \
     || ! git show-ref --verify --quiet "refs/remotes/origin/$br"; then
    echo "removing merged/gone worktree: $wt ($br)"
    git worktree remove "$wt"
    git branch -D "$br" 2>/dev/null || true
  fi
done

git worktree prune
git branch -vv | awk '/: gone]/{print $1}' | grep -vx main | xargs -r git branch -D
```

The `commit-commands:clean_gone` skill does the branch+worktree `[gone]` cleanup if you prefer a
one-shot command.

---

## 4. Picking the next issue

```bash
# All grabbable issues, highest priority first:
gh issue list --state open --label "status:ready" \
  --json number,title,assignees,labels \
  --jq 'map(select(.assignees|length==0))
        | sort_by( ( .labels|map(.name)|map(select(startswith("priority:")))[0] // "priority:P9" ) )
        | .[] | "\(.number)\t\(.title)"'
```

Take the top one. **Always re-read the issue body and confirm every `blocked_by` issue is CLOSED**
before starting — the `status:ready` label can lag reality.

---

## 5. Branch, implement, verify, PR

- **Worktree + branch:** `git worktree add .claude/worktrees/<type>-<slug> -b <type>/<slug> origin/main`.
  Branch types: `fix/`, `feat/`, `perf/`, `refactor/`, `docs/`, `test/`, `chore/`. `<slug>` is the
  ticket slug.
- **Scope:** do exactly one issue. If you discover adjacent work, file a follow-up issue rather than
  expanding the PR.
- **Checks (must pass):**
  ```bash
  uv run pytest -q
  uv run ruff check src tests
  uv run black --check src tests
  uv run mypy src
  ```
- **PR into `main`:** body must contain `Closes #<n>`, plus a checklist mapping each acceptance
  criterion to the change. The `commit-commands:commit-push-pr` skill automates commit → push → PR.
  Remember: merging publishes nothing (releases frozen) — it just lands the change on `main`.

---

## 6. Maintaining the dependency state (on merge)

When a PR merges and its issue #X closes, flip every issue it unblocked to `status:ready` **iff all
that issue's other blockers are now closed**. The just-closed issue's `blocks=` footer lists the
candidates (slugs); resolve each slug to its issue number via `gh issue list --search "<slug>"`:

```bash
X=<just-closed issue number>
slugs=$(gh issue view "$X" -R El3ssar/resdag --json body --jq .body \
        | grep -oP 'blocks=\K[^ ]+' | tr ',' ' ')
for slug in $slugs; do
  [ "$slug" = "-" ] && continue
  D=$(gh issue list -R El3ssar/resdag --state open --search "$slug in:body" --json number --jq '.[0].number')
  [ -z "$D" ] && continue
  blockers=$(gh issue view "$D" -R El3ssar/resdag --json body --jq .body \
             | grep -oP 'blocked_by=\K[^ ]+' | tr ',' ' ')
  still_open=0
  for b in $blockers; do
    [ "$b" = "-" ] && continue
    n=$(gh issue list -R El3ssar/resdag --search "$b in:body" --json number,state --jq '.[0]|select(.state=="OPEN")|.number')
    [ -n "$n" ] && still_open=1
  done
  if [ "$still_open" -eq 0 ]; then
    gh issue edit "$D" -R El3ssar/resdag --remove-label status:blocked --add-label status:ready
    echo "unblocked #$D ($slug)"
  fi
done
```

(Closing an issue does not auto-flip dependents — keep this honest so `status:ready` stays
trustworthy.)

---

## 7. Conventions recap

- Python ≥ 3.11, `src/` layout, NumPy-style docstrings, black line length **100**, ruff (E,F,I,N,W),
  mypy (`disallow_untyped_defs`). Tests mirror `src/` under `tests/`.
- Public-API changes update **both** the import and `__all__` in `src/resdag/__init__.py`.
- Breaking changes are expected pre-1.0; add a cheap deprecation shim only when it's nearly free.
- Version lives only in `src/resdag/__init__.py`. **Releases are frozen until 1.0** — never re-enable
  publishing or rename `release.yml` without an explicit 1.0 cut decision.
