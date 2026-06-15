# Parallel Development Protocol

This repository's roadmap lives as a **dependency-ordered set of GitHub issues**. The work is
designed to be picked up by **multiple parallel Claude Code (or human) sessions**, each taking one
unblocked issue, doing it on its own git worktree + branch, and opening a PR. This document is the
contract those sessions follow.

> TL;DR for a new session: **clean merged worktrees → grab the highest-priority `status:ready`,
> unassigned issue → branch in a fresh worktree → implement with tests + docs → run checks → open a
> PR that closes the issue.** Then flip any newly-unblocked dependents to `status:ready`.

---

## 1. The verbatim prompt to start a worker session

Paste this into a fresh session (in this repo) to have it pick up and complete the next available
piece of work autonomously:

```
Follow PARALLEL_DEVELOPMENT.md. Do ONE issue end to end:

1. Sync and clean: `git fetch --prune origin`, then remove any worktree/branch whose branch is
   already merged into origin/main or is gone on the remote (see §3). Never touch main or a
   worktree with uncommitted changes.
2. Pick work: from `gh issue list --state open --label status:ready` choose the UNASSIGNED issue
   with the highest priority (P0 > P1 > P2 > P3), breaking ties by lowest `wave`, then lowest
   issue number. Re-read its body and confirm every `Blocked by` issue is CLOSED. If none qualify,
   stop and report.
3. Claim it: assign it to me (`gh issue edit <n> --add-assignee @me`) and comment that you are
   starting.
4. Branch + worktree: `git worktree add ../resdag-wt/<n>-<slug> -b <type>/<n>-<slug> origin/main`
   (type ∈ fix|feat|perf|refactor|docs|test|chore). Work ONLY inside that worktree.
5. Implement the full "Proposed solution" and satisfy EVERY "Acceptance criteria" checkbox,
   including tests, docs, and __all__/public-API updates. Match house style (NumPy docstrings,
   black line length 100, ruff, mypy).
6. Verify: `uv run pytest -q` (or `pytest`), `uv run ruff check src tests`,
   `uv run black --check src tests`, `uv run mypy src`. All must pass. Report any you skip and why.
7. PR: commit, push, and open a PR whose body contains `Closes #<n>` and a short summary mapping
   each acceptance-criteria checkbox to the change that satisfies it.
8. Hand off: in a comment on the issue, list the issues this one BLOCKS (from its body) so the next
   session can flip them to status:ready once this PR merges (see §6).

Do not start a second issue. Keep the change scoped to this one issue.
```

A new session can also be told simply: **"Follow PARALLEL_DEVELOPMENT.md and do the next ready
issue."**

---

## 2. How issues are organized

| Dimension | Encoding | Notes |
|---|---|---|
| **Epic / theme** | `type:epic` tracking issues | One per epic (foundation, core-correctness, readouts, reservoirs, init, transforms, training, models, hpo, data-ergonomics, docs-tests). Each lists its child issues and a Mermaid DAG. |
| **Area** | `area:*` label | foundation, core, readouts, reservoirs, init, transforms, training, models, hpo, data, docs, tests. |
| **Kind** | `type:*` label | bug, feature, cleanup, refactor, perf, docs, test, chore. |
| **Priority** | `priority:P0..P3` | P0 = unblocker / verified blocker; reserve for foundation + verified high-severity bugs. |
| **Effort** | `size:S..XL` | rough estimate. |
| **Readiness** | `status:ready` / `status:blocked` | `ready` = all dependency issues are closed; `blocked` = at least one open blocker. |

**Dependencies** are encoded in each issue body as human-readable `**Blocked by:** #a, #b` /
`**Blocks:** #m, #n` lines, mirrored in a machine-readable footer:

```
<!-- resdag-deps: blocked_by=#12,#15 blocks=#40,#41 wave=2 epic=readouts -->
```

- `blocked_by` — issues that must be **merged/closed before** this one starts (`-` if none).
- `blocks` — issues this one unblocks (`-` if none); present only on issues that block something.
- `wave` — longest dependency-path depth (wave 0 = no blockers). A scheduling hint, not a hard gate;
  the source of truth is "are all `blocked_by` issues closed?".

The current dependency graph is rendered as Mermaid diagrams in the **master roadmap issue
[#109](https://github.com/El3ssar/ResDAG/issues/109)** (epic-level graph + "start here" list) and
per-epic in each `type:epic` tracking issue (#98–#108).

---

## 3. Cleaning merged worktrees (always step 1)

Run before grabbing new work so stale worktrees from merged PRs don't pile up:

```bash
git fetch --prune origin

# Remove worktrees whose branch has been merged into origin/main or deleted on the remote.
# Skips the main checkout and anything with uncommitted changes.
git worktree list --porcelain | awk '/^worktree /{print $2}' | while read -r wt; do
  [ "$wt" = "$(git rev-parse --show-toplevel)" ] && continue
  br=$(git -C "$wt" symbolic-ref --quiet --short HEAD 2>/dev/null) || continue
  # uncommitted changes? leave it alone
  [ -n "$(git -C "$wt" status --porcelain)" ] && continue
  if git branch --merged origin/main | grep -qx "  $br" \
     || ! git show-ref --verify --quiet "refs/remotes/origin/$br"; then
    echo "removing merged/gone worktree: $wt ($br)"
    git worktree remove "$wt"
    git branch -D "$br" 2>/dev/null || true
  fi
done

git worktree prune
# Prune local branches whose upstream is gone:
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

- **Worktree + branch:** `git worktree add ../resdag-wt/<n>-<slug> -b <type>/<n>-<slug> origin/main`.
  Branch types: `fix/`, `feat/`, `perf/`, `refactor/`, `docs/`, `test/`, `chore/`.
- **Scope:** do exactly one issue. If you discover adjacent work, file a follow-up issue rather than
  expanding the PR.
- **Checks (must pass):**
  ```bash
  uv run pytest -q
  uv run ruff check src tests
  uv run black --check src tests
  uv run mypy src
  ```
- **PR:** body must contain `Closes #<n>`, plus a checklist mapping each acceptance criterion to the
  change. The `commit-commands:commit-push-pr` skill automates commit → push → PR.

---

## 6. Maintaining the dependency state (on merge)

When a PR merges and its issue #X closes, flip every issue it unblocked to `status:ready` **iff all
that issue's other blockers are now closed**. The just-closed issue's `blocks=` footer lists the
candidates:

```bash
X=<just-closed issue number>
# Candidates this issue unblocks (from its blocks= footer):
cands=$(gh issue view "$X" -R El3ssar/ResDAG --json body --jq .body \
        | grep -oP 'blocks=\K[#0-9,-]+' | tr ',#' ' ')
for D in $cands; do
  blockers=$(gh issue view "$D" -R El3ssar/ResDAG --json body --jq .body \
             | grep -oP 'blocked_by=\K[#0-9,-]+' | tr ',#' ' ')
  still_open=0
  for b in $blockers; do
    [ "$(gh issue view "$b" -R El3ssar/ResDAG --json state --jq .state)" = "OPEN" ] && still_open=1
  done
  if [ "$still_open" -eq 0 ]; then
    gh issue edit "$D" -R El3ssar/ResDAG --remove-label status:blocked --add-label status:ready
    echo "unblocked #$D"
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
- Breaking changes are acceptable pre-1.0 when they improve adoption; add a cheap deprecation shim
  when it's not free.
- Version lives only in `src/resdag/__init__.py`.
