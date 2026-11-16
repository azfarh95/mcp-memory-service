# Workflow divergence from upstream

This fork (`azfarh95/mcp-memory-service`) is a **private deployment fork** of
`doobidoo/mcp-memory-service`, not a contribution fork. Seven upstream-maintainer
workflows were deleted on 2026-11-16 because they serve upstream's public project and
fire, fail, or consume scheduled minutes on this fork without gating anything real:

| Deleted | Why |
|---|---|
| `claim-check.yml`, `claim-pending.yml` | Enforce upstream's "claim the issue before you start" convention for outside contributors |
| `merge-train.yml` | Upstream's auto-merge queue; needs `MERGE_TRAIN_CLIENT_ID` + a private key, and conflicts with this estate's own merge gate |
| `triage-digest.yml` | Daily digest of upstream's issue triage, commented on upstream's issues |
| `release.yml` | Publishes upstream releases from tags; this fork does not tag releases |
| `deploy-site.yml` | Deploys upstream's marketing site to mcpmemory.services |
| `cleanup-images.yml` | Weekly prune of upstream's GHCR images |

## Kept

`ci.yml`, `changelog.yml`, `links.yml`, `codeql.yml`, `site.yml` — these are real gates.
CI on this fork caught three defects that six rounds of cross-family review and a live
cutover rehearsal both missed, so the test/lint surface is worth its runtime.

## Merging upstream later

`git merge upstream/main` will try to re-add the seven deleted files. That is expected:
resolve by keeping them deleted (`git rm`). Nothing in the retained workflows depends on
the deleted ones — the only references left are prose in `CLAUDE.md`,
`.claude/commands/*.md`, `docs/IMAGE_RETENTION_POLICY.md` and comments in
`scripts/ci/mark_claim_pending.py`, all of which describe upstream's release process and
are harmless here.
