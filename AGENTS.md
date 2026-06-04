# Working notes for Claude

This is a public-facing tool. Two non-negotiable lenses for every change.

## Code

- **Simple**: prefer single small file, stdlib only, no new deps. Reuse what's already in the file before adding helpers.
- **Reliable**: graceful fallbacks (missing config → defaults; failed Slurm call → previous data). Never crash on an empty/odd line.
- **Efficient**: render runs every second — don't add work to the hot path. Subprocess fan-out goes through `parallel_results`.
- **General**: cluster-agnostic. Nothing in the script is BIU-specific; topology and access come from standard `squeue` / `sinfo` / `scontrol` / `sacctmgr` at startup.

## UX

- **Easy setup**: `./setup.sh` and you're done. Topology is auto-discovered at startup; no required config.
- **Easy use**: keys are discoverable — single letter, surfaced in the footer.
- **Functional**: the answer to "what's free, where am I in the queue, what are the limits" should be one glance away.
- **No bloat**: every key, every config field has to earn its place. If it can be defaulted, default it.
- **Good design, intuitive**: names that read like English (`favorites`, not "core"). Same key (`/`) means search everywhere. Same key (`Esc`) means "back" everywhere.

## Layout

- `myslurmstatus.py` — the dashboard (curses TUI, single file). Topology is auto-discovered at startup via `sinfo`; nothing is hardcoded.
- `setup.sh` — installs the script into `~/.local/bin/` (or `$PREFIX/bin`).
- `README.md` — concise, code-fence examples, simple tables. Match the existing tone.
- `SLURM_GUIDE.md` / `slurm-usage.pdf` — bundled BIU reference docs.

## Startup contract

`main()` runs three Slurm calls in parallel via `parallel_results`:
- `discover_topology()` — `sinfo --all -h -N -o ...` (`--all` is required to include hidden partitions) parsed into NodeConfig/CpuClusterConfig.
- `fetch_user_access()` — `sacctmgr` + local Unix groups → the user's Slurm accounts, groups, and explicit partition associations.
- `fetch_partition_info()` — `scontrol`+`sacctmgr` → partition `AllowAccounts`/`AllowGroups`, max time, GPUs/user, jobs/user.

Latency floor = max of the three. Don't add new serial Slurm calls there.

## Single user file (optional)

`~/.config/myslurmstatus/settings.json`:
```json
{
  "favorite_nodes":     [...],
  "favorite_clusters":  [...],
  "show_all_by_default": false,
  "sort_running_first":  true
}
```
Written by the in-app `s` panel. Missing = no favorites, defaults apply.

## When changing things

1. Read this file first.
2. If you're adding state, ask whether it can be derived instead.
3. If you're adding a key, check it doesn't collide and add it to the footer + README key table.
4. After code changes: `python3 -m py_compile myslurmstatus.py` then `myslurmstatus --help` to spot regressions.
5. Update `README.md` *and* the in-script docstring/argparse if user-visible behavior changes.
