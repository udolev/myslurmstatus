# myslurmstatus

Interactive terminal dashboard for Slurm. One screen, live, refreshes every
second: which GPUs are free, who's using them, where your jobs are in the
queue, and what the per-partition limits are. CPU clusters too.

```
╔═══════════════════════════════════════════════════════════════════╗
║  GPU Dashboard [FAVORITES]              [user]  Tue 14:02:11      ║
║  16/24 GPUs free      You: 2 running (4 GPUs), 1 pending          ║
╚═══════════════════════════════════════════════════════════════════╝

┌─ dgx-b200-01  8x B200 ── [████░░░░] 4/8 used ─────────────────────┐
│  Available Partitions:  p_b200_goldberg*  B200-4h  B200-8h        │
├───────────────────────────────────────────────────────────────────┤
│  JOBID      PARTITION         NAME                  USER  GPUs    │
│  12417489   p_b200_goldberg   train_llama           alice    2    │
│  12417522   B200-4h           sweep_lr              bob      2    │
└───────────────────────────────────────────────────────────────────┘

 ↑↓:Nav  Enter:Details  /:Search  s:Settings  a:All  c:CPU  r:Refresh  Esc:Quit
```

## Features

- **Live dashboard** — running jobs and free-GPU counts refresh every second.
- **Pending queue inline** — Enter on any node shows the queue, your position
  in it, and ETAs.
- **Per-partition limits** — max time, max GPUs/user, max jobs/user, all
  pulled live from Slurm and shown next to each partition.
- **Only relevant partitions** — nodes/clusters with no partition available
  to your Slurm accounts are hidden. Account-restricted partitions you can use
  get a `*`; shared partitions stay unstarred.
- **GPU and CPU views** — toggle with `c`. CPU clusters are auto-grouped by
  shape (CPUs/RAM/partitions).
- **Pick your favorites** — press `s` to mark which nodes/clusters appear by
  default; `/` anywhere for incremental search.
- **No setup step** — your cluster's nodes, partitions, and per-QOS limits
  are auto-discovered at startup. Just run it.
- **Pure stdlib** — no `pip install`. Single file. Python 3.9+ and a Slurm
  client are all you need.

## Compatibility

**Works on any Slurm cluster.** Nothing in the script is BIU-specific —
everything is read live from `sinfo` / `scontrol` / `sacctmgr`, including
partition access via standard Slurm account/group metadata. There are no
hardcoded node lists, no per-cluster code paths, no required config file.
First run on any new cluster looks the same as the hundredth.

## Setup

```bash
git clone https://github.com/udolev/myslurmstatus.git
cd myslurmstatus
./setup.sh
```

The installer drops a single binary at `~/.local/bin/myslurmstatus` (override
the location with `PREFIX=/some/where ./setup.sh`). If `~/.local/bin` isn't on
your `$PATH`, the installer prints the line to add to your shell rc.

## Usage

```bash
myslurmstatus           # your favorites only (or all, if you set that pref)
myslurmstatus -a        # include extended (everything else) from start
```

Inside the dashboard:

| Key      | Action                            |
| -------- | --------------------------------- |
| `↑` `↓`  | navigate between nodes/clusters   |
| `Enter`  | open detail view (queue + limits) |
| `/`      | search — type to filter, Enter to keep selection, Esc to cancel |
| `s`      | settings panel (favorites + preferences) |
| `a`      | toggle extended nodes/clusters    |
| `c`      | switch GPU ⇄ CPU view             |
| `r`      | force refresh                     |
| `q`      | quit                              |
| `Esc`    | back one level (detail → dashboard → exit) |

Press `s` to open the settings panel — the panel's footer lists every key.
Saves to `~/.config/myslurmstatus/settings.json`.

## Configuration

One optional file at `~/.config/myslurmstatus/settings.json` — written by
the in-app settings panel. Hand-editable too:

```json
{
  "favorite_nodes":     ["dgx-b200-01", "mm-lab02"],
  "favorite_clusters":  ["dml02", "skittles"],
  "show_all_by_default": false,
  "sort_running_first":  true
}
```

Cluster names match the first node in each shape group as shown by the
dashboard. With no favorites set, the dashboard shows every discovered
node/cluster on first launch — pick yours via `s`, and from then on the
default view filters to favorites (toggle back to all with `a`).

### Environment variables

All optional:

| Variable                       | Default | Meaning                                  |
| ------------------------------ | ------- | ---------------------------------------- |
| `MYSLURMSTATUS_TIMEOUT`        | 60      | per-Slurm-command timeout (seconds)      |
| `MYSLURMSTATUS_REFRESH`        | 1       | dashboard refresh cadence (seconds)      |
| `MYSLURMSTATUS_QUEUE_REFRESH`  | 10      | pending-queue refresh cadence (seconds)  |
| `MYSLURMSTATUS_USER_REFRESH`   | 5       | user-summary refresh cadence (seconds)   |
| `MYSLURMSTATUS_MAX_QUEUE`      | 50      | max pending rows in detail view          |
| `MYSLURMSTATUS_SETTINGS`       | —       | override path to `settings.json`         |

## Requirements

- Python 3.9+
- A Slurm-client host with `squeue`, `sinfo`, `scontrol`, and `sacctmgr` on `$PATH`
- A terminal that supports curses (any modern Linux/macOS terminal)

## Slurm reference

Bundled BIU NLP cluster docs, useful if you're new to Slurm and want to know
how to actually submit jobs to the partitions the dashboard shows:

- `SLURM_GUIDE.md` — short markdown cheat-sheet (partitions, sample sbatch
  scripts, common pitfalls).
- `slurm-usage.pdf` — the long-form official PDF.

## Authors

Built by **Uriel Dolev** with **Claude Code** and **Codex**. Free to use and modify.
