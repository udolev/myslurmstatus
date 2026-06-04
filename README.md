# myslurmstatus

Interactive terminal dashboard for Slurm.

It shows available GPU/CPU resources, running jobs, pending jobs, queue details,
and partition limits using standard Slurm commands. It auto-discovers the
cluster at startup and only shows nodes and partitions available to your Slurm
accounts/groups.

```text
GPU Dashboard [FAVORITES]                         2026-06-04 14:02
16/24 GPUs free                         You: 2 GPU jobs (4 GPUs)

dgx-b200-02  8x B200  [████░░░░] 4/8 used
Available Partitions: p_b200_schwartz*, B200-4h

JOBID      PARTITION         NAME                  USER        GP  TIME
12417489   p_b200_schwartz   train_llama           you          2  01:12:31
12417522   B200-4h           sweep_lr              alice        2  00:18:44
```

## Install

```bash
git clone https://github.com/udolev/myslurmstatus.git
cd myslurmstatus
./setup.sh
```

This installs `myslurmstatus` to `~/.local/bin` by default.

To install somewhere else:

```bash
PREFIX=/path/to/prefix ./setup.sh
```

## Run

```bash
myslurmstatus
```

Show all available nodes/clusters from startup:

```bash
myslurmstatus --all
```

## Keys

| Key | Action |
| --- | --- |
| `Up` / `Down` | Move between nodes/clusters |
| `Enter` | Open details, queue, and limits |
| `/` | Search |
| `s` | Choose favorites and preferences |
| `a` | Toggle favorites/all |
| `c` | Switch GPU/CPU view |
| `r` | Refresh |
| `q` | Quit |
| `Esc` | Back, then quit |

## What It Shows

- GPU and CPU utilization
- Running jobs on each node/cluster
- Pending queue and your position in it
- Partition limits, including time and per-user limits
- Account/group-restricted partitions you can use, marked with `*`
- Shared partitions, shown without `*`

## Requirements

- Python 3.9+
- Slurm client commands on `PATH`: `squeue`, `sinfo`, `scontrol`, `sacctmgr`
- A terminal with curses support

No Python packages are required.

## Configuration

Settings are optional and are written by the in-app settings screen:

```text
~/.config/myslurmstatus/settings.json
```

Supported environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MYSLURMSTATUS_TIMEOUT` | `60` | Slurm command timeout in seconds |
| `MYSLURMSTATUS_REFRESH` | `1` | Dashboard refresh interval |
| `MYSLURMSTATUS_QUEUE_REFRESH` | `10` | Pending queue refresh interval |
| `MYSLURMSTATUS_USER_REFRESH` | `5` | User summary refresh interval |
| `MYSLURMSTATUS_MAX_QUEUE` | `50` | Max pending rows in detail view |
| `MYSLURMSTATUS_SETTINGS` | unset | Override settings file path |

## Included Slurm Docs

This repo also includes BIU Slurm reference material:

- `SLURM_GUIDE.md`
- `slurm-usage.pdf`

## License

No license file is included yet.
