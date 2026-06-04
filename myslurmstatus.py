#!/usr/bin/env python3
"""myslurmstatus — Interactive GPU/CPU dashboard for Slurm (curses TUI).

Shows live utilisation, your running and pending jobs, and per-partition
limits for GPU nodes and CPU clusters available to your Slurm accounts.
Auto-discovers the cluster topology at startup; nothing is hardcoded.

Keys: ↑↓ navigate, Enter for detail view, / to search, s for settings,
a to toggle extended, c to switch GPU/CPU, r to refresh, q to quit
(Esc backs out one level — detail → dashboard → exit).

Requirements:
    Python 3.9+, a working terminal (curses), and the Slurm client commands
    `squeue`, `sinfo`, `scontrol`, `sacctmgr` on $PATH.

User settings:
    ~/.config/myslurmstatus/settings.json — written by the in-app settings
    panel ('s' shortcut). Holds favorite nodes/clusters and a couple of
    preferences. Hand-editable; missing file = no favorites, no override.

Environment variables (all optional):
    MYSLURMSTATUS_TIMEOUT         per-Slurm-command timeout in seconds (default 60)
    MYSLURMSTATUS_REFRESH         dashboard refresh cadence in seconds (default 1)
    MYSLURMSTATUS_QUEUE_REFRESH   pending-queue refresh cadence (default 10)
    MYSLURMSTATUS_USER_REFRESH    user-summary refresh cadence (default 5)
    MYSLURMSTATUS_MAX_QUEUE       max pending rows shown in detail view (default 50)
    MYSLURMSTATUS_SETTINGS        override path to settings.json
"""

import argparse
import curses
import getpass
import grp
import json
import os
import re
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence

MAX_JOBS_IN_BOX = 8

# Column widths for the running-job table. Sized so the longest typical Slurm
# value sits ~3 chars from the next column (e.g. "p_b200_goldberg" + 3 = 18).
# Shorter values get more padding by virtue of alignment.
JOBID_W = 10
PART_W = 18
NAME_W = 26
USER_W = 14
GP_W = 3
CPU_W = 5
TIME_W = 10

# Pending-queue table widths (detail view only).
QUEUE_RANK_W = 3
QUEUE_PRI_W = 8
QUEUE_ETA_W = 20
QUEUE_REASON_W = 16

# Natural box widths, computed from the table that fits inside each.
# Dashboard shows only the running-jobs table; detail also shows the (wider)
# queue table. On wider terminals, the box is centered.
_RIGHT_MARGIN = 3
_JOB_ROW = 2 + JOBID_W + 1 + PART_W + 1 + NAME_W + 1 + USER_W + 1 + CPU_W + 2 + TIME_W
_QUEUE_ROW = (2 + QUEUE_RANK_W + 1 + JOBID_W + 1 + PART_W + 1 + USER_W + 1 + CPU_W + 2
              + QUEUE_PRI_W + 1 + QUEUE_ETA_W + 1 + QUEUE_REASON_W)
DASHBOARD_BOX_WIDTH = _JOB_ROW + _RIGHT_MARGIN + 2
DETAIL_BOX_WIDTH = _QUEUE_ROW + _RIGHT_MARGIN + 2

# ── Node / cluster shapes ───────────────────────────────────────────────────

@dataclass
class NodeConfig:
    name: str
    total_gpus: int
    gpu_label: str
    my_partitions: List[str]
    all_partitions: List[str]
    shared_partitions: List[str] = field(default_factory=list)
    is_favorite: bool = False
    cpus_per_node: int = 0    # logical CPUs per node
    node_ram: str = ""        # node system RAM, e.g. "1.9TB"

# Populated at startup: discover_topology() returns the full set into
# EXTENDED_NODES, then apply_favorites() partitions it into favorites vs
# extended based on the user's settings.json. Empty until main() runs.
FAVORITE_NODES: List["NodeConfig"] = []
EXTENDED_NODES: List["NodeConfig"] = []

USERNAME = getpass.getuser()

KEY_ESC = 27


def env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer from the environment with a safe fallback."""
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


SLURM_TIMEOUT = env_int("MYSLURMSTATUS_TIMEOUT", 60, 5)
REFRESH_INTERVAL = env_int("MYSLURMSTATUS_REFRESH", 1, 1)
QUEUE_REFRESH_INTERVAL = env_int("MYSLURMSTATUS_QUEUE_REFRESH", 10, 1)
USER_REFRESH_INTERVAL = env_int("MYSLURMSTATUS_USER_REFRESH", 5, 1)
MAX_DETAIL_QUEUE_ROWS = env_int("MYSLURMSTATUS_MAX_QUEUE", 50, 5)
LAST_QUERY_ERROR = ""
_QUERY_ERROR_MSG = ""
_QUERY_ERROR_LOCK = threading.Lock()
_X_OFFSET = 0  # horizontal shift applied by safe_addstr; set by view dispatch, reset for footer

# ── CPU cluster shape ───────────────────────────────────────────────────────

@dataclass
class CpuClusterConfig:
    name: str
    node_pattern: str       # sinfo/squeue hostlist, e.g. "dml[02-25]"
    num_nodes: int
    cpus_per_node: int
    mem_label: str           # e.g. "192G", "1T"
    my_partitions: List[str]
    all_partitions: List[str]
    shared_partitions: List[str] = field(default_factory=list)
    is_favorite: bool = False

FAVORITE_CPU_CLUSTERS: List["CpuClusterConfig"] = []
EXTENDED_CPU_CLUSTERS: List["CpuClusterConfig"] = []
# Populated once at startup by fetch_partition_info(); used by part_limit_line().
PARTITION_LIMITS: dict = {}

# ── User settings file ──────────────────────────────────────────────────────
# Single human-editable JSON at ~/.config/myslurmstatus/settings.json holding
# both the favorites picker output and a couple of preferences. The in-app
# settings panel ('s' shortcut) reads and writes this file.

SETTINGS_PATH = Path(
    os.environ.get("MYSLURMSTATUS_SETTINGS")
    or Path.home() / ".config" / "myslurmstatus" / "settings.json"
)


def load_settings() -> dict:
    """Read settings.json. Returns a dict with favorite_nodes,
    favorite_clusters, show_all_by_default, and sort_running_first;
    missing/invalid file → empty favorites and the documented defaults."""
    try:
        raw = json.loads(SETTINGS_PATH.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "favorite_nodes": list(raw.get("favorite_nodes", [])),
        "favorite_clusters": list(raw.get("favorite_clusters", [])),
        "show_all_by_default": bool(raw.get("show_all_by_default", False)),
        "sort_running_first": bool(raw.get("sort_running_first", True)),
    }


def save_settings(settings: dict) -> bool:
    """Write settings.json atomically (write to tmp file, then rename).
    Returns True on success; on failure surfaces the error via
    set_query_error and returns False so the caller can keep running
    instead of crashing curses and corrupting the user's terminal."""
    payload = {
        "favorite_nodes": list(settings.get("favorite_nodes", [])),
        "favorite_clusters": list(settings.get("favorite_clusters", [])),
        "show_all_by_default": bool(settings.get("show_all_by_default", False)),
        "sort_running_first": bool(settings.get("sort_running_first", True)),
    }
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_suffix(SETTINGS_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, SETTINGS_PATH)  # atomic on POSIX
        return True
    except OSError as e:
        set_query_error(f"could not save settings: {e}")
        return False


def apply_favorites(fav_node_names, fav_cluster_names) -> None:
    """Re-split the global node/cluster lists into favorite vs extended.
    `fav_node_names` and `fav_cluster_names` are ordered; the resulting
    FAVORITE_* lists preserve that order so the dashboard renders favorites
    in the user's chosen sequence. Names that don't match a discovered node
    are silently skipped."""
    global FAVORITE_NODES, EXTENDED_NODES, FAVORITE_CPU_CLUSTERS, EXTENDED_CPU_CLUSTERS

    by_node = {nc.name: nc for nc in (FAVORITE_NODES + EXTENDED_NODES)}
    by_cluster = {cc.name: cc for cc in (FAVORITE_CPU_CLUSTERS + EXTENDED_CPU_CLUSTERS)}
    fav_node_set = set(fav_node_names)
    fav_cluster_set = set(fav_cluster_names)

    FAVORITE_NODES = []
    for name in fav_node_names:
        nc = by_node.get(name)
        if nc is not None:
            nc.is_favorite = True
            FAVORITE_NODES.append(nc)
    EXTENDED_NODES = []
    for nc in by_node.values():
        if nc.name not in fav_node_set:
            nc.is_favorite = False
            EXTENDED_NODES.append(nc)

    FAVORITE_CPU_CLUSTERS = []
    for name in fav_cluster_names:
        cc = by_cluster.get(name)
        if cc is not None:
            cc.is_favorite = True
            FAVORITE_CPU_CLUSTERS.append(cc)
    EXTENDED_CPU_CLUSTERS = []
    for cc in by_cluster.values():
        if cc.name not in fav_cluster_set:
            cc.is_favorite = False
            EXTENDED_CPU_CLUSTERS.append(cc)

# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Job:
    jobid: str
    partition: str
    name: str
    user: str
    state: str
    gpus: int
    elapsed: str = ""
    timelimit: str = ""
    nodelist: str = ""
    start_time: str = ""
    priority: str = ""
    reason: str = ""
    cpus: int = 0
    nodes: List[str] = field(default_factory=list)

@dataclass
class NodeData:
    config: NodeConfig
    running_jobs: List[Job] = field(default_factory=list)
    pending_jobs: List[Job] = field(default_factory=list)
    used_gpus: int = 0

@dataclass
class CpuClusterData:
    config: CpuClusterConfig
    running_jobs: List[Job] = field(default_factory=list)
    pending_jobs: List[Job] = field(default_factory=list)
    total_cpus: int = 0
    used_cpus: int = 0
    nodes_in_use: int = 0

@dataclass
class UserJobSummary:
    gpu_running: int = 0
    gpu_gpus: int = 0
    cpu_running: int = 0
    cpu_cpus: int = 0
    gpu_pending: int = 0
    cpu_pending: int = 0
    available: bool = True

@dataclass
class RefreshSnapshot:
    seq: int
    view_mode: str
    show_all: bool
    cpu_show_all: bool
    node_data: List[NodeData] = field(default_factory=list)
    cpu_cluster_data: List[CpuClusterData] = field(default_factory=list)
    user_summary: UserJobSummary = field(default_factory=UserJobSummary)
    include_pending: bool = True
    include_user: bool = True
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

@dataclass
class UserAccess:
    accounts: frozenset = field(default_factory=frozenset)
    groups: frozenset = field(default_factory=frozenset)
    partitions: frozenset = field(default_factory=frozenset)
    available: bool = True

@dataclass
class PartitionInfo:
    limits: dict = field(default_factory=dict)
    shared_partitions: frozenset = field(default_factory=frozenset)
    partition_accounts: dict = field(default_factory=dict)
    partition_groups: dict = field(default_factory=dict)

# ── Slurm Queries ───────────────────────────────────────────────────────────

def gpu_count_from_tres(tres: str) -> int:
    """Extract GPU count from TRES like 'gres/gpu:2' or 'gres/gpu:nvidia_b200:2'."""
    m = re.search(r'gres/gpu[^|]*', tres)
    if not m:
        return 0
    parts = m.group().split(':')
    try:
        return int(parts[-1])
    except ValueError:
        return 0


def clear_query_error() -> None:
    global _QUERY_ERROR_MSG
    with _QUERY_ERROR_LOCK:
        _QUERY_ERROR_MSG = ""


def get_query_error() -> str:
    with _QUERY_ERROR_LOCK:
        return _QUERY_ERROR_MSG


def set_query_error(message: str) -> None:
    global _QUERY_ERROR_MSG
    with _QUERY_ERROR_LOCK:
        if not _QUERY_ERROR_MSG:
            _QUERY_ERROR_MSG = message[:180]


def split_hostlist(value: str) -> List[str]:
    """Split a Slurm hostlist on top-level commas."""
    parts = []
    depth = 0
    start = 0
    for idx, char in enumerate(value):
        if char == "[":
            depth += 1
        elif char == "]" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:idx])
            start = idx + 1
    parts.append(value[start:])
    return [p for p in parts if p]


def expand_host_token(token: str) -> List[str]:
    match = re.search(r"\[([^\]]+)\]", token)
    if not match:
        return [token] if token else []

    prefix = token[:match.start()]
    suffix = token[match.end():]
    tails = expand_host_token(suffix) if suffix else [""]
    expanded = []
    for item in match.group(1).split(","):
        if "-" in item:
            start, end = item.split("-", 1)
            if start.isdigit() and end.isdigit():
                width = max(len(start), len(end))
                values = [str(i).zfill(width) for i in range(int(start), int(end) + 1)]
            else:
                values = [item]
        else:
            values = [item]
        for value in values:
            for tail in tails:
                expanded.append(f"{prefix}{value}{tail}")
    return expanded


def expand_hostlist(hostlist: str) -> List[str]:
    if not hostlist or hostlist in {"N/A", "(null)", "(None)", "None"}:
        return []
    if "[" not in hostlist and "," not in hostlist:
        return [hostlist]
    nodes = []
    for token in split_hostlist(hostlist):
        nodes.extend(expand_host_token(token.strip()))
    return nodes


_CLUSTER_NODE_SET_CACHE: dict = {}


def cluster_node_set(cc: "CpuClusterConfig") -> frozenset:
    cached = _CLUSTER_NODE_SET_CACHE.get(cc.name)
    if cached is None:
        cached = frozenset(expand_hostlist(cc.node_pattern))
        _CLUSTER_NODE_SET_CACHE[cc.name] = cached
    return cached


def split_resource(total: int, count: int) -> List[int]:
    if count <= 0:
        return []
    total = max(0, total)
    base, remainder = divmod(total, count)
    return [base + (1 if idx < remainder else 0) for idx in range(count)]


def priority_key(job: Job) -> int:
    return -int(job.priority) if job.priority.isdigit() else 0


def run_cmd(cmd: Sequence[str], label: str) -> Optional[str]:
    """Run a command. Returns stdout on success, None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SLURM_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        set_query_error(f"{label} timed out after {SLURM_TIMEOUT}s")
        return None
    except FileNotFoundError:
        set_query_error(f"{cmd[0]} not found")
        return None

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        if details:
            details = details.splitlines()[-1]
            set_query_error(f"{label} failed: {details}")
        else:
            set_query_error(f"{label} failed with exit code {result.returncode}")
        return None
    return result.stdout.strip()


def parallel_results(*calls: Callable):
    """Run zero-arg callables concurrently; return results in input order."""
    if len(calls) <= 1:
        return [c() for c in calls]
    results = [None] * len(calls)

    def runner(idx, fn):
        results[idx] = fn()

    threads = [
        threading.Thread(target=runner, args=(i, fn), daemon=True)
        for i, fn in enumerate(calls)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _parse_timelimit(t: str) -> str:
    """Convert a Slurm time limit string to a compact display string (e.g. '4:00:00' → '4h')."""
    t = t.strip().upper()
    if not t or t in ("UNLIMITED", "INFINITE", "N/A", "NONE", "(NULL)"):
        return "unlimited"
    try:
        if '-' in t:
            days, rest = t.split('-', 1)
            hours = int(days) * 24 + int(rest.split(':')[0])
        else:
            hours = int(t.split(':')[0])
        return f"{hours}h"
    except (ValueError, IndexError):
        return t.lower()


def _split_slurm_list(value: str) -> List[str]:
    value = value.strip()
    if not value or value.upper() in {"ALL", "N/A", "NONE", "(NULL)"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _slurm_all(value: str) -> bool:
    return value.strip().upper() == "ALL"


def fetch_partition_info() -> PartitionInfo:
    """Query Slurm; returns partition limits and account-access metadata.
    Uses scontrol for per-partition max time and QOS name, sacctmgr for
    per-user GPU/job limits stored in the QOS. The same scontrol payload also
    exposes AllowAccounts/AllowGroups, which are the general Slurm signals for
    whether a partition is shared (ALL/ALL) or restricted.

    Called once during startup (alongside discover_topology and
    fetch_user_access in `parallel_results`) and cached for the lifetime
    of the dashboard. Cluster-admin changes to QOS limits will only show up
    after a restart — they're stable enough that a per-tick refresh isn't
    worth the extra Slurm load."""

    def _scontrol():
        return run_cmd(["scontrol", "-o", "show", "partition"], "partition info")

    def _sacctmgr():
        return run_cmd(
            ["sacctmgr", "show", "qos", "-P", "-n",
             "format=Name,MaxJobsPerUser,MaxTRESPerUser"],
            "QOS limits",
        )

    scontrol_raw, sacctmgr_raw = parallel_results(_scontrol, _sacctmgr)

    # {qos_name: (max_jobs, max_gpus_per_user)}
    qos_limits: dict = {}
    for line in (sacctmgr_raw or "").splitlines():
        parts = line.strip().split('|')
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            max_jobs = int(parts[1]) if parts[1].strip() else 0
        except ValueError:
            max_jobs = 0
        max_gpus = 0
        m = re.search(r'gres/gpu[^,=]*=(\d+)', parts[2])
        if m:
            max_gpus = int(m.group(1))
        qos_limits[parts[0].strip().lower()] = (max_jobs, max_gpus)

    limits: dict = {}
    shared_partitions = set()
    partition_accounts: dict = {}
    partition_groups: dict = {}
    for line in (scontrol_raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        name_m = re.search(r'\bPartitionName=(\S+)', line)
        if not name_m:
            continue
        name = name_m.group(1)

        time_m = re.search(r'\bMaxTime=(\S+)', line)
        max_time = _parse_timelimit(time_m.group(1) if time_m else "")

        # Match 'QoS=' but not 'AllowQos=' (different capitalisation in Slurm output)
        qos_m = re.search(r'(?:^|\s)QoS=(\S+)', line)
        qos_name = qos_m.group(1) if qos_m else ""

        segments = [max_time] if max_time else []
        max_jobs, max_gpus = qos_limits.get(qos_name.lower(), (0, 0))
        if max_gpus > 0:
            segments.append(f"{max_gpus} GPU{'s' if max_gpus != 1 else ''}/user")
        if max_jobs > 0:
            segments.append(f"{max_jobs} job{'s' if max_jobs != 1 else ''}")

        allow_m = re.search(r'\bAllowAccounts=(\S+)', line)
        allow_accounts = allow_m.group(1) if allow_m else "ALL"
        group_m = re.search(r'\bAllowGroups=(\S+)', line)
        allow_groups = group_m.group(1) if group_m else "ALL"
        if _slurm_all(allow_accounts) and _slurm_all(allow_groups):
            shared_partitions.add(name)
        else:
            if not _slurm_all(allow_accounts):
                partition_accounts[name] = frozenset(_split_slurm_list(allow_accounts))
            if not _slurm_all(allow_groups):
                partition_groups[name] = frozenset(_split_slurm_list(allow_groups))

        limits[name] = " · ".join(segments)

    return PartitionInfo(
        limits=limits,
        shared_partitions=frozenset(shared_partitions),
        partition_accounts=partition_accounts,
        partition_groups=partition_groups,
    )


def current_user_groups() -> frozenset:
    groups = set()
    try:
        for gid in os.getgroups():
            groups.add(grp.getgrgid(gid).gr_name)
        groups.add(grp.getgrgid(os.getgid()).gr_name)
    except (KeyError, OSError):
        pass
    return frozenset(groups)


def fetch_user_access() -> UserAccess:
    """Return the user's Slurm accounts and explicit partition associations.
    Most clusters attach users to accounts and gate partitions with
    scontrol's AllowAccounts, so blank Partition fields are still useful."""
    raw = run_cmd(
        ["sacctmgr", "list", "assoc", "where", f"user={USERNAME}",
         "format=Account,Partition", "-nP"],
        "user associations",
    )
    if raw is None:
        return UserAccess(groups=current_user_groups(), available=False)
    accounts = set()
    partitions = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        account = parts[0].strip() if parts else ""
        partition = parts[1].strip() if len(parts) > 1 else ""
        if account:
            accounts.add(account)
        if partition and partition.upper() not in {"ALL", "N/A", "NONE", "(NULL)"}:
            partitions.add(partition)
    return UserAccess(
        accounts=frozenset(accounts),
        groups=current_user_groups(),
        partitions=frozenset(partitions),
    )


_GPU_NAME_UPPER_TOKENS = {"gtx", "rtx", "b200", "h100", "h200", "a100", "l4", "l40s"}


def _clean_gpu_name(raw: str) -> str:
    s = raw.replace("nvidia_", "").replace("_", " ").strip()
    out = []
    for tok in s.split():
        out.append(tok.upper() if tok.lower() in _GPU_NAME_UPPER_TOKENS
                   else tok[:1].upper() + tok[1:])
    return " ".join(out)


def _parse_gres(gres: str):
    """Slurm GRES string → (total_gpus:int, label:str)."""
    if not gres or gres == "(null)":
        return 0, ""
    total = 0
    labels = []
    for part in gres.split(","):
        bits = part.split(":")
        if not bits or bits[0] != "gpu":
            continue
        if len(bits) == 3:
            try:
                n = int(bits[2])
            except ValueError:
                continue
            total += n
            labels.append(_clean_gpu_name(bits[1]))
        elif len(bits) == 2:
            try:
                n = int(bits[1])
            except ValueError:
                continue
            total += n
            labels.append("GPU")
    if not labels:
        return 0, ""
    uniq = list(dict.fromkeys(labels))
    return total, uniq[0] if len(uniq) == 1 else "/".join(uniq)


def _format_mem(mem_mb: int) -> str:
    if mem_mb <= 0:
        return ""
    gb = mem_mb / 1024
    if gb >= 900:
        return f"{round(gb / 1024)}T"
    return f"{int(round(gb))}G"


def discover_topology() -> tuple:
    """Walk `sinfo` once at startup to learn every node's hardware and which
    partitions back it. Returns (nodes, clusters) — both lists, both with
    is_favorite=False (the user picks favorites in the settings panel).
    CPU-only nodes are grouped into clusters keyed by (cpus, mem, partitions).

    `--all` is required: without it, sinfo skips hidden partitions and the
    nodes that only live in them disappear from the dashboard."""
    raw = run_cmd(
        ["sinfo", "--all", "-h", "-N", "-o", "%P|%n|%c|%m|%G"],
        "topology",
    )
    if not raw:
        return [], []

    node_info: dict = {}
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        partition = parts[0].rstrip("*")
        nodename = parts[1]
        try:
            cpus = int(parts[2].rstrip("+"))
        except ValueError:
            cpus = 0
        try:
            mem = int(parts[3].rstrip("+"))
        except ValueError:
            mem = 0
        gpus, gpu_label = _parse_gres(parts[4])

        info = node_info.setdefault(nodename, {
            "gpus": 0, "gpu_label": "", "cpus": 0, "mem_mb": 0, "partitions": [],
        })
        if gpus > info["gpus"]:
            info["gpus"] = gpus
            info["gpu_label"] = gpu_label
        info["cpus"] = max(info["cpus"], cpus)
        info["mem_mb"] = max(info["mem_mb"], mem)
        if partition not in info["partitions"]:
            info["partitions"].append(partition)

    nodes: List[NodeConfig] = []
    cpu_only_by_shape = defaultdict(list)
    for name, info in sorted(node_info.items()):
        if info["gpus"] > 0:
            nodes.append(NodeConfig(
                name=name,
                total_gpus=info["gpus"],
                gpu_label=info["gpu_label"],
                my_partitions=[],
                all_partitions=list(info["partitions"]),
                cpus_per_node=info["cpus"],
                node_ram=_format_mem(info["mem_mb"]),
            ))
        else:
            key = (info["cpus"], info["mem_mb"], tuple(sorted(info["partitions"])))
            cpu_only_by_shape[key].append((name, info))

    clusters: List[CpuClusterConfig] = []
    for (cpus, mem, parts), members in cpu_only_by_shape.items():
        names = sorted(n for n, _ in members)
        # Cluster name = first node's hostname. Always unique (nodes are unique),
        # and the dashboard shows "(N nodes, …)" next to it so multi-node groups
        # are obvious.
        clusters.append(CpuClusterConfig(
            name=names[0],
            node_pattern=",".join(names),
            num_nodes=len(names),
            cpus_per_node=cpus,
            mem_label=_format_mem(mem),
            my_partitions=[],
            all_partitions=list(parts),
        ))
    clusters.sort(key=lambda c: c.name)
    return nodes, clusters


def apply_user_partitions(
    nodes: List[NodeConfig],
    clusters: List[CpuClusterConfig],
    user_access,
    partition_info: Optional[PartitionInfo] = None,
) -> None:
    """Set shared and restricted partitions for each node/cluster.
    Public partitions come from AllowAccounts=ALL and AllowGroups=ALL. User
    partitions are explicitly associated or match the user's accounts/groups."""
    if isinstance(user_access, UserAccess):
        access = user_access
    else:
        access = UserAccess(partitions=frozenset(user_access or ()))
    info = partition_info or PartitionInfo()
    has_partition_access = bool(
        info.shared_partitions or info.partition_accounts or info.partition_groups
    )
    can_filter_accounts = access.available and bool(access.accounts or access.partitions)
    can_filter_groups = bool(access.groups)

    def classify(all_parts: List[str]) -> tuple:
        my_parts = []
        shared_parts = []
        for p in all_parts:
            if p in info.shared_partitions:
                shared_parts.append(p)
            elif has_partition_access:
                allowed_accounts = info.partition_accounts.get(p)
                allowed_groups = info.partition_groups.get(p)
                has_restriction = (
                    p in access.partitions
                    or allowed_accounts is not None
                    or allowed_groups is not None
                )
                account_ok = (
                    p in access.partitions
                    or allowed_accounts is None
                    or (can_filter_accounts and bool(allowed_accounts & access.accounts))
                )
                group_ok = (
                    allowed_groups is None
                    or (can_filter_groups and bool(allowed_groups & access.groups))
                )
                if has_restriction and account_ok and group_ok:
                    my_parts.append(p)
            else:
                if p in access.partitions:
                    my_parts.append(p)
                else:
                    shared_parts.append(p)
        return my_parts, shared_parts

    for nc in nodes:
        nc.my_partitions, nc.shared_partitions = classify(nc.all_partitions)
    for cc in clusters:
        cc.my_partitions, cc.shared_partitions = classify(cc.all_partitions)


def collect_data(
    nodes: List[NodeConfig],
    previous: List[NodeData] = None,
    include_pending: bool = True,
) -> List[NodeData]:
    """Query Slurm for running and pending jobs on `nodes` and return per-node
    NodeData. Running and pending queries run in parallel; if `include_pending`
    is False (or the pending fetch fails), pending lists from `previous` are
    reused so the UI doesn't flicker to empty between refreshes."""
    if not nodes:
        return []
    node_names = ",".join(n.name for n in nodes)
    configured_names = {n.name for n in nodes}
    previous_pending = {
        nd.config.name: nd.pending_jobs for nd in (previous or [])
    }

    # Pending query is restricted to partitions available on the displayed GPU
    # nodes, so Slurm doesn't ship every queued job in the cluster.
    part_to_nodes = {}
    for nc in nodes:
        for p in display_partitions(nc):
            part_to_nodes.setdefault(p, []).append(nc.name)
    part_str = ",".join(sorted(part_to_nodes))

    def _running():
        return run_cmd(
            [
                "squeue", "-w", node_names,
                "-o", "%i|%P|%j|%u|%T|%b|%M|%l|%N",
                "-h", "-t", "RUNNING",
            ],
            "GPU running jobs",
        )

    def _pending():
        if not (include_pending and part_str):
            return ""
        return run_cmd(
            [
                "squeue", "-p", part_str, "-t", "PENDING",
                "-o", "%i|%P|%j|%u|%T|%b|%S|%Q|%r",
                "-h",
            ],
            "GPU pending jobs",
        )

    running_raw, pending_raw = parallel_results(_running, _pending)

    running_by_node = {n.name: [] for n in nodes}
    for line in (running_raw or "").splitlines():
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) < 9:
            continue
        job = Job(
            jobid=parts[0], partition=parts[1], name=parts[2],
            user=parts[3], state=parts[4], gpus=gpu_count_from_tres(parts[5]),
            elapsed=parts[6], timelimit=parts[7], nodelist=parts[8],
            nodes=expand_hostlist(parts[8]),
        )
        job_nodes = [n for n in job.nodes if n in configured_names]
        if not job_nodes and job.nodelist in running_by_node:
            job_nodes = [job.nodelist]
        for nname, gpus in zip(job_nodes, split_resource(job.gpus, len(job_nodes))):
            running_by_node[nname].append(replace(job, gpus=gpus))

    pending_succeeded = include_pending and pending_raw is not None
    pending_by_node = {n.name: [] for n in nodes}
    if pending_succeeded:
        for line in pending_raw.splitlines():
            if not line.strip():
                continue
            parts = line.split('|')
            if len(parts) < 9:
                continue
            partition = parts[1]
            job = Job(
                jobid=parts[0], partition=partition, name=parts[2],
                user=parts[3], state=parts[4], gpus=gpu_count_from_tres(parts[5]),
                start_time=parts[6] if parts[6] != "N/A" else "",
                priority=parts[7], reason=parts[8],
            )
            if partition in part_to_nodes:
                for nname in part_to_nodes[partition]:
                    pending_by_node[nname].append(job)

    result = []
    for nc in nodes:
        nd = NodeData(config=nc)
        nd.running_jobs = running_by_node.get(nc.name, [])
        pending_jobs = (pending_by_node.get(nc.name, []) if pending_succeeded
                        else previous_pending.get(nc.name, []))
        nd.pending_jobs = sorted(pending_jobs, key=priority_key)
        nd.used_gpus = sum(j.gpus for j in nd.running_jobs)
        result.append(nd)
    return result


def collect_cpu_data(
    clusters: List[CpuClusterConfig],
    previous: List[CpuClusterData] = None,
    include_pending: bool = True,
) -> List[CpuClusterData]:
    """Query Slurm for CPU-cluster utilisation and pending jobs. Running and
    pending fetches run in parallel; pending jobs from `previous` are reused
    when skipped or when the fetch fails so the UI stays stable across refreshes."""
    previous_pending = {
        cd.config.name: cd.pending_jobs for cd in (previous or [])
    }
    all_parts = set()
    for cc in clusters:
        all_parts.update(display_partitions(cc))
    if not all_parts:
        return [CpuClusterData(config=cc, total_cpus=cc.num_nodes * cc.cpus_per_node)
                for cc in clusters]

    part_str = ",".join(sorted(all_parts))

    def _running():
        return run_cmd(
            [
                "squeue", "-p", part_str,
                "-o", "%i|%P|%j|%u|%T|%b|%C|%M|%l|%N",
                "-h", "-t", "RUNNING",
            ],
            "CPU running jobs",
        )

    def _pending():
        if not include_pending:
            return ""
        return run_cmd(
            [
                "squeue", "-p", part_str, "-t", "PENDING",
                "-o", "%i|%P|%j|%u|%T|%b|%C|%S|%Q|%r",
                "-h",
            ],
            "CPU pending jobs",
        )

    running_raw, pending_raw = parallel_results(_running, _pending)

    part_to_clusters = {}
    cluster_node_sets = {cc.name: cluster_node_set(cc) for cc in clusters}
    for cc in clusters:
        for p in display_partitions(cc):
            part_to_clusters.setdefault(p, []).append(cc.name)

    running_by_cluster = {cc.name: [] for cc in clusters}
    for line in (running_raw or "").splitlines():
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) < 10:
            continue
        partition = parts[1]
        try:
            cpus = int(parts[6])
        except ValueError:
            cpus = 0
        job = Job(
            jobid=parts[0], partition=partition, name=parts[2],
            user=parts[3], state=parts[4], gpus=gpu_count_from_tres(parts[5]),
            cpus=cpus, elapsed=parts[7], timelimit=parts[8], nodelist=parts[9],
            nodes=expand_hostlist(parts[9]),
        )
        if partition not in part_to_clusters:
            continue
        matches = []
        job_nodes = set(job.nodes)
        for cname in part_to_clusters[partition]:
            matched_nodes = job_nodes & cluster_node_sets.get(cname, set())
            if matched_nodes:
                matches.append((cname, len(matched_nodes), matched_nodes))
        if not matches and not job_nodes:
            matches = [(cname, 1, set()) for cname in part_to_clusters[partition]]
        weights = [m[1] for m in matches]
        split_cpus = split_resource(cpus, sum(weights))
        offset = 0
        for cname, weight, matched_nodes in matches:
            share = sum(split_cpus[offset:offset + weight]) if split_cpus else cpus
            offset += weight
            running_by_cluster[cname].append(
                replace(job, cpus=share, nodes=sorted(matched_nodes) or job.nodes)
            )

    pending_succeeded = include_pending and pending_raw is not None
    pending_by_cluster = {cc.name: [] for cc in clusters}
    if pending_succeeded:
        for line in pending_raw.splitlines():
            if not line.strip():
                continue
            parts = line.split('|')
            if len(parts) < 10:
                continue
            partition = parts[1]
            try:
                cpus = int(parts[6])
            except ValueError:
                cpus = 0
            job = Job(
                jobid=parts[0], partition=partition, name=parts[2],
                user=parts[3], state=parts[4], gpus=gpu_count_from_tres(parts[5]),
                cpus=cpus, start_time=parts[7] if parts[7] != "N/A" else "",
                priority=parts[8], reason=parts[9],
            )
            if partition in part_to_clusters:
                for cname in part_to_clusters[partition]:
                    pending_by_cluster[cname].append(job)

    result = []
    for cc in clusters:
        cd = CpuClusterData(config=cc)
        cd.total_cpus = cc.num_nodes * cc.cpus_per_node
        cd.running_jobs = running_by_cluster.get(cc.name, [])
        pending_jobs = (pending_by_cluster.get(cc.name, []) if pending_succeeded
                        else previous_pending.get(cc.name, []))
        cd.pending_jobs = sorted(pending_jobs, key=priority_key)
        cd.used_cpus = sum(j.cpus for j in cd.running_jobs)
        cluster_nodes = cluster_node_sets.get(cc.name, set())
        used_nodes = {
            node
            for job in cd.running_jobs
            for node in job.nodes
            if not cluster_nodes or node in cluster_nodes
        }
        cd.nodes_in_use = len(used_nodes)
        result.append(cd)
    return result


def collect_user_job_summary() -> UserJobSummary:
    """Return a one-line summary of the current user's jobs across the whole
    cluster: running/pending counts and total GPUs/CPUs allocated, split by
    GPU vs CPU partitions. `available=False` indicates squeue itself failed,
    so the footer can render 'unavailable' instead of misleading zeros."""
    raw = run_cmd(
        ["squeue", "-u", USERNAME, "-o", "%i|%P|%T|%b|%C", "-h"],
        "user job summary",
    )
    s = UserJobSummary()
    if raw is None:
        s.available = False
        return s
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) < 5:
            continue
        state = parts[2]
        gpus = gpu_count_from_tres(parts[3])
        try:
            cpus = int(parts[4])
        except ValueError:
            cpus = 0
        is_gpu = gpus > 0
        if state == "RUNNING":
            if is_gpu:
                s.gpu_running += 1
                s.gpu_gpus += gpus
            else:
                s.cpu_running += 1
                s.cpu_cpus += cpus
        elif state == "PENDING":
            if is_gpu:
                s.gpu_pending += 1
            else:
                s.cpu_pending += 1
    return s


# ── Helpers ──────────────────────────────────────────────────────────────────

def gpu_bar(used: int, total: int) -> str:
    used = max(0, min(total, used))
    return "█" * used + "░" * max(0, total - used)


def cpu_bar(used: int, total: int, bar_width: int = 16) -> str:
    if total <= 0:
        return "░" * bar_width
    filled = round(bar_width * used / total)
    filled = max(0, min(bar_width, filled))
    return "█" * filled + "░" * (bar_width - filled)


def user_label(job: Job) -> str:
    return f"{job.user}*" if job.user == USERNAME else job.user


def user_attr(job: Job) -> int:
    return curses.color_pair(1) | curses.A_BOLD if job.user == USERNAME else 0


def _job_widths(inner: int, cpu: bool):
    """Return (jobid, partition, name, user, metric, time) widths for running-job rows.
    Uses fixed natural widths when room allows; squeezes PART/NAME on narrow terminals."""
    metric = CPU_W if cpu else GP_W
    overhead = 2 + JOBID_W + 1 + 1 + 1 + USER_W + 1 + metric + 2 + TIME_W
    if inner >= overhead + PART_W + NAME_W:
        return JOBID_W, PART_W, NAME_W, USER_W, metric, TIME_W
    available = max(0, inner - overhead)
    part_w = min(PART_W, max(10, available * PART_W // (PART_W + NAME_W)))
    name_w = max(10, available - part_w)
    return JOBID_W, part_w, name_w, USER_W, metric, TIME_W


def format_job_header(inner: int, cpu: bool = False) -> str:
    j, p, n, u, m, t = _job_widths(inner, cpu)
    label = "CPUs" if cpu else "GP"
    return f"  {'JOBID':<{j}} {'PARTITION':<{p}} {'NAME':<{n}} {'USER':<{u}} {label:>{m}}  {'TIME':<{t}}"


def format_job_row(job: Job, inner: int, cpu: bool = False) -> str:
    j, p, n, u, m, t = _job_widths(inner, cpu)
    val = job.cpus if cpu else job.gpus
    jobid = job.jobid.partition("_")[0][:j]
    return (
        f"  {jobid:<{j}} {job.partition[:p]:<{p}} {job.name[:n]:<{n}} "
        f"{user_label(job)[:u]:<{u}} {val:>{m}}  {job.elapsed[:t]:<{t}}"
    )


def _queue_widths(inner: int, cpu: bool):
    """Return widths for the pending-queue table. Fixed natural widths when room allows."""
    metric = CPU_W if cpu else GP_W
    overhead = (2 + QUEUE_RANK_W + 1 + JOBID_W + 1 + 1 + USER_W + 1 + metric + 2
                + QUEUE_PRI_W + 1 + QUEUE_ETA_W + 1)
    if inner >= overhead + PART_W + QUEUE_REASON_W:
        return QUEUE_RANK_W, JOBID_W, PART_W, USER_W, metric, QUEUE_PRI_W, QUEUE_ETA_W, QUEUE_REASON_W
    available = max(0, inner - overhead)
    p = min(PART_W, max(10, available * PART_W // (PART_W + QUEUE_REASON_W)))
    r = max(8, available - p)
    return QUEUE_RANK_W, JOBID_W, p, USER_W, metric, QUEUE_PRI_W, QUEUE_ETA_W, r


def format_queue_header(inner: int, cpu: bool = False) -> str:
    r, j, p, u, m, pr, e, rs = _queue_widths(inner, cpu)
    label = "CPUs" if cpu else "GP"
    return (
        f"  {'#':<{r}} {'JOBID':<{j}} {'PARTITION':<{p}} {'USER':<{u}} "
        f"{label:>{m}}  {'PRI':<{pr}} {'ETA':<{e}} {'REASON':<{rs}}"
    )


def format_queue_row(rank: int, job: Job, inner: int, cpu: bool = False) -> str:
    r, j, p, u, m, pr, e, rs = _queue_widths(inner, cpu)
    val = job.cpus if cpu else job.gpus
    eta = (job.start_time or "N/A")[:e]
    reason = (job.reason or "")[:rs]
    jobid = job.jobid.partition("_")[0][:j]
    return (
        f"  {rank:<{r}} {jobid:<{j}} {job.partition[:p]:<{p}} "
        f"{user_label(job)[:u]:<{u}} {val:>{m}}  {job.priority[:pr]:<{pr}} {eta:<{e}} {reason:<{rs}}"
    )


def visible_partitions(
    all_parts: List[str],
    my_parts: List[str],
    shared_parts: Optional[List[str]] = None,
) -> List[str]:
    """Return available partitions with user/starred partitions first."""
    my_set = set(my_parts)
    if shared_parts is None:
        return list(all_parts)
    shared_set = set(shared_parts)
    mine = [p for p in all_parts if p in my_set]
    shared = [p for p in all_parts if p in shared_set and p not in my_set]
    return mine + shared


def display_partitions(item) -> List[str]:
    return visible_partitions(
        item.all_partitions,
        item.my_partitions,
        getattr(item, "shared_partitions", None),
    )


def filter_accessible_items(items):
    return [item for item in items if display_partitions(item)]


def empty_view_message(
    label: str,
    loading: bool,
    favorites_only: bool,
    favorite_count: int,
    total_count: int,
) -> tuple:
    if loading:
        return f"Loading {label}...", ""
    if favorites_only and favorite_count == 0 and total_count > 0:
        return f"No {label} favorites configured", "Press a to show all, or s to choose favorites"
    if favorites_only and total_count > 0:
        return f"No {label} favorites to show", "Press a to show all, or s to edit favorites"
    return f"No {label} data available", ""


def part_limit_line(p: str) -> str:
    """Format a partition line with its limit info for the detail view."""
    info = PARTITION_LIMITS.get(p, "")
    return f"    {p}  —  {info}" if info else f"    {p}"


def partition_lines(my_parts, all_parts, pending_count: int, inner: int) -> List[str]:
    """Format the 'Available Partitions: ...' line(s) for a node/cluster, wrapping if too long.
    Partitions in my_parts are marked with * to indicate priority access."""
    my_set = set(my_parts)
    tokens = [f"{p}*" if p in my_set else p for p in all_parts]
    queue_suffix = f"  |  Queue: {pending_count}" if pending_count else ""
    prefix = "  Available Partitions: "
    indent = " " * len(prefix)
    one_line = prefix + ", ".join(tokens) + queue_suffix
    if len(one_line) <= inner:
        return [one_line]
    lines: List[str] = []
    current = prefix
    line_has_token = False
    for tok in tokens:
        sep = ", " if line_has_token else ""
        candidate = current + sep + tok
        if len(candidate) > inner and line_has_token:
            lines.append(current)
            current = indent + tok
        else:
            current = candidate
        line_has_token = True
    if queue_suffix:
        if len(current + queue_suffix) <= inner:
            current += queue_suffix
        else:
            lines.append(current)
            current = indent + queue_suffix.lstrip()
    lines.append(current)
    return lines


# ── Curses TUI ──────────────────────────────────────────────────────────────

def curses_main(stdscr, show_all_initial: bool, cpu_show_all_initial: bool,
                user_settings: dict):
    global LAST_QUERY_ERROR, _X_OFFSET

    curses.set_escdelay(25)  # Make Esc key respond fast (25ms instead of 1000ms)
    curses.curs_set(0)
    # Claim the wheel so it doesn't bleed through to the terminal's scrollback.
    curses.mousemask(curses.ALL_MOUSE_EVENTS)
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)      # user jobs
    curses.init_pair(2, curses.COLOR_RED, -1)        # full
    curses.init_pair(3, curses.COLOR_GREEN, -1)      # free/idle
    curses.init_pair(4, curses.COLOR_YELLOW, -1)     # partial
    curses.init_pair(5, curses.COLOR_WHITE, -1)      # header
    curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_WHITE)  # selected
    curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_CYAN)   # selected header
    curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_BLUE)   # footer
    curses.init_pair(9, curses.COLOR_MAGENTA, -1)    # section headers in detail

    # GPU view state
    show_all = show_all_initial
    selected = 0
    detail_node = None  # index of node in detail view
    detail_scroll = 0
    last_refresh = 0
    node_data = []

    # CPU view state
    view_mode = "gpu"  # "gpu" or "cpu"
    cpu_show_all = cpu_show_all_initial
    cpu_selected = 0
    cpu_detail_cluster = None
    cpu_detail_scroll = 0
    cpu_cluster_data = []

    # Shared state
    user_summary = UserJobSummary()
    last_queue_refresh = 0
    last_user_refresh = 0
    refresh_lock = threading.Lock()
    refresh_thread = None
    refresh_result = None
    refresh_seq = 0
    refresh_requested = False

    # Main-view search state ('/' shortcut)
    search_active = False
    search_query = ""
    selected_before_search = 0  # restored on Esc; per-view (set when entering search)

    # Settings panel state ('s' shortcut)
    settings_open = False
    settings_idx = 0
    settings_query = ""
    settings_searching = False
    # Ordered lists, not sets — order is what the user sees on the dashboard.
    settings_marked_nodes: list = []
    settings_marked_clusters: list = []
    settings_section = "favorites"  # "favorites" or "preferences"
    settings_prefs = dict(user_settings)
    settings_pref_keys = ["show_all_by_default", "sort_running_first"]
    settings_pref_idx = 0

    def get_nodes():
        return FAVORITE_NODES + EXTENDED_NODES if show_all else FAVORITE_NODES

    def get_cpu_clusters():
        return FAVORITE_CPU_CLUSTERS + EXTENDED_CPU_CLUSTERS if cpu_show_all else FAVORITE_CPU_CLUSTERS

    def current_target():
        return (view_mode, show_all, cpu_show_all)

    def is_refreshing():
        return refresh_thread is not None and refresh_thread.is_alive()

    def collect_refresh(
        seq, target, nodes, clusters, previous_nodes, previous_clusters,
        base_user_summary, include_pending, include_user
    ):
        nonlocal refresh_result
        started = time.time()
        snapshot = RefreshSnapshot(
            seq=seq,
            view_mode=target[0],
            show_all=target[1],
            cpu_show_all=target[2],
            user_summary=base_user_summary,
            include_pending=include_pending,
            include_user=include_user,
            started_at=started,
        )
        try:
            clear_query_error()
            if target[0] == "gpu":
                def _main():
                    return ("gpu", collect_data(
                        nodes, previous=previous_nodes, include_pending=include_pending,
                    ))
            else:
                def _main():
                    return ("cpu", collect_cpu_data(
                        clusters, previous=previous_clusters, include_pending=include_pending,
                    ))
            calls = [_main]
            if include_user:
                calls.append(collect_user_job_summary)
            results = parallel_results(*calls)
            mode, payload = results[0]
            if mode == "gpu":
                snapshot.node_data = payload
            else:
                snapshot.cpu_cluster_data = payload
            if include_user:
                snapshot.user_summary = results[1]
            snapshot.error = get_query_error()
        except Exception as exc:
            snapshot.error = f"refresh failed: {exc}"
        snapshot.finished_at = time.time()
        with refresh_lock:
            refresh_result = snapshot

    def start_refresh(force: bool = False):
        nonlocal refresh_thread, refresh_seq, refresh_requested
        global LAST_QUERY_ERROR
        if is_refreshing():
            refresh_requested = refresh_requested or force
            return

        refresh_requested = False
        LAST_QUERY_ERROR = ""
        refresh_seq += 1
        now = time.time()
        target = current_target()
        nodes = get_nodes() if target[0] == "gpu" else []
        clusters = get_cpu_clusters() if target[0] == "cpu" else []
        include_pending = force or not last_queue_refresh or now - last_queue_refresh >= QUEUE_REFRESH_INTERVAL
        include_user = force or not last_user_refresh or now - last_user_refresh >= USER_REFRESH_INTERVAL
        previous_nodes = list(node_data)
        previous_clusters = list(cpu_cluster_data)
        base_user_summary = user_summary
        refresh_thread = threading.Thread(
            target=collect_refresh,
            args=(
                refresh_seq, target, nodes, clusters, previous_nodes,
                previous_clusters, base_user_summary, include_pending, include_user
            ),
            daemon=True,
        )
        refresh_thread.start()

    def _has_user_running(running_jobs) -> bool:
        return any(j.user == USERNAME for j in running_jobs)

    def _sort_user_first(items):
        # Stable sort: items with user's running jobs go first; relative
        # order within each group is preserved (favorites stay in user
        # order, then extended in discovery order).
        items.sort(key=lambda d: 0 if _has_user_running(d.running_jobs) else 1)

    def apply_snapshot(snapshot):
        nonlocal node_data, cpu_cluster_data, last_refresh, user_summary
        nonlocal selected, cpu_selected, detail_node, cpu_detail_cluster
        nonlocal last_queue_refresh, last_user_refresh
        if snapshot.view_mode == "gpu":
            node_data = snapshot.node_data
            if user_settings.get("sort_running_first", True):
                _sort_user_first(node_data)
            selected = min(selected, max(0, len(node_data) - 1))
            if detail_node is not None and detail_node >= len(node_data):
                detail_node = None
        else:
            cpu_cluster_data = snapshot.cpu_cluster_data
            if user_settings.get("sort_running_first", True):
                _sort_user_first(cpu_cluster_data)
            cpu_selected = min(cpu_selected, max(0, len(cpu_cluster_data) - 1))
            if cpu_detail_cluster is not None and cpu_detail_cluster >= len(cpu_cluster_data):
                cpu_detail_cluster = None
        user_summary = snapshot.user_summary
        last_refresh = snapshot.finished_at
        if snapshot.include_pending:
            last_queue_refresh = snapshot.finished_at
        if snapshot.include_user:
            last_user_refresh = snapshot.finished_at

    def finish_refresh_if_ready():
        nonlocal refresh_thread, refresh_result, refresh_requested
        global LAST_QUERY_ERROR
        snapshot = None
        with refresh_lock:
            if refresh_result is not None:
                snapshot = refresh_result
                refresh_result = None

        if snapshot is None:
            if refresh_thread is not None and not refresh_thread.is_alive():
                refresh_thread = None
            return

        refresh_thread = None
        target = (snapshot.view_mode, snapshot.show_all, snapshot.cpu_show_all)
        if target == current_target():
            apply_snapshot(snapshot)
            LAST_QUERY_ERROR = snapshot.error

        if refresh_requested or target != current_target():
            refresh_requested = False
            start_refresh(force=True)

    def render_waiting(
        label: str,
        width: int,
        height: int,
        favorites_only: bool,
        favorite_count: int,
        total_count: int,
    ):
        msg, hint = empty_view_message(
            label,
            is_refreshing(),
            favorites_only,
            favorite_count,
            total_count,
        )
        row = max(0, height // 2 - 1)
        safe_addstr(stdscr, row, max(0, (width - len(msg)) // 2), msg, curses.A_BOLD)
        if hint:
            safe_addstr(stdscr, row + 1, max(0, (width - len(hint)) // 2), hint, curses.A_DIM)
        if LAST_QUERY_ERROR:
            err = LAST_QUERY_ERROR[:max(0, width - 4)]
            safe_addstr(stdscr, row + (2 if hint else 1), 2, err, curses.color_pair(4))
        render_footer(stdscr, height - 1, width, " s:Settings  a:All  r:Refresh  Esc:Quit")

    start_refresh(force=True)
    stdscr.timeout(200)

    while True:
        finish_refresh_if_ready()
        if not is_refreshing() and time.time() - last_refresh > REFRESH_INTERVAL:
            start_refresh()

        stdscr.erase()
        height, width = stdscr.getmaxyx()

        def center_offset(box_width):
            return max(0, (width - min(width, box_width)) // 2)

        if settings_open:
            _X_OFFSET = 0
            items = (FAVORITE_NODES + EXTENDED_NODES) if view_mode == "gpu" else (
                FAVORITE_CPU_CLUSTERS + EXTENDED_CPU_CLUSTERS)
            marked = settings_marked_nodes if view_mode == "gpu" else settings_marked_clusters
            label = "GPU nodes" if view_mode == "gpu" else "CPU clusters"
            render_settings_view(stdscr, settings_section, items, settings_idx, marked,
                                 settings_query, settings_searching, label,
                                 settings_prefs, settings_pref_keys, settings_pref_idx,
                                 width, height)
        elif view_mode == "gpu":
            visible_nd = (filter_by_query(node_data, search_query,
                                          name_of=lambda nd: _searchable_text(nd.config))
                          if search_active else node_data)
            if not node_data:
                _X_OFFSET = 0
                render_waiting(
                    "GPU",
                    width,
                    height,
                    favorites_only=not show_all,
                    favorite_count=len(FAVORITE_NODES),
                    total_count=len(FAVORITE_NODES) + len(EXTENDED_NODES),
                )
            elif detail_node is not None and detail_node < len(node_data):
                _X_OFFSET = center_offset(DETAIL_BOX_WIDTH)
                render_detail_view(stdscr, node_data[detail_node], detail_scroll, width, height)
            else:
                _X_OFFSET = center_offset(DASHBOARD_BOX_WIDTH)
                sel_in_view = min(selected, max(0, len(visible_nd) - 1))
                render_dashboard_view(stdscr, visible_nd, sel_in_view, width, height, show_all, user_summary)
                if search_active:
                    bar = f"  / {search_query}_  ({len(visible_nd)}/{len(node_data)} match)"
                    safe_addstr(stdscr, height - 2, 0, bar.ljust(width),
                                curses.color_pair(4) | curses.A_BOLD)
        else:
            visible_cd = (filter_by_query(cpu_cluster_data, search_query,
                                          name_of=lambda cd: _searchable_text(cd.config))
                          if search_active else cpu_cluster_data)
            if not cpu_cluster_data:
                _X_OFFSET = 0
                render_waiting(
                    "CPU",
                    width,
                    height,
                    favorites_only=not cpu_show_all,
                    favorite_count=len(FAVORITE_CPU_CLUSTERS),
                    total_count=len(FAVORITE_CPU_CLUSTERS) + len(EXTENDED_CPU_CLUSTERS),
                )
            elif cpu_detail_cluster is not None and cpu_detail_cluster < len(cpu_cluster_data):
                _X_OFFSET = center_offset(DETAIL_BOX_WIDTH)
                render_cpu_detail_view(stdscr, cpu_cluster_data[cpu_detail_cluster], cpu_detail_scroll, width, height)
            else:
                _X_OFFSET = center_offset(DASHBOARD_BOX_WIDTH)
                sel_in_view = min(cpu_selected, max(0, len(visible_cd) - 1))
                render_cpu_dashboard_view(stdscr, visible_cd, sel_in_view, width, height, cpu_show_all, user_summary)
                if search_active:
                    bar = f"  / {search_query}_  ({len(visible_cd)}/{len(cpu_cluster_data)} match)"
                    safe_addstr(stdscr, height - 2, 0, bar.ljust(width),
                                curses.color_pair(4) | curses.A_BOLD)

        stdscr.refresh()

        ch = stdscr.getch()

        if ch == -1:
            continue

        # Wheel up/down feed the existing arrow-key handling. Clicks are dropped.
        if ch == curses.KEY_MOUSE:
            try:
                bstate = curses.getmouse()[4]
            except curses.error:
                continue
            if bstate & curses.BUTTON4_PRESSED:
                ch = curses.KEY_UP
            elif bstate & getattr(curses, "BUTTON5_PRESSED", 0x00200000):
                ch = curses.KEY_DOWN
            else:
                continue

        BACKSPACE_KEYS = (curses.KEY_BACKSPACE, 127, 8)
        ENTER_KEYS = (curses.KEY_ENTER, 10, 13)

        if settings_open:
            items = (FAVORITE_NODES + EXTENDED_NODES) if view_mode == "gpu" else (
                FAVORITE_CPU_CLUSTERS + EXTENDED_CPU_CLUSTERS)
            marked = settings_marked_nodes if view_mode == "gpu" else settings_marked_clusters
            visible = filter_by_query(_ordered_settings_items(items, marked),
                                      settings_query, name_of=_searchable_text)

            def _save_and_close():
                nonlocal settings_open, settings_query, settings_idx
                nonlocal settings_searching, detail_node, cpu_detail_cluster
                nonlocal selected, cpu_selected, node_data, cpu_cluster_data
                user_settings["favorite_nodes"] = list(settings_marked_nodes)
                user_settings["favorite_clusters"] = list(settings_marked_clusters)
                for _k in settings_pref_keys:
                    user_settings[_k] = settings_prefs[_k]
                if not save_settings(user_settings):
                    # Disk write failed — keep the panel open so the user sees
                    # the error in the footer and can retry or Esc to discard.
                    return
                apply_favorites(settings_marked_nodes, settings_marked_clusters)
                settings_open = False
                settings_query = ""
                settings_idx = 0
                settings_searching = False
                detail_node = None
                cpu_detail_cluster = None
                selected = 0
                cpu_selected = 0
                node_data = []
                cpu_cluster_data = []
                start_refresh(force=True)

            if settings_section == "favorites" and settings_searching:
                if ch == KEY_ESC:
                    settings_searching = False
                    settings_query = ""
                    settings_idx = 0
                elif ch in ENTER_KEYS:
                    settings_searching = False
                elif ch in BACKSPACE_KEYS:
                    settings_query = settings_query[:-1]
                    settings_idx = 0
                elif ch == ord(' ') and visible:
                    # Space inside search picks the highlighted result and
                    # drops back to the favorites layout so the user can see
                    # the new pick land at the bottom of the favorites group.
                    target = visible[min(settings_idx, len(visible) - 1)].name
                    if target in marked:
                        marked.remove(target)
                    else:
                        marked.append(target)
                    settings_searching = False
                    settings_query = ""
                    settings_idx = 0
                elif 32 < ch < 127:
                    settings_query += chr(ch)
                    settings_idx = 0
            elif settings_section == "favorites":
                if ch == KEY_ESC:
                    settings_open = False
                    settings_query = ""
                    settings_idx = 0
                    settings_searching = False
                elif ch == ord('\t'):
                    settings_section = "preferences"
                elif ch == ord('/'):
                    settings_searching = True
                elif ch == curses.KEY_UP:
                    settings_idx = max(0, settings_idx - 1)
                elif ch == curses.KEY_DOWN:
                    settings_idx = min(max(0, len(visible) - 1), settings_idx + 1)
                elif ch == ord(' ') and visible:
                    target = visible[min(settings_idx, len(visible) - 1)].name
                    if target in marked:
                        marked.remove(target)
                    else:
                        # Append: the newly-checked item slots into the
                        # favorites group right after the previous one, so
                        # toggle order builds the displayed order naturally.
                        marked.append(target)
                elif ch == ord('[') and visible:
                    target = visible[min(settings_idx, len(visible) - 1)].name
                    if target in marked:
                        i = marked.index(target)
                        if i > 0:
                            marked[i - 1], marked[i] = marked[i], marked[i - 1]
                elif ch == ord(']') and visible:
                    target = visible[min(settings_idx, len(visible) - 1)].name
                    if target in marked:
                        i = marked.index(target)
                        if i < len(marked) - 1:
                            marked[i + 1], marked[i] = marked[i], marked[i + 1]
                elif ch in ENTER_KEYS:
                    _save_and_close()
            else:  # preferences
                if ch == KEY_ESC:
                    settings_open = False
                    settings_section = "favorites"
                elif ch == ord('\t'):
                    settings_section = "favorites"
                elif ch == curses.KEY_UP:
                    settings_pref_idx = max(0, settings_pref_idx - 1)
                elif ch == curses.KEY_DOWN:
                    settings_pref_idx = min(len(settings_pref_keys) - 1, settings_pref_idx + 1)
                elif ch == ord(' '):
                    key = settings_pref_keys[settings_pref_idx]
                    settings_prefs[key] = not settings_prefs.get(key, False)
                elif ch in ENTER_KEYS:
                    _save_and_close()
            continue

        if search_active:
            items_full = node_data if view_mode == "gpu" else cpu_cluster_data
            visible_full = filter_by_query(items_full, search_query,
                                           name_of=lambda x: _searchable_text(x.config))
            cur_sel = selected if view_mode == "gpu" else cpu_selected

            if ch == KEY_ESC:
                search_active = False
                search_query = ""
                if view_mode == "gpu":
                    selected = selected_before_search
                else:
                    cpu_selected = selected_before_search
            elif ch in ENTER_KEYS:
                if visible_full:
                    chosen = visible_full[min(cur_sel, len(visible_full) - 1)].config.name
                    for i, x in enumerate(items_full):
                        if x.config.name == chosen:
                            if view_mode == "gpu":
                                selected = i
                            else:
                                cpu_selected = i
                            break
                search_active = False
                search_query = ""
            elif ch in BACKSPACE_KEYS:
                search_query = search_query[:-1]
                if view_mode == "gpu":
                    selected = 0
                else:
                    cpu_selected = 0
            elif ch == curses.KEY_UP:
                new_sel = max(0, cur_sel - 1)
                if view_mode == "gpu":
                    selected = new_sel
                else:
                    cpu_selected = new_sel
            elif ch == curses.KEY_DOWN:
                new_sel = min(max(0, len(visible_full) - 1), cur_sel + 1)
                if view_mode == "gpu":
                    selected = new_sel
                else:
                    cpu_selected = new_sel
            elif 32 <= ch < 127 and chr(ch) != '/':
                search_query += chr(ch)
                if view_mode == "gpu":
                    selected = 0
                else:
                    cpu_selected = 0
            continue

        # 'q' = quit from anywhere outside a text-input mode. Settings panel
        # and main-view search both `continue` earlier, so 'q' is still a
        # normal printable character in those contexts.
        if ch == ord('q'):
            break

        if view_mode == "gpu":
            if detail_node is not None:
                # GPU detail view controls
                if ch == KEY_ESC:
                    detail_node = None
                    detail_scroll = 0
                elif ch == curses.KEY_UP:
                    detail_scroll = max(0, detail_scroll - 1)
                elif ch == curses.KEY_DOWN:
                    detail_scroll += 1
            else:
                # GPU dashboard controls
                if ch == KEY_ESC:
                    break
                elif ch == curses.KEY_UP:
                    selected = max(0, selected - 1)
                elif ch == curses.KEY_DOWN and node_data:
                    selected = min(len(node_data) - 1, selected + 1)
                elif ch in ENTER_KEYS and node_data:
                    detail_node = selected
                    detail_scroll = 0
                elif ch == ord('a'):
                    show_all = not show_all
                    detail_node = None
                    node_data = []
                    selected = 0
                    start_refresh(force=True)
                elif ch == ord('r'):
                    start_refresh(force=True)
                elif ch == ord('c'):
                    view_mode = "cpu"
                    detail_node = None
                    start_refresh(force=True)
                elif ch == ord('/') and node_data:
                    search_active = True
                    search_query = ""
                    selected_before_search = selected
                    selected = 0
                elif ch == ord('s'):
                    settings_open = True
                    settings_marked_nodes = [n.name for n in FAVORITE_NODES]
                    settings_marked_clusters = [c.name for c in FAVORITE_CPU_CLUSTERS]
                    settings_idx = 0
                    settings_query = ""
                    settings_searching = False
        else:
            if cpu_detail_cluster is not None:
                # CPU detail view controls
                if ch == KEY_ESC:
                    cpu_detail_cluster = None
                    cpu_detail_scroll = 0
                elif ch == curses.KEY_UP:
                    cpu_detail_scroll = max(0, cpu_detail_scroll - 1)
                elif ch == curses.KEY_DOWN:
                    cpu_detail_scroll += 1
            else:
                # CPU dashboard controls
                if ch == KEY_ESC:
                    break
                elif ch == ord('c'):  # c -> back to GPU view
                    view_mode = "gpu"
                    cpu_detail_cluster = None
                    start_refresh(force=True)
                elif ch == curses.KEY_UP:
                    cpu_selected = max(0, cpu_selected - 1)
                elif ch == curses.KEY_DOWN and cpu_cluster_data:
                    cpu_selected = min(len(cpu_cluster_data) - 1, cpu_selected + 1)
                elif ch in ENTER_KEYS and cpu_cluster_data:
                    cpu_detail_cluster = cpu_selected
                    cpu_detail_scroll = 0
                elif ch == ord('a'):
                    cpu_show_all = not cpu_show_all
                    cpu_detail_cluster = None
                    cpu_cluster_data = []
                    cpu_selected = 0
                    start_refresh(force=True)
                elif ch == ord('r'):
                    start_refresh(force=True)
                elif ch == ord('/') and cpu_cluster_data:
                    search_active = True
                    search_query = ""
                    selected_before_search = cpu_selected
                    cpu_selected = 0
                elif ch == ord('s'):
                    settings_open = True
                    settings_marked_nodes = [n.name for n in FAVORITE_NODES]
                    settings_marked_clusters = [c.name for c in FAVORITE_CPU_CLUSTERS]
                    settings_idx = 0
                    settings_query = ""
                    settings_searching = False


def safe_addstr(win, y, x, text, attr=0):
    """Write text to window, applying _X_OFFSET so content can be centered. Truncates to fit."""
    h, w = win.getmaxyx()
    actual_x = x + _X_OFFSET
    if y < 0 or y >= h or actual_x >= w or actual_x < 0:
        return
    avail = w - actual_x
    if avail <= 0:
        return
    text = text[:avail]
    try:
        win.addstr(y, actual_x, text, attr)
    except curses.error:
        pass


def render_footer(stdscr, row: int, width: int, controls: str) -> None:
    """Footer always spans the full terminal; resets _X_OFFSET so it ignores box centering."""
    global _X_OFFSET
    _X_OFFSET = 0
    footer = controls
    attr = curses.color_pair(8)
    if LAST_QUERY_ERROR:
        footer = f"{footer}  |  {LAST_QUERY_ERROR}"
        attr = curses.color_pair(8) | curses.A_BOLD
    safe_addstr(stdscr, row, 0, " " * width, curses.color_pair(8))
    safe_addstr(stdscr, row, 0, footer, attr)


def filter_by_query(items, query: str, name_of=lambda x: x.name):
    """Case-insensitive substring filter; preserves input order."""
    if not query:
        return list(items)
    q = query.lower()
    return [x for x in items if q in name_of(x).lower()]


def _searchable_text(item) -> str:
    """Concatenate searchable fields for filter_by_query: hostname,
    GPU label (so 'b200' or 'rtx' match by hardware) and partition names
    (so 'p_zohar' matches the cluster behind that partition)."""
    bits = [item.name]
    label = getattr(item, "gpu_label", "")
    if label:
        bits.append(label)
    bits.extend(display_partitions(item))
    return " ".join(bits)


def _ordered_settings_items(items, marked):
    """Reorder items: favorites (in `marked` order) first, rest after.
    Same ordering used by both the panel render and its keymap so the
    cursor index always refers to the same item."""
    by_name = {item.name: item for item in items}
    out = [by_name[n] for n in marked if n in by_name]
    out += [item for item in items if item.name not in marked]
    return out


_PREF_LABELS = {
    "show_all_by_default": "Show all (favorites + extended) by default",
    "sort_running_first":  "Sort entries with my running jobs to the top",
}


def render_settings_view(stdscr, section: str, items, idx: int, marked: list,
                         query: str, searching: bool, view_label: str,
                         prefs: dict, pref_keys, pref_idx: int,
                         width: int, height: int):
    """Settings panel. Two sections, switched with Tab:
      - 'favorites': checkbox list of nodes/clusters with optional search.
      - 'preferences': small key/value list of toggles."""
    fav_tab = "[ Favorites ]" if section == "favorites" else "  Favorites  "
    pref_tab = "[ Preferences ]" if section == "preferences" else "  Preferences  "
    title = f"  Settings  {fav_tab}  {pref_tab}    Tab to switch"
    safe_addstr(stdscr, 0, 0, title.ljust(width), curses.color_pair(7) | curses.A_BOLD)

    if section == "favorites":
        # Favorites first in `marked` order, then the rest. Toggling slots
        # the new pick into the favorites group right after the previous one.
        visible = filter_by_query(_ordered_settings_items(items, marked),
                                  query, name_of=_searchable_text)
        sub = f"  Favorite {view_label}  —  {len(marked)} selected, {len(items)} total"
        safe_addstr(stdscr, 1, 0, sub, curses.A_DIM)

        avail = height - 4
        first = max(0, min(idx - avail // 2, max(0, len(visible) - avail)))
        last = first + avail

        for i, item in enumerate(visible[first:last]):
            row = 3 + i
            if row >= height - 2:
                break
            is_sel = (first + i == idx)
            check = "[x] " if item.name in marked else "[ ] "
            item_parts = display_partitions(item)
            partitions = ", ".join(item_parts[:4])
            if len(item_parts) > 4:
                partitions += f", +{len(item_parts) - 4} more"
            line = f"  {check}{item.name:<20} {partitions}"
            attr = curses.color_pair(7) | curses.A_BOLD if is_sel else (
                curses.color_pair(1) if item.name in marked else 0)
            safe_addstr(stdscr, row, 0, line[:width].ljust(width), attr)

        if not visible:
            msg = f"  (no matches for \"{query}\")" if query else "  (nothing to show)"
            safe_addstr(stdscr, 3, 0, msg, curses.A_DIM)

        if searching:
            bar = f"  / {query}_"
            safe_addstr(stdscr, height - 2, 0, bar.ljust(width),
                        curses.color_pair(4) | curses.A_BOLD)
            controls = " Type to filter  Enter:Done  Esc:Clear  Backspace:Erase"
        else:
            controls = (" ↑↓:Move  Space:Toggle  [ / ]:Reorder fav  "
                        "/:Search  Tab:Preferences  Enter:Save  Esc:Cancel")
    else:
        safe_addstr(stdscr, 1, 0, "  Preferences", curses.A_DIM)
        for i, key in enumerate(pref_keys):
            row = 3 + i
            is_sel = (i == pref_idx)
            check = "[x] " if prefs.get(key) else "[ ] "
            label = _PREF_LABELS.get(key, key)
            line = f"  {check}{label}"
            attr = curses.color_pair(7) | curses.A_BOLD if is_sel else (
                curses.color_pair(1) if prefs.get(key) else 0)
            safe_addstr(stdscr, row, 0, line[:width].ljust(width), attr)
        controls = " ↑↓:Move  Space:Toggle  Tab:Favorites  Enter:Save  Esc:Cancel"

    render_footer(stdscr, height - 1, width, controls)


def format_user_jobs_str(us: UserJobSummary) -> str:
    """Format user job summary for the banner."""
    if not us.available:
        return "You: unavailable  "
    parts = []
    if us.gpu_running > 0:
        gpu_w = "GPU" if us.gpu_gpus == 1 else "GPUs"
        parts.append(f"{us.gpu_running} GPU ({us.gpu_gpus} {gpu_w})")
    if us.cpu_running > 0:
        cpu_w = "CPU" if us.cpu_cpus == 1 else "CPUs"
        parts.append(f"{us.cpu_running} CPU ({us.cpu_cpus} {cpu_w})")
    if not parts:
        return "You: no running jobs  "
    sep = " \u00b7 "
    return f"You: {sep.join(parts)}  "


def selected_pending_rows(pending: List[Job]):
    ranked = list(enumerate(pending, 1))
    if len(ranked) <= MAX_DETAIL_QUEUE_ROWS:
        return ranked, 0

    shown = []
    seen = set()
    for rank, job in ranked:
        if job.user == USERNAME:
            shown.append((rank, job))
            seen.add(job.jobid)
            if len(shown) >= MAX_DETAIL_QUEUE_ROWS:
                return shown, len(ranked) - len(shown)

    for rank, job in ranked:
        if len(shown) >= MAX_DETAIL_QUEUE_ROWS:
            break
        if job.jobid in seen:
            continue
        shown.append((rank, job))
        seen.add(job.jobid)

    return shown, len(ranked) - len(shown)


def queue_title(pending: List[Job], shown_count: int) -> str:
    if shown_count >= len(pending):
        return f"  QUEUE ({len(pending)} pending jobs)"
    return f"  QUEUE ({len(pending)} pending jobs, showing {shown_count}; yours first)"


def my_pending_summary(pending: List[Job], subject: str) -> tuple:
    """Return (text, has_user_pending) for the user-summary footer line."""
    positions = []
    for rank, job in enumerate(pending, 1):
        if job.user != USERNAME:
            continue
        eta = job.start_time if job.start_time else "N/A"
        positions.append(f"#{rank} (ETA {eta})")

    if not positions:
        return f"  You: no pending jobs for this {subject}", False

    shown = positions[:5]
    extra = len(positions) - len(shown)
    suffix = f", +{extra} more" if extra > 0 else ""
    text = f"  You: {len(positions)} pending - " + ", ".join(shown) + suffix
    return text, True


def render_dashboard_view(stdscr, node_data, selected, width, height, show_all, user_summary):
    """Render the main dashboard with node list."""
    row = 0
    W = min(width, DASHBOARD_BOX_WIDTH)

    # ── Banner ──
    total_gpus = sum(nd.config.total_gpus for nd in node_data)
    total_used = sum(nd.used_gpus for nd in node_data)
    free = max(0, total_gpus - total_used)

    if free <= 0:
        fc = curses.color_pair(2) | curses.A_BOLD
    elif free <= 4:
        fc = curses.color_pair(4) | curses.A_BOLD
    else:
        fc = curses.color_pair(3) | curses.A_BOLD

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    inner = W - 2
    border = "═" * inner

    safe_addstr(stdscr, row, 0, f"╔{border}╗", curses.A_BOLD)
    row += 1

    title = "GPU Dashboard"
    mode = "[ALL]" if show_all else "[FAVORITES]"
    title_full = f"  {title} {mode}"
    pad1 = inner - len(title_full) - len(now) - 2
    safe_addstr(stdscr, row, 0, "║", curses.A_BOLD)
    safe_addstr(stdscr, row, 1, f"  {title} ", curses.A_BOLD)
    safe_addstr(stdscr, row, 2 + len(title) + 1, mode, curses.color_pair(4))
    safe_addstr(stdscr, row, 1 + len(title_full) + max(pad1, 1), now, curses.A_DIM)
    safe_addstr(stdscr, row, W - 1, "║", curses.A_BOLD)
    row += 1

    free_str = f"  {free}/{total_gpus} GPUs free"
    jobs_str = format_user_jobs_str(user_summary)
    pad2 = inner - len(free_str) - len(jobs_str)
    safe_addstr(stdscr, row, 0, "║", curses.A_BOLD)
    safe_addstr(stdscr, row, 1, free_str, fc)
    safe_addstr(stdscr, row, 1 + len(free_str) + max(pad2, 1), jobs_str, curses.color_pair(1))
    safe_addstr(stdscr, row, W - 1, "║", curses.A_BOLD)
    row += 1

    safe_addstr(stdscr, row, 0, f"╚{border}╝", curses.A_BOLD)
    row += 1

    banner_rows = row
    avail = height - 2 - banner_rows

    part_lines_per_node = []
    node_heights = []
    for nd in node_data:
        nc = nd.config
        show_parts = display_partitions(nc)
        plines = partition_lines(nc.my_partitions, show_parts, len(nd.pending_jobs), inner)
        part_lines_per_node.append(plines)
        # header + partitions + separator + (header + jobs OR idle line) + bottom + gap
        body = (1 + len(nd.running_jobs)) if nd.running_jobs else 1
        node_heights.append(1 + len(plines) + 1 + body + 1 + 1)

    cumulative = [0]
    for h in node_heights:
        cumulative.append(cumulative[-1] + h)
    first_visible = 0
    while first_visible < len(node_data) and cumulative[selected + 1] - cumulative[first_visible] > avail:
        first_visible += 1

    for idx in range(first_visible, len(node_data)):
        nd = node_data[idx]
        row += 1
        if row >= height - 2:
            break
        is_sel = (idx == selected)
        row = render_node_box(stdscr, nd, row, W, height, is_sel, part_lines_per_node[idx])

    # ── Footer ──
    footer_row = height - 1
    footer = " ↑↓:Nav  Enter:Details  /:Search  s:Settings  a:All  c:CPU  r:Refresh  q:Quit"
    render_footer(stdscr, footer_row, width, footer)


def render_node_box(stdscr, nd, start_row, W, height, is_selected, part_lines=None):
    """Render a single node box. Returns the next row position."""
    nc = nd.config
    row = start_row
    inner = W - 2
    bar = gpu_bar(nd.used_gpus, nc.total_gpus)

    if nd.used_gpus >= nc.total_gpus:
        sc = curses.color_pair(2) | curses.A_BOLD
        st = f"{nd.used_gpus}/{nc.total_gpus} FULL"
    elif nd.used_gpus > 0:
        sc = curses.color_pair(4)
        st = f"{nd.used_gpus}/{nc.total_gpus} used"
    else:
        sc = curses.color_pair(3)
        st = f"{nd.used_gpus}/{nc.total_gpus} used"

    sel_attr = curses.color_pair(7) | curses.A_BOLD if is_selected else 0
    border_attr = curses.A_BOLD if is_selected else 0

    mid = f" ── {nc.total_gpus}x {nc.gpu_label} ── ["
    header_content = f" {nc.name}{mid}{bar}] {st} "
    pad = inner - len(header_content)
    dashes = "─" * max(pad, 0)

    if row < height - 2:
        safe_addstr(stdscr, row, 0, "┌─", border_attr)
        safe_addstr(stdscr, row, 2, nc.name, sel_attr if is_selected else curses.A_BOLD)
        col = 2 + len(nc.name)
        safe_addstr(stdscr, row, col, mid, border_attr)
        col += len(mid)
        safe_addstr(stdscr, row, col, bar, sc)
        col += len(bar)
        safe_addstr(stdscr, row, col, "] ", border_attr)
        col += 2
        safe_addstr(stdscr, row, col, st, sc)
        col += len(st)
        safe_addstr(stdscr, row, col, " " + dashes + "┐", border_attr)
    row += 1

    if part_lines is None:
        show_parts = display_partitions(nc)
        part_lines = partition_lines(nc.my_partitions, show_parts, len(nd.pending_jobs), inner)
    for line in part_lines:
        if row < height - 2:
            safe_addstr(stdscr, row, 0, "│", border_attr)
            safe_addstr(stdscr, row, 1, line[:inner], curses.color_pair(1))
            safe_addstr(stdscr, row, W - 1, "│", border_attr)
        row += 1

    if row < height - 2:
        safe_addstr(stdscr, row, 0, "├" + "─" * inner + "┤", border_attr)
    row += 1

    if nd.running_jobs:
        if row < height - 2:
            safe_addstr(stdscr, row, 0, "│", border_attr)
            safe_addstr(stdscr, row, 1, format_job_header(inner)[:inner], curses.A_DIM)
            safe_addstr(stdscr, row, W - 1, "│", border_attr)
        row += 1
        for j in nd.running_jobs:
            if row >= height - 2:
                break
            safe_addstr(stdscr, row, 0, "│", border_attr)
            safe_addstr(stdscr, row, 1, format_job_row(j, inner)[:inner], user_attr(j))
            safe_addstr(stdscr, row, W - 1, "│", border_attr)
            row += 1
    else:
        if row < height - 2:
            msg = f"  (idle — all {nc.total_gpus} GPUs free)"
            safe_addstr(stdscr, row, 0, "│", border_attr)
            safe_addstr(stdscr, row, 1, msg[:inner], curses.color_pair(3))
            safe_addstr(stdscr, row, W - 1, "│", border_attr)
        row += 1

    if row < height - 2:
        safe_addstr(stdscr, row, 0, "└" + "─" * inner + "┘", border_attr)
    row += 1

    return row


def render_detail_view(stdscr, nd, scroll, width, height):
    """Render the detail view for a single node."""
    nc = nd.config
    W = min(width, DETAIL_BOX_WIDTH)
    inner = W - 2

    lines: List[tuple] = []

    bar = gpu_bar(nd.used_gpus, nc.total_gpus)
    if nd.used_gpus >= nc.total_gpus:
        st = f"{nd.used_gpus}/{nc.total_gpus} FULL"
    elif nd.used_gpus > 0:
        st = f"{nd.used_gpus}/{nc.total_gpus} used"
    else:
        st = f"0/{nc.total_gpus} used"

    title = f" {nc.name} ── {nc.total_gpus}x {nc.gpu_label} ── [{bar}] {st} "
    lines.append((title, curses.A_BOLD))
    lines.append(("", 0))

    # Hardware
    if nc.cpus_per_node or nc.node_ram:
        lines.append(("  HARDWARE", curses.color_pair(9) | curses.A_BOLD))
        lines.append((f"    GPUs:        {nc.total_gpus}x {nc.gpu_label}", 0))
        if nc.cpus_per_node:
            lines.append((f"    CPUs/node:   {nc.cpus_per_node}", 0))
        if nc.node_ram:
            lines.append((f"    System RAM:  {nc.node_ram}", 0))
        lines.append(("", 0))

    # Your partitions
    if nc.my_partitions:
        lines.append(("  YOUR PARTITIONS", curses.color_pair(9) | curses.A_BOLD))
        for p in nc.my_partitions:
            lines.append((part_limit_line(p), curses.color_pair(1) | curses.A_BOLD))
        lines.append(("", 0))

    # Other partitions
    other_parts = [p for p in display_partitions(nc) if p not in nc.my_partitions]
    if other_parts:
        lines.append(("  OTHER PARTITIONS", curses.color_pair(9) | curses.A_BOLD))
        for p in other_parts:
            lines.append((part_limit_line(p), curses.color_pair(1)))
        lines.append(("", 0))

    # Running jobs
    lines.append((f"  RUNNING JOBS ({nd.used_gpus} GPUs used)", curses.color_pair(9) | curses.A_BOLD))
    sep_line = "  " + "─" * (inner - 4)
    lines.append((sep_line, curses.A_DIM))

    if nd.running_jobs:
        lines.append((format_job_header(inner), curses.A_DIM))
        for j in nd.running_jobs:
            lines.append((format_job_row(j, inner), user_attr(j)))
    else:
        lines.append((f"  (idle — all {nc.total_gpus} GPUs free)", curses.color_pair(3)))

    lines.append(("", 0))

    pending = nd.pending_jobs
    shown_pending, hidden_pending = selected_pending_rows(pending)
    lines.append((queue_title(pending, len(shown_pending)), curses.color_pair(9) | curses.A_BOLD))
    lines.append((sep_line, curses.A_DIM))

    if pending:
        lines.append((format_queue_header(inner), curses.A_DIM))
        for rank, j in shown_pending:
            lines.append((format_queue_row(rank, j, inner), user_attr(j)))
        if hidden_pending > 0:
            lines.append((f"  ... {hidden_pending} jobs hidden", curses.A_DIM))
    else:
        lines.append(("  (no pending jobs)", curses.color_pair(3)))

    lines.append(("", 0))

    # User summary
    summary, has_user = my_pending_summary(pending, "node")
    attr = curses.color_pair(1) | curses.A_BOLD if has_user else curses.A_DIM
    lines.append((summary, attr))

    lines.append(("", 0))

    max_scroll = max(0, len(lines) - (height - 4))
    actual_scroll = min(scroll, max_scroll)

    safe_addstr(stdscr, 0, 0, f"┌{'─' * inner}┐", curses.A_BOLD)

    visible = lines[actual_scroll:actual_scroll + height - 4]
    for i, (text, attr) in enumerate(visible):
        row = i + 1
        if row >= height - 2:
            break
        safe_addstr(stdscr, row, 0, "│", curses.A_BOLD)
        safe_addstr(stdscr, row, 1, text[:inner].ljust(inner), attr)
        safe_addstr(stdscr, row, W - 1, "│", curses.A_BOLD)

    bot_row = min(len(visible) + 1, height - 2)
    safe_addstr(stdscr, bot_row, 0, f"└{'─' * inner}┘", curses.A_BOLD)

    if max_scroll > 0:
        pct = int(100 * actual_scroll / max_scroll) if max_scroll else 0
        safe_addstr(stdscr, bot_row, inner - 6, f" {pct}% ", curses.A_DIM)

    footer = " Esc:Back  ↑↓:Scroll"
    render_footer(stdscr, height - 1, width, footer)


# ── CPU Dashboard ──────────────────────────────────────────────────────────

def render_cpu_dashboard_view(stdscr, cluster_data, selected, width, height, show_all, user_summary):
    """Render the CPU cluster dashboard."""
    row = 0
    W = min(width, DASHBOARD_BOX_WIDTH)

    # ── Banner ──
    total_cpus = sum(cd.total_cpus for cd in cluster_data)
    total_used = sum(cd.used_cpus for cd in cluster_data)
    free = max(0, total_cpus - total_used)

    if free <= 0:
        fc = curses.color_pair(2) | curses.A_BOLD
    elif free <= total_cpus * 0.2:
        fc = curses.color_pair(4) | curses.A_BOLD
    else:
        fc = curses.color_pair(3) | curses.A_BOLD

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    inner = W - 2
    border = "═" * inner

    safe_addstr(stdscr, row, 0, f"╔{border}╗", curses.A_BOLD)
    row += 1

    title = "CPU Dashboard"
    mode = "[ALL]" if show_all else "[FAVORITES]"
    title_full = f"  {title} {mode}"
    pad1 = inner - len(title_full) - len(now) - 2
    safe_addstr(stdscr, row, 0, "║", curses.A_BOLD)
    safe_addstr(stdscr, row, 1, f"  {title} ", curses.A_BOLD)
    safe_addstr(stdscr, row, 2 + len(title) + 1, mode, curses.color_pair(4))
    safe_addstr(stdscr, row, 1 + len(title_full) + max(pad1, 1), now, curses.A_DIM)
    safe_addstr(stdscr, row, W - 1, "║", curses.A_BOLD)
    row += 1

    free_str = f"  {free}/{total_cpus} CPUs free"
    jobs_str = format_user_jobs_str(user_summary)
    pad2 = inner - len(free_str) - len(jobs_str)
    safe_addstr(stdscr, row, 0, "║", curses.A_BOLD)
    safe_addstr(stdscr, row, 1, free_str, fc)
    safe_addstr(stdscr, row, 1 + len(free_str) + max(pad2, 1), jobs_str, curses.color_pair(1))
    safe_addstr(stdscr, row, W - 1, "║", curses.A_BOLD)
    row += 1

    safe_addstr(stdscr, row, 0, f"╚{border}╝", curses.A_BOLD)
    row += 1

    banner_rows = row
    avail = height - 2 - banner_rows

    part_lines_per_cluster = []
    cluster_heights = []
    for cd in cluster_data:
        cc = cd.config
        show_parts = display_partitions(cc)
        plines = partition_lines(cc.my_partitions, show_parts, len(cd.pending_jobs), inner)
        part_lines_per_cluster.append(plines)
        if cd.running_jobs:
            shown = min(len(cd.running_jobs), MAX_JOBS_IN_BOX)
            overflow = 1 if len(cd.running_jobs) > MAX_JOBS_IN_BOX else 0
            body = 1 + shown + overflow
        else:
            body = 1
        cluster_heights.append(1 + len(plines) + 1 + body + 1 + 1)

    cumulative = [0]
    for h in cluster_heights:
        cumulative.append(cumulative[-1] + h)
    first_visible = 0
    while first_visible < len(cluster_data) and cumulative[selected + 1] - cumulative[first_visible] > avail:
        first_visible += 1

    for idx in range(first_visible, len(cluster_data)):
        cd = cluster_data[idx]
        row += 1
        if row >= height - 2:
            break
        is_sel = (idx == selected)
        row = render_cpu_cluster_box(stdscr, cd, row, W, height, is_sel, part_lines_per_cluster[idx])

    # ── Footer ──
    footer_row = height - 1
    footer = " ↑↓:Nav  Enter:Details  /:Search  s:Settings  a:All  c:GPU  r:Refresh  q:Quit"
    render_footer(stdscr, footer_row, width, footer)


def render_cpu_cluster_box(stdscr, cd, start_row, W, height, is_selected, part_lines=None):
    """Render a single CPU cluster box. Returns the next row position."""
    cc = cd.config
    row = start_row
    inner = W - 2

    pct = int(100 * cd.used_cpus / cd.total_cpus) if cd.total_cpus > 0 else 0
    bar = cpu_bar(cd.used_cpus, cd.total_cpus)

    if cd.used_cpus >= cd.total_cpus:
        sc = curses.color_pair(2) | curses.A_BOLD
        st = f"{pct}% FULL"
    elif cd.used_cpus > 0:
        sc = curses.color_pair(4)
        st = f"{pct}% used"
    else:
        sc = curses.color_pair(3)
        st = f"{pct}% used"

    sel_attr = curses.color_pair(7) | curses.A_BOLD if is_selected else 0
    border_attr = curses.A_BOLD if is_selected else 0

    info = f"{cc.num_nodes} nodes, {cc.cpus_per_node} CPUs/node, {cc.mem_label} RAM/node"
    header_content = f" {cc.name} ({info}) ── [{bar}] {st} "
    pad = inner - len(header_content)
    dashes = "─" * max(pad, 0)

    if row < height - 2:
        safe_addstr(stdscr, row, 0, "┌─", border_attr)
        safe_addstr(stdscr, row, 2, cc.name, sel_attr if is_selected else curses.A_BOLD)
        col = 2 + len(cc.name)
        rest_hdr = f" ({info}) ── ["
        safe_addstr(stdscr, row, col, rest_hdr, border_attr)
        col += len(rest_hdr)
        safe_addstr(stdscr, row, col, bar, sc)
        col += len(bar)
        safe_addstr(stdscr, row, col, "] ", border_attr)
        col += 2
        safe_addstr(stdscr, row, col, st, sc)
        col += len(st)
        remaining = W - 1 - col
        safe_addstr(stdscr, row, col, " " + "─" * max(remaining - 2, 0) + "┐", border_attr)
    row += 1

    if part_lines is None:
        show_parts = display_partitions(cc)
        part_lines = partition_lines(cc.my_partitions, show_parts, len(cd.pending_jobs), inner)
    for line in part_lines:
        if row < height - 2:
            safe_addstr(stdscr, row, 0, "│", border_attr)
            safe_addstr(stdscr, row, 1, line[:inner], curses.color_pair(1))
            safe_addstr(stdscr, row, W - 1, "│", border_attr)
        row += 1

    if row < height - 2:
        safe_addstr(stdscr, row, 0, "├" + "─" * inner + "┤", border_attr)
    row += 1

    if cd.running_jobs:
        if row < height - 2:
            safe_addstr(stdscr, row, 0, "│", border_attr)
            safe_addstr(stdscr, row, 1, format_job_header(inner, cpu=True)[:inner], curses.A_DIM)
            safe_addstr(stdscr, row, W - 1, "│", border_attr)
        row += 1
        shown = cd.running_jobs[:MAX_JOBS_IN_BOX]
        for j in shown:
            if row >= height - 2:
                break
            safe_addstr(stdscr, row, 0, "│", border_attr)
            safe_addstr(stdscr, row, 1, format_job_row(j, inner, cpu=True)[:inner], user_attr(j))
            safe_addstr(stdscr, row, W - 1, "│", border_attr)
            row += 1
        remaining_jobs = len(cd.running_jobs) - MAX_JOBS_IN_BOX
        if remaining_jobs > 0 and row < height - 2:
            more = f"  ... and {remaining_jobs} more jobs"
            safe_addstr(stdscr, row, 0, "│", border_attr)
            safe_addstr(stdscr, row, 1, more[:inner], curses.A_DIM)
            safe_addstr(stdscr, row, W - 1, "│", border_attr)
            row += 1
    else:
        if row < height - 2:
            msg = f"  (idle — all {cd.total_cpus} CPUs free across {cc.num_nodes} nodes)"
            safe_addstr(stdscr, row, 0, "│", border_attr)
            safe_addstr(stdscr, row, 1, msg[:inner], curses.color_pair(3))
            safe_addstr(stdscr, row, W - 1, "│", border_attr)
        row += 1

    if row < height - 2:
        safe_addstr(stdscr, row, 0, "└" + "─" * inner + "┘", border_attr)
    row += 1

    return row


def render_cpu_detail_view(stdscr, cd, scroll, width, height):
    """Render the detail view for a single CPU cluster."""
    cc = cd.config
    W = min(width, DETAIL_BOX_WIDTH)
    inner = W - 2

    lines = []

    pct = int(100 * cd.used_cpus / cd.total_cpus) if cd.total_cpus > 0 else 0
    bar = cpu_bar(cd.used_cpus, cd.total_cpus)
    st = f"{pct}% FULL" if cd.used_cpus >= cd.total_cpus else f"{pct}% used"

    title = f" {cc.name} ── {cc.num_nodes} nodes, {cc.cpus_per_node} CPUs/node ── [{bar}] {st} "
    lines.append((title, curses.A_BOLD))
    lines.append(("", 0))

    # Utilization summary
    lines.append(("  UTILIZATION", curses.color_pair(9) | curses.A_BOLD))
    lines.append((f"    Total CPUs: {cd.total_cpus} ({cc.cpus_per_node} x {cc.num_nodes} nodes)", 0))
    lines.append((f"    Used CPUs:  {cd.used_cpus} ({pct}%)", 0))
    lines.append((f"    Nodes in use: {cd.nodes_in_use}/{cc.num_nodes}", 0))
    lines.append((f"    Memory/node: {cc.mem_label}", 0))
    lines.append(("", 0))

    if cc.my_partitions:
        lines.append(("  YOUR PARTITIONS", curses.color_pair(9) | curses.A_BOLD))
        for p in cc.my_partitions:
            lines.append((part_limit_line(p), curses.color_pair(1) | curses.A_BOLD))
        lines.append(("", 0))

    other_parts = [p for p in display_partitions(cc) if p not in cc.my_partitions]
    if other_parts:
        lines.append(("  OTHER PARTITIONS", curses.color_pair(9) | curses.A_BOLD))
        for p in other_parts:
            lines.append((part_limit_line(p), curses.color_pair(1)))
        lines.append(("", 0))

    # Running jobs
    lines.append((f"  RUNNING JOBS ({cd.used_cpus} CPUs used)", curses.color_pair(9) | curses.A_BOLD))
    sep_line = "  " + "─" * (inner - 4)
    lines.append((sep_line, curses.A_DIM))

    if cd.running_jobs:
        lines.append((format_job_header(inner, cpu=True), curses.A_DIM))
        for j in cd.running_jobs:
            lines.append((format_job_row(j, inner, cpu=True), user_attr(j)))
    else:
        lines.append((f"  (idle — all {cd.total_cpus} CPUs free across {cc.num_nodes} nodes)", curses.color_pair(3)))

    lines.append(("", 0))

    pending = cd.pending_jobs
    shown_pending, hidden_pending = selected_pending_rows(pending)
    lines.append((queue_title(pending, len(shown_pending)), curses.color_pair(9) | curses.A_BOLD))
    lines.append((sep_line, curses.A_DIM))

    if pending:
        lines.append((format_queue_header(inner, cpu=True), curses.A_DIM))
        for rank, j in shown_pending:
            lines.append((format_queue_row(rank, j, inner, cpu=True), user_attr(j)))
        if hidden_pending > 0:
            lines.append((f"  ... {hidden_pending} jobs hidden", curses.A_DIM))
    else:
        lines.append(("  (no pending jobs)", curses.color_pair(3)))

    lines.append(("", 0))

    # User summary
    summary, has_user = my_pending_summary(pending, "cluster")
    attr = curses.color_pair(1) | curses.A_BOLD if has_user else curses.A_DIM
    lines.append((summary, attr))

    lines.append(("", 0))

    max_scroll = max(0, len(lines) - (height - 4))
    actual_scroll = min(scroll, max_scroll)

    safe_addstr(stdscr, 0, 0, f"┌{'─' * inner}┐", curses.A_BOLD)

    visible = lines[actual_scroll:actual_scroll + height - 4]
    for i, (text, attr) in enumerate(visible):
        row = i + 1
        if row >= height - 2:
            break
        safe_addstr(stdscr, row, 0, "│", curses.A_BOLD)
        safe_addstr(stdscr, row, 1, text[:inner].ljust(inner), attr)
        safe_addstr(stdscr, row, W - 1, "│", curses.A_BOLD)

    bot_row = min(len(visible) + 1, height - 2)
    safe_addstr(stdscr, bot_row, 0, f"└{'─' * inner}┘", curses.A_BOLD)

    if max_scroll > 0:
        pct_s = int(100 * actual_scroll / max_scroll) if max_scroll else 0
        safe_addstr(stdscr, bot_row, inner - 6, f" {pct_s}% ", curses.A_DIM)

    footer = " Esc:Back  ↑↓:Scroll"
    render_footer(stdscr, height - 1, width, footer)


# ── Entry Point ─────────────────────────────────────────────────────────────

def main():
    global FAVORITE_NODES, EXTENDED_NODES, FAVORITE_CPU_CLUSTERS, EXTENDED_CPU_CLUSTERS

    parser = argparse.ArgumentParser(
        description="Interactive GPU/CPU Dashboard for Slurm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Controls: ↑↓ Navigate  Enter Details  /:Search  s:Settings  "
                "a Toggle All  c CPU/GPU  r Refresh  q Quit  Esc Back")
    )
    parser.add_argument(
        '-a', '--all', action='store_true',
        help=('Include extended nodes/clusters from the start. By default the '
              'dashboard shows only your favorites; press "a" inside the '
              'dashboard to toggle. Edit favorites with "s" or by hand in '
              '~/.config/myslurmstatus/settings.json.'),
    )
    args = parser.parse_args()

    settings = load_settings()

    # Discover topology and partition info in parallel — this is all the
    # startup latency the user pays before the screen appears.
    topology, user_access, partition_info = parallel_results(
        discover_topology, fetch_user_access, fetch_partition_info,
    )
    nodes, clusters = topology
    EXTENDED_NODES = nodes
    EXTENDED_CPU_CLUSTERS = clusters

    apply_user_partitions(
        FAVORITE_NODES + EXTENDED_NODES,
        FAVORITE_CPU_CLUSTERS + EXTENDED_CPU_CLUSTERS,
        user_access,
        partition_info,
    )
    EXTENDED_NODES = filter_accessible_items(EXTENDED_NODES)
    EXTENDED_CPU_CLUSTERS = filter_accessible_items(EXTENDED_CPU_CLUSTERS)
    PARTITION_LIMITS.update(partition_info.limits)

    apply_favorites(settings["favorite_nodes"], settings["favorite_clusters"])

    # Default each view to "show all" when the user has no favorites for it
    # yet, so the first-run dashboard isn't empty. Once they pick favorites in
    # the 's' panel, that filter takes effect on next launch.
    gpu_show_all_initial = (args.all or settings["show_all_by_default"]
                            or not settings["favorite_nodes"])
    cpu_show_all_initial = (args.all or settings["show_all_by_default"]
                            or not settings["favorite_clusters"])

    try:
        curses.wrapper(lambda stdscr: curses_main(
            stdscr, gpu_show_all_initial, cpu_show_all_initial, settings))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
