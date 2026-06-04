# Slurm Guide — dolevur

> Based on BIU Slurm System guide v1.8 (3 Dec 2025), personalized for my accounts and workflows.

---

## Table of Contents

1. [My Partitions](#my-partitions)
2. [All Public Partitions](#all-public-partitions)
3. [SBATCH Template](#sbatch-template)
4. [Quick Examples](#quick-examples)
5. [Resource Allocation Defaults](#resource-allocation-defaults)
6. [Cache & Storage](#cache--storage)
7. [How to Submit a Job](#how-to-submit-a-job)
8. [Interactive Jobs (srun)](#interactive-jobs-srun)
9. [Docker Jobs](#docker-jobs)
10. [inspect_ai / SWE-bench Docker Jobs](#inspect_ai--swe-bench-docker-jobs)
11. [Conda / Venv Jobs](#conda--venv-jobs)
12. [Email Notifications](#email-notifications)
13. [Job Time Limits & Requeue Policy](#job-time-limits--requeue-policy)
14. [Using Jupyter on the Cluster](#using-jupyter-on-the-cluster)
15. [Common Commands](#common-commands)
16. [Pending Job Reasons](#pending-job-reasons)

---

## My Partitions

### Private (high priority — use these first)

| Partition | Node | GPUs | Account | Max Time | Max GPUs |
|---|---|---|---|---|---|
| p_b200_schwartz | dgx-b200-02 | 8× B200 | ug_schwartz | unlimited | unlimited |
| p_rtx_schwartz | mm-lab02 | 8× RTX PRO 6000 | ug_schwartz | unlimited | unlimited |
| p_b200_goldberg | dgx-b200-01 | 8× B200 | ug_goldberg | 4h | 1 |
| p_b200_nlp | dgx-b200-01 | 8× B200 | ug_goldberg | 4h | 4 (shared across NLP group: Tsarfaty, Dagan, Goldberg) |

To check which accounts you belong to:

```bash
sacctmgr list assoc where user=dolevur format=User,Account -nP
```

### How to use private partitions

Always include **both** `--partition` and `--account`:

```bash
#SBATCH --partition=p_b200_schwartz
#SBATCH --account=ug_schwartz
```

```bash
#SBATCH --partition=p_b200_nlp
#SBATCH --account=ug_goldberg
```

---

## All Public Partitions

For all public partitions use `--account=ug_hpc` (or any account you belong to).

### GPU Partitions

| Partition | Node(s) | GPUs | Max Time | Max Jobs/user | Max GPUs/user | Node RAM |
|---|---|---|---|---|---|---|
| B200-4h | dgx-b200-01, dgx-b200-02 | 8× B200 | 4h | 1 | 2 | ~2TB |
| B200-8h | dgx-b200-01 | 8× B200 | 8h | 1 | 2 | ~2TB |
| RTX6000-4h | mm-lab02 | 8× RTX PRO 6000 | 4h | 2 | 2 | ~2TB |
| H200-4h | hpc8h200-01 | 8× H200 | 4h | 2 | 2 | ~2TB |
| H200-12h | hpc8h200-01 | 8× H200 | 12h | 2 | 2 | ~2TB |
| A100-4h | dsiasaf01, dsiuriofir01, hpc2a100-01 | 2–4× A100 | 4h | 2 | 2 | ~512G |
| L4-4h | hpc8l4-01, origin01 | 2–8× L4 | 4h | 2 | 2 | ~512G |
| L4-12h | hpc8l4-01, origin01 | 2–8× L4 | 12h | 2 | 2 | ~512G |
| L40s-4h | quantum5 | 2× L40s | 4h | 2 | 2 | — |
| generic | dsicsgpu[02-09], dsisarit[02,05] | mixed | 4h | 8 | 16 | ~192G |
| generic-48G | dsiaw15 | mixed | 4h | 2 | 2 | ~128G |

### CPU-only Partitions

| Partition | Node(s) | Max Time | Max Jobs/user | Max CPUs/user | Node RAM |
|---|---|---|---|---|---|
| cpu1T-24h | hpccpu01 | 24h | 8 | 48 | 1TB |
| cpu192G-48h | dml[02-25] | 48h | 18 | 72 | 192G |
| cpu512G-48h | hpccpu[02-04], skittles, skittles[01-22] | 48h | 18 | 112 | 512G |

---

## SBATCH Template

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --gres=gpu:<N>
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --mem=<RAM>
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=logs/<name>/train_%j.log
#SBATCH --error=logs/<name>/train_%j.err

# Cache setup (avoid re-downloading models)
export HF_HOME=/private/schwartz-lab/dolevur/cache/huggingface
export TORCH_HOME=/private/schwartz-lab/dolevur/cache/torch

# Your command here
```

Create the log dir before submitting: `mkdir -p logs/<name>`

---

## Quick Examples

**4× B200, long run (schwartz private):**

```bash
#SBATCH --gres=gpu:4
#SBATCH --partition=p_b200_schwartz
#SBATCH --account=ug_schwartz
#SBATCH --mem=300G
#SBATCH --time=150:00:00
```

**1× B200, quick test (public):**

```bash
#SBATCH --gres=gpu:1
#SBATCH --partition=B200-4h
#SBATCH --account=ug_hpc
#SBATCH --mem=80G
#SBATCH --time=4:00:00
```

**8× RTX PRO 6000 (schwartz private):**

```bash
#SBATCH --gres=gpu:8
#SBATCH --partition=p_rtx_schwartz
#SBATCH --account=ug_schwartz
#SBATCH --mem=500G
#SBATCH --time=72:00:00
```

**CPU-only job (e.g., data preprocessing):**

```bash
#SBATCH --partition=cpu512G-48h
#SBATCH --account=ug_hpc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
```

---

## Resource Allocation Defaults

These are the Slurm defaults if you don't specify them:

- **CPUs**: 1 CPU per job. Request more with `--cpus-per-task=N`.
- **Memory**: 16 GB per CPU allocated. Request more with `--mem=<size>[M|G]`.
- **GPUs**: 0 GPUs per job. Request with `--gres=gpu:N`.
- **Time**: Partition default. Override with `--time=HH:MM:SS` (must not exceed partition max).

### Requesting a specific GPU type

If a partition has mixed GPU types, you can request a specific type:

```bash
#SBATCH --gres=gpu:a100:2
```

### --mem is RAM, not VRAM

`--mem` reserves **system RAM** on the node, not GPU memory. VRAM is fixed per GPU and always fully available. RAM guidelines:

| GPUs | Recommended --mem |
|---|---|
| 1 | 80G |
| 2 | 150G |
| 4 | 300G |
| 8 | 500G |

Node total RAM: B200 nodes ~2TB, RTX ~2TB, H200 ~2TB, A100 ~512G, L4 ~512G.

---

## Cache & Storage

Home dir (`~`) has limited space. Store model weights, datasets, and caches on shared lab storage:

```bash
export HF_HOME=/private/schwartz-lab/dolevur/cache/huggingface
export TORCH_HOME=/private/schwartz-lab/dolevur/cache/torch
```

Add these to every sbatch script so downloads persist across jobs.

---

## How to Submit a Job

1. **Connect to a Slurm login server via SSH:**

```bash
ssh slurm-login1.lnx.biu.ac.il
# or: slurm-login2.lnx.biu.ac.il / slurm-login3.lnx.biu.ac.il
```

2. **Submit your job:**

```bash
sbatch myJob.sh
```

3. **Monitor your job:**

```bash
squeue -u dolevur
```

4. **Check output and error files** (paths set by `--output` and `--error`) for progress.

---

## Interactive Jobs (srun)

`srun` runs immediate/interactive commands directly on compute nodes. Use it for quick tests, debugging, or development — **not** for long production runs.

**Run a single command on a compute node:**

```bash
srun --partition=<partition-name> <command>
```

**Start an interactive shell:**

```bash
srun --partition=<partition-name> --pty bash
```

**Interactive shell with 1 GPU for up to 4 hours:**

```bash
srun --partition=B200-4h --account=ug_hpc --gres=gpu:1 --time=04:00:00 --pty bash
```

**Important:**
- Interactive jobs are **not** automatically requeued on timeout.
- When you `exit` the shell, the allocation ends and resources are released.
- Use srun only for short tests/debugging. For production, use `sbatch` with checkpointing.
- Prefer lower-tier partitions (generic, L4, A100) for debugging to avoid blocking high-demand GPUs.

---

## Docker Jobs

This section covers running your **entire job inside a Docker container** — i.e., Slurm launches the container and your code runs within it. (For Python scripts that *spawn* Docker containers internally, see the [inspect_ai section](#inspect_ai--swe-bench-docker-jobs) below.)

### Rules

1. **Run the container in the foreground** — do **not** use `-d` (detached mode).
2. **Name the container `slurm-job-$SLURM_JOB_ID`** for traceability.
3. **Pass Slurm-allocated GPUs** via the `SLURM_JOB_GPUS` environment variable.

### Basic Docker batch script

```bash
#!/bin/bash
#SBATCH --job-name=docker_job
#SBATCH --output=docker_job-%j.out
#SBATCH --error=docker_job-%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

export HF_HOME=/private/schwartz-lab/dolevur/cache/huggingface
export TORCH_HOME=/private/schwartz-lab/dolevur/cache/torch

DockerName=slurm-job-$SLURM_JOB_ID

docker run --name "$DockerName" --rm \
  --gpus "device=${SLURM_JOB_GPUS}" \
  my-image:latest python script.py
```

### GPU passthrough details

- `SLURM_JOB_GPUS` is a comma-separated list of device indices (e.g., `0,2`).
- If your workflow uses `CUDA_VISIBLE_DEVICES`, pass it into the container:

```bash
docker run --name "$DockerName" --rm \
  --gpus "device=${SLURM_JOB_GPUS}" \
  -e CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}" \
  my-image:latest python script.py
```

### Mounting cache directories

To avoid re-downloading models inside the container:

```bash
docker run --name "$DockerName" --rm \
  --gpus "device=${SLURM_JOB_GPUS}" \
  -v /private/schwartz-lab/dolevur/cache:/cache \
  -e HF_HOME=/cache/huggingface \
  -e TORCH_HOME=/cache/torch \
  my-image:latest python script.py
```

---

## inspect_ai / SWE-bench Docker Jobs

This covers the case where your **Python script runs on the host** (in a venv) but **internally spawns Docker containers** — e.g., `inspect_ai` launching SWE-bench sandbox environments.

**Script location:** `/home/nlp/dolevur/CoTControl/CoT-Control-Agent/run_swe_bench.py`

### How it works

1. Your Slurm job runs `python run_swe_bench.py ...` directly on the compute node (no wrapping Docker container).
2. `inspect_ai` internally uses Docker Compose to create isolated sandbox containers for each SWE-bench sample (repo checkout, test execution, etc.). Each sample gets its own container.
3. The sandbox containers do **not** need GPUs — they're for code editing and testing. LLM inference happens via API calls (OpenRouter, Together AI) or a local vLLM server.

### Container naming — does NOT follow BIU convention

inspect_ai names its containers with an `inspect-task-` prefix (e.g., `inspect-task-ielnkhh-default-1`), **not** the `slurm-job-$SLURM_JOB_ID` convention from the BIU guide. This is fine because the BIU naming rule is for jobs where Slurm launches the container directly. inspect_ai spawns containers *from within* a running Slurm job, and the cluster doesn't enforce naming on those.

**The risk is orphaned containers.** If your Slurm job gets killed (time limit, `scancel`, OOM), inspect_ai may not get a chance to clean up its containers. Always add a cleanup trap (see below).

### Controlling container concurrency

`inspect_ai` can spawn many Docker containers in parallel. Control this with:

```bash
export INSPECT_MAX_SANDBOXES=8   # Max concurrent Docker containers (default: 2 * cpu_count)
```

When `max_sandboxes` is applied, it also acts as a cap on `max_samples`. Keep it ≤ your `--max-connections` / `--max-samples` flags.

### Cleanup: trap + manual

Always add a cleanup trap in your sbatch script to handle job termination:

```bash
# Cleanup trap — runs on EXIT, TERM (scancel), and USR1 (Slurm time limit preemption)
cleanup() {
    echo "🧹 Cleaning up inspect_ai Docker containers..."
    inspect sandbox cleanup docker 2>/dev/null || true
    docker container prune -f 2>/dev/null || true
    docker system prune -f --volumes 2>/dev/null || true
}
trap cleanup EXIT TERM USR1
```

If containers are ever left behind (e.g., after a crash), clean up manually:

```bash
# List inspect_ai containers
docker ps -a --filter "name=inspect-task"

# Clean up all inspect_ai sandbox containers
inspect sandbox cleanup docker

# Or nuclear option
docker container prune -f
docker system prune -f --volumes
```

### Example: SWE-bench with remote API (CPU partition)

When using OpenRouter/Together AI for inference, no GPU is needed — use a CPU partition:

```bash
#!/bin/bash
#SBATCH --job-name=swe-bench-api
#SBATCH --partition=cpu512G-48h
#SBATCH --account=ug_hpc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/swe-bench/run_%j.log
#SBATCH --error=logs/swe-bench/run_%j.err

# Cache
export HF_HOME=/private/schwartz-lab/dolevur/cache/huggingface
export TORCH_HOME=/private/schwartz-lab/dolevur/cache/torch

# API keys
export OPENROUTER_API_KEY="..."
export TOGETHER_API_KEY="..."

# Control Docker sandbox concurrency
export INSPECT_MAX_SANDBOXES=8

# Cleanup trap
cleanup() {
    echo "🧹 Cleaning up Docker containers..."
    inspect sandbox cleanup docker 2>/dev/null || true
    docker container prune -f 2>/dev/null || true
    docker system prune -f --volumes 2>/dev/null || true
}
trap cleanup EXIT TERM USR1

# Activate venv
source ~/venvs/inspect/bin/activate
cd /home/nlp/dolevur/CoTControl/CoT-Control-Agent

# Run
python run_swe_bench.py \
  --model openrouter/openai/gpt-oss-120b \
  --reasoning-tokens 25000 \
  --max-connections 8 \
  --max-samples 8 \
  --batch-size 20 \
  --limit 100 \
  --auto-continue \
  --max-retries 3 \
  --sample-timeout 1800 \
  --output-folder ./logs/swe-bench-run \
  --mode baseline
```

### Example: SWE-bench with local vLLM (GPU partition)

To serve a model locally via vLLM and point inspect_ai at it, you need a GPU partition. The approach: start vLLM as a background process, wait for it to be ready, then run the evaluation.

```bash
#!/bin/bash
#SBATCH --job-name=swe-bench-vllm
#SBATCH --partition=p_b200_schwartz
#SBATCH --account=ug_schwartz
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --time=72:00:00
#SBATCH --output=logs/swe-bench/vllm_%j.log
#SBATCH --error=logs/swe-bench/vllm_%j.err

# Cache
export HF_HOME=/private/schwartz-lab/dolevur/cache/huggingface
export TORCH_HOME=/private/schwartz-lab/dolevur/cache/torch

# Docker sandbox concurrency
export INSPECT_MAX_SANDBOXES=8

# Cleanup trap — kill vLLM and clean up Docker containers
cleanup() {
    echo "🧹 Shutting down vLLM and cleaning up..."
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null
    inspect sandbox cleanup docker 2>/dev/null || true
    docker container prune -f 2>/dev/null || true
    docker system prune -f --volumes 2>/dev/null || true
}
trap cleanup EXIT TERM USR1

# Activate venv (should have vllm + inspect_ai installed)
source ~/venvs/inspect/bin/activate

# Start vLLM server in background
VLLM_PORT=8000
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # or whatever model

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --port $VLLM_PORT \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --trust-remote-code \
  --dtype bfloat16 \
  &
VLLM_PID=$!

# Wait for vLLM to be ready (poll the health endpoint)
echo "⏳ Waiting for vLLM server to start on port $VLLM_PORT..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "✅ vLLM server ready after ~${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ vLLM process died"
        exit 1
    fi
    sleep 5
done

# Verify vLLM is running
if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "❌ vLLM failed to start within timeout"
    exit 1
fi

# Run SWE-bench pointing at local vLLM
cd /home/nlp/dolevur/CoTControl/CoT-Control-Agent

python run_swe_bench.py \
  --model "openai/$MODEL_NAME" \
  --api-base "http://localhost:$VLLM_PORT/v1" \
  --reasoning-tokens 25000 \
  --max-connections 8 \
  --max-samples 8 \
  --batch-size 20 \
  --limit 100 \
  --auto-continue \
  --max-retries 3 \
  --sample-timeout 1800 \
  --output-folder ./logs/swe-bench-vllm-run \
  --mode baseline
```

**Notes on vLLM + inspect_ai setup:**
- `--api-base` tells `run_swe_bench.py` to use the local vLLM endpoint instead of OpenRouter/Together. The script's `api_base` parameter sets `model_base_url` for inspect_ai.
- `--model "openai/$MODEL_NAME"` uses the OpenAI-compatible provider in inspect_ai, which works with vLLM's OpenAI-compatible API.
- The vLLM server uses the Slurm-allocated GPUs for inference. The Docker sandbox containers spawned by inspect_ai do NOT need GPU access — they only run bash/python for code editing.
- `--tensor-parallel-size` should match `--gres=gpu:N`.
- The cleanup trap ensures vLLM is killed when the Slurm job ends.

### Key considerations

- **Docker must be available on the compute node.** The login servers may not have Docker; the compute nodes should.
- **Disk space**: SWE-bench Docker images can be large. inspect_ai pulls repo images per sample. The script calls `docker system prune -f --volumes` and `docker container prune -f` between batches to reclaim space.
- **Auto-continue with checkpointing**: The script's `--auto-continue` flag automatically resumes from where it left off by scanning the output folder for completed eval files. This works well with Slurm's requeue policy — if the job hits the time limit, resubmit and it will pick up where it left off.
- **Requeue safety**: Because `--auto-continue` scans for completed samples, you can safely let jobs hit the partition time limit. On resubmission, the script skips already-completed instances. The cleanup trap handles Docker containers.
- **`inspect sandbox cleanup docker`**: This is inspect_ai's built-in command to remove all sandbox containers. Use it if you see orphaned `inspect-task-*` containers via `docker ps -a`.

---

## Conda / Venv Jobs

### Conda

```bash
#!/bin/bash
#SBATCH --job-name=conda_job
#SBATCH --output=conda_job_%j.out
#SBATCH --error=conda_job_%j.err
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

export HF_HOME=/private/schwartz-lab/dolevur/cache/huggingface
export TORCH_HOME=/private/schwartz-lab/dolevur/cache/torch

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
python script.py
```

### Venv (my typical workflow)

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/train/train_%j.log
#SBATCH --error=logs/train/train_%j.err
#SBATCH --partition=p_b200_schwartz
#SBATCH --account=ug_schwartz
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --time=150:00:00

export HF_HOME=/private/schwartz-lab/dolevur/cache/huggingface
export TORCH_HOME=/private/schwartz-lab/dolevur/cache/torch

source ~/venvs/myenv/bin/activate
python train.py
```

---

## Email Notifications

Add these to your sbatch script to get email alerts:

```bash
#SBATCH --mail-user=your.email@biu.ac.il
#SBATCH --mail-type=ALL    # Options: ALL, BEGIN, END, FAIL
```

---

## Job Time Limits & Requeue Policy

- If you don't specify `--time`, the job runs up to the **partition's default time limit**.
- If you set `--time`, it must **not exceed** the partition's maximum.
- When a job reaches the partition time limit, it is **automatically suspended and requeued**.

### Checkpointing

To benefit from automatic requeue, implement checkpointing:

- Save progress periodically (e.g., model checkpoints every N steps).
- On restart, resume from the latest checkpoint.
- **Without checkpointing, the requeued job starts from the beginning.**

This is especially important for long training runs on time-limited public partitions. For the SWE-bench script, `--auto-continue` handles this automatically by scanning for completed samples.

---

## Using Jupyter on the Cluster

### 1. Install Jupyter locally (one-time)

```bash
pip3 install notebook
```

### 2. Start Jupyter on a compute node

SSH into a login server, then:

```bash
srun --partition=<partition> --gres=gpu:1 --time=02:00:00 jupyter-notebook --no-browser --ip=0.0.0.0 &
```

Note the **node name** and **port number** from the output, and copy the URL:

```
http://nodename:8888/tree?token=e34d73f70fa...
http://127.0.0.1:8888/tree?token=e34d73f70fa...
```

### 3. Create an SSH tunnel (on your local PC)

```bash
ssh -L <port>:localhost:<port> -J dolevur@slurm-login1.lnx.biu.ac.il dolevur@<node>
```

Replace `<port>` with the port from step 2 (e.g., 8888) and `<node>` with the assigned compute node.

### 4a. Use in browser

Paste the second URL (`http://127.0.0.1:8888/tree?token=...`) into your local browser.

### 4b. Use in VS Code

1. Install the Microsoft **Python** and **Jupyter** extensions (no Remote-SSH needed).
2. `Ctrl+Shift+P` → "Jupyter: Create New Jupyter Notebook".
3. Click "Select Kernel" (top-right) → "Existing Jupyter Server…".
4. Paste the Jupyter URL (e.g., `http://localhost:8888/?token=abcd1234...`).

VS Code connects to the Jupyter kernel running on the Slurm compute node.

### Important

- After debugging, submit actual production jobs via `sbatch`.
- Use lower-tier partitions (generic, L4) for interactive Jupyter sessions to avoid blocking high-demand resources.
- `--time=HH:MM:SS` ensures the job stops automatically when the limit is reached.

---

## Common Commands

```bash
sbatch script.sh              # Submit a job
squeue -u dolevur             # My running/pending jobs
scancel <jobid>               # Cancel one job
scancel -u dolevur            # Cancel all my jobs
sinfo                         # View available partitions
myslurmstatus                 # Interactive GPU dashboard (custom tool)
myslurmstatus -a              # Include public nodes
```

---

## Pending Job Reasons

| Reason | Meaning |
|---|---|
| Resources | Waiting for GPUs/CPUs to free up — you're in line |
| Priority | Other jobs have higher priority, waiting your turn |
| QOSMaxJobsPerUser | You hit the max concurrent jobs limit for this partition |