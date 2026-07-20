# Running DeepSeek-R1-Distill-Qwen-7B on the TAU CS Slurm Cluster

A step-by-step runbook for the "Silent Tax" typo project. Follow it top to bottom on a
fresh account and you will end with the model downloaded and a working GPU job that returns
a full chain-of-thought.

Everything here runs on the TAU cluster. Your own laptop is only used to SSH in. Each team
member has their own account, so every path below uses `$USER` and works for anyone.

---

## ⚠️ CRITICAL: `/vol/scratch` is wiped every few days

`/vol/scratch` is **not permanent**. TAU auto-purges files that have not been touched for a
few days. We lost a full setup (model + conda + scripts) this way. So:

- **Keep everything small and precious in HOME** (`~/nlp_project/`): code, datasets, results,
  and a `bootstrap.sh` that can rebuild the rest. Home is 6 GB but persistent.
- **Treat `/vol/scratch` as disposable**: only the conda env and the 15 GB model live there.
  When scratch gets wiped, run `bootstrap.sh` to rebuild it in ~15 min.
- **Put results in HOME (or git) immediately**, never leave them only on scratch.

Durable layout:
```
~/nlp_project/            (HOME - persistent, git-backed)
   bootstrap.sh           rebuilds scratch env + model after a purge
   env.sh                 activates the scratch conda env
   scripts/               ask_local.py, ask_local.sbatch, ...
   data/                  the typo datasets
   results/               reasoning traces (JSONL)
/vol/scratch/$USER/       (DISPOSABLE - purged periodically)
   miniconda/  envs/typo/  hf_cache/ (the 15 GB model)
```

To rehydrate after a purge:
```bash
ssh <user>@slurm-client.cs.tau.ac.il
bash ~/nlp_project/bootstrap.sh      # ~15 min: conda + torch + model
```

---

## 0. Facts you must know first (these caused most of our headaches)

| Thing | Reality on this cluster |
|-------|-------------------------|
| Login host | `slurm-client.cs.tau.ac.il` (password login, or SSH key). It is a **pool**: each login lands on a different physical node, so the SSH host key changes every time. |
| Off campus | You must be on the **TAU VPN** or the host is unreachable. |
| Home quota | Tiny (**6 GB**). The model is 15 GB. **Never** put the model, conda, or caches in home. |
| Big storage | `/vol/scratch/$USER` is writable with lots of space. Put everything there. It can be purged over time, so keep code and results in git/home too. |
| Python | System `python3` has **no venv and is locked** (PEP 668). Use **Miniconda** installed into `/vol/scratch`. |
| Downloader | The HuggingFace `hf_xet` backend **stalls**. Remove it and use plain HTTP. |
| GPUs for students | Only **RTX 2080 Ti (11 GB)** and **Titan Xp (12 GB)**. No 24 GB cards. |
| Titan Xp | **Too old** (compute capability sm_61) for modern PyTorch. Avoid it. Use RTX 2080 Ti only. |
| 7B model size | 15 GB in bf16, does **not** fit on one 11 GB card. Request **2 GPUs**. |
| PyTorch build | Node driver supports up to **CUDA 12.9**. Install torch **cu128**, not the default cu130 (cu130 silently falls back to CPU). |
| Model loading | Reading 15 GB from `/vol/scratch` takes ~8 min. Copying to the node's local `/tmp` first loads in ~4 sec. Always use the local-copy trick. |

---

## 1. Connect to the cluster

Off campus: connect the **TAU VPN** first.

```bash
ssh <your-tau-username>@slurm-client.cs.tau.ac.il
```

Enter your TAU password. The default shell is **tcsh** (prompt ends in `%`). Switch to bash
for everything in this guide:

```bash
bash
```

### If you see "REMOTE HOST IDENTIFICATION HAS CHANGED"

This is normal. The login pool hands you a different node with a different host key. Fix it
on your **laptop** and reconnect:

```bash
ssh-keygen -R slurm-client.cs.tau.ac.il
ssh <your-tau-username>@slurm-client.cs.tau.ac.il
```

---

## 2. One-time setup: storage, Miniconda, environment

Run this whole block once, on the login node, in bash. It installs Miniconda into
`/vol/scratch`, creates the Python env, and points all caches at scratch.

```bash
export STORAGE=/vol/scratch/$USER
export HF_HOME=$STORAGE/hf_cache
export PIP_CACHE_DIR=$STORAGE/pip_cache
export TMPDIR=$STORAGE/tmp
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

# Miniconda into scratch (system python is unusable)
cd $STORAGE
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $STORAGE/miniconda
source $STORAGE/miniconda/etc/profile.d/conda.sh
conda config --add pkgs_dirs $STORAGE/conda_pkgs

# Accept Anaconda channel Terms of Service (required before creating an env)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the project env
conda create -y -p $STORAGE/envs/typo python=3.11
conda activate $STORAGE/envs/typo

# HuggingFace CLI + transformers (torch is installed in the NEXT step, with the right CUDA)
pip install -U "huggingface_hub[cli]" transformers accelerate
```

### Save a reusable activation helper

So you never retype the setup after a reconnect:

```bash
cat > /vol/scratch/$USER/env.sh <<'EOF'
export STORAGE=/vol/scratch/$USER
export HF_HOME=$STORAGE/hf_cache
export TMPDIR=$STORAGE/tmp
source $STORAGE/miniconda/etc/profile.d/conda.sh
conda activate $STORAGE/envs/typo
EOF
```

From now on, after connecting, just run:

```bash
bash
source /vol/scratch/$USER/env.sh
```

---

## 3. Install the correct PyTorch (CUDA 12.8)

The default `pip install torch` pulls a CUDA 13 build that the cluster driver cannot use, so
it silently runs on CPU. Install the cu128 build instead:

```bash
source /vol/scratch/$USER/env.sh
pip install --force-reinstall "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
```

Verify (on the login node it will say `cuda: False`, that is fine, there is no GPU here, we
just check the version):

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# expect: 2.11.0+cu128 12.8
```

---

## 4. Download the model

The model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (open weights, no token needed).
Remove `hf_xet` first because it stalls, then download. This must run on the **login node**
(compute nodes have no internet).

```bash
source /vol/scratch/$USER/env.sh
pip uninstall -y hf_xet
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

It downloads ~15 GB into `$HF_HOME`. It is resumable, so if it drops just run the last line
again.

### If your SSH keeps dropping mid-download

Run it detached so a disconnect cannot kill it, then poll:

```bash
cd /vol/scratch/$USER
setsid nohup hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B > download.log 2>&1 < /dev/null &
# check progress later:
du -sh $HF_HOME/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B
```

### Verify it is complete

```bash
D=$HF_HOME/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B
find $D -name '*.incomplete' -delete          # clean partial files from any stalls
du -sh $D/blobs                               # should be ~15G
```

---

## 4b. Two ways to load the model (and which we use)

There are two ways a job can get the 15 GB model into GPU memory. We measured both.

**Way 1 - load directly from `/vol/scratch` (network drive).**
The job reads the model straight from the shared network storage. The loading code fetches it
in 339 small pieces over the network, which is slow.
- Model load time: **~8 minutes**
- Think of it as reading a book by driving to a library for each page.

**Way 2 - copy to the GPU node's local `/tmp` first, then load (WHAT WE USE).**
At the start of the job, do one bulk copy of the 15 GB onto the disk inside the GPU machine,
then load from there. Local disk is right next to the GPU, so loading is almost instant.
- Model load time: **~4 seconds** (after a one-time bulk copy of ~1 min)
- Think of it as photocopying the whole book once, then reading it instantly.

**Decision: always use Way 2 (node-local).** It is ~120x faster to load. For a job that runs
many questions you copy once and then every question loads instantly, which is the whole point
of the project. `ask_local.sbatch` below implements Way 2. A Way 1 script is shown only so you
can see the difference.

Way 1 (the slow baseline, for reference only) is just `ask_local.sbatch` without the copy
block, loading `MODEL_PATH` set directly to the `/vol/scratch` snapshot path.

---

## 5. The job scripts

Put these two files in `/vol/scratch/$USER/scripts/`. The key trick is the **copy to node-local
`/tmp`** at the start of the job (Way 2 above), which makes model loading ~120x faster
(4 sec vs 8 min).

### `ask_local.py`

```python
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ["MODEL_PATH"]                       # node-local copy
question = open("/vol/scratch/%s/scripts/question.txt" % os.environ["USER"]).read().strip()

print("loading from:", MODEL, flush=True)
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
print("QUESTION:", question, flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
msgs = [{"role": "user", "content": question}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=4096, do_sample=True,
                     temperature=0.6, top_p=0.95)
text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n========== MODEL REASONING + ANSWER ==========\n", flush=True)
print(text, flush=True)
print("\n========== END ==========", flush=True)
```

### `ask_local.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=asklocal
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --cpus-per-task=4
#SBATCH --mem=50000
#SBATCH --time=25
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

source /vol/scratch/$USER/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "node: $(hostname)"

# --- the fast-loading trick: copy model to this node's local disk, then load from there
SRC=$(ls -d /vol/scratch/$USER/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/*/)
DST=/tmp/$USER/dsr1
mkdir -p "$DST"
cp -rL "$SRC"/. "$DST"/
export MODEL_PATH="$DST"

python ask_local.py
rm -rf /tmp/$USER
```

---

## 6. Run a question

```bash
source /vol/scratch/$USER/env.sh
cd /vol/scratch/$USER/scripts
mkdir -p logs

echo 'If 3x + 7 = 22, what is x?' > question.txt
sbatch ask_local.sbatch
```

Check status and read the result:

```bash
squeue -u $USER                                  # ST = PD (pending) then R (running)
tail -f logs/asklocal-<JOBID>.out                # watch it, Ctrl-C to stop watching
```

The output has the model's **reasoning** first, then a `</think>` marker, then the **final
answer**. That reasoning block is the chain-of-thought the project analyses.

---

## 7. Running many questions (the real project)

Do **not** submit one job per question. Every job reloads the model, which wastes almost all
your time. Instead:

1. **Load once, loop many.** One job copies and loads the model a single time, then loops over
   a whole list of questions, writing each result to a file.
2. **Shard across parallel jobs.** Split the dataset into chunks and submit several jobs at once
   (your cap is about **12 GPUs**, and each 7B job uses 2, so up to ~6 jobs in parallel).

The batched pipeline that reads a dataset (MATH-500 / GSM8K / GPQA), builds the typo variants,
and writes reasoning traces to JSONL is the next thing to build on top of `ask_local.py`.

---

## 8. Cluster cheat sheet

```bash
# connect
ssh <user>@slurm-client.cs.tau.ac.il ; bash ; source /vol/scratch/$USER/env.sh

# submit / monitor / cancel
sbatch ask_local.sbatch
squeue -u $USER
scancel <JOBID>
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed          # after it finishes

# your limits
sacctmgr -P show assoc user=$USER format=Account,Partition,QOS   # account=gpu-students
sinfo -p studentkillable -N -o "%N %G %f"                        # GPUs in the partition

# interactive test session (3h), instead of sbatch
srun --account=gpu-students --partition=studentkillable --gpus=2 \
     --constraint=geforce_rtx_2080 --pty bash
```

---

## 9. Troubleshooting (everything we hit)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection timed out` on SSH | Not on TAU VPN | Connect the VPN |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | Login pool, host key differs per node | `ssh-keygen -R slurm-client.cs.tau.ac.il` then reconnect |
| `Ambiguous output redirect` | You are in tcsh, not bash | Run `bash` first |
| `externally-managed-environment` / no venv | System python is locked | Use Miniconda in `/vol/scratch` |
| conda `Terms of Service have not been accepted` | Anaconda channel ToS | Run the two `conda tos accept` commands |
| Download shows `Reconstructing ...` and stalls | `hf_xet` backend | `pip uninstall -y hf_xet`, re-run download |
| Job dies at `du: cannot access '2'` etc. | Remote command ran in tcsh | Wrap remote commands in `bash -l` |
| `cuda: False` in the job | torch cu130 vs driver CUDA 12.9 | Install `torch==2.11.0+cu128` |
| `sm_61 is not compatible` / Titan Xp warning | Titan Xp too old for PyTorch | `--constraint=geforce_rtx_2080` |
| `Requested node configuration is not available` | Asked for a GPU type students cannot get | Use `geforce_rtx_2080` only |
| Out of memory loading model | 15 GB does not fit one 11 GB card | `--gpus=2` |
| Model load takes ~8 min | Reading from network `/vol/scratch` | Copy to `/tmp` first (see `ask_local.sbatch`) |
| Home "Disk quota exceeded" | Something wrote to home | Keep everything under `/vol/scratch/$USER` |

---

## 10. What each teammate needs to do

Each person has their own account and their own `/vol/scratch/$USER`, so everyone runs
sections 1 through 5 once. After that, section 6 onward is the daily workflow. Downloading the
model per person is simplest. If you prefer to share one copy, point `HF_HOME` at a shared
scratch path, but be aware scratch can be purged.
```
