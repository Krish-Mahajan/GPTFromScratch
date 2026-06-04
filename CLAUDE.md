# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational project building GPT from scratch, structured as Jupyter notebooks. The notebooks are designed to run on a **P5.48xlarge instance** (8x H100 GPUs) in ap-south-1.

## Development Setup

**Primary workflow: VS Code + Remote Jupyter Kernel**

Notebooks are edited locally in VS Code and executed on the P5 via a remote kernel connection. Outputs are saved in the local `.ipynb` file on save — no sync needed for day-to-day work.

1. Start Jupyter on P5 (SSH in first with `ssh-p5-ap-south-1`):
   ```bash
   source /opt/pytorch/bin/activate
   cd ~/mahajak-workspace/GPTFromScratch
   jupyter notebook --no-browser --ip=0.0.0.0 --port=8889
   ```

2. Port forward from Mac (separate terminal):
   ```bash
   export $(ada credentials print --provider conduit --account 199086640399 --role magnus-training-users --format env | xargs) && \
   aws ec2-instance-connect send-ssh-public-key \
     --instance-id i-0d0a09580b9f3a0b7 \
     --instance-os-user ubuntu \
     --ssh-public-key file:///tmp/temp_ec2_key.pub \
     --region ap-south-1 && \
   ssh -i /tmp/temp_ec2_key -L 8889:localhost:8889 -N ubuntu@65.2.170.61
   ```

3. In VS Code: open notebook → Select Kernel → Existing Jupyter Server → `http://127.0.0.1:8889`

Note: Port 8888 is typically in use — use **8889**.

**sync.sh (for bulk file transfers)**

Only needed when pushing new files to P5 or pulling files edited directly on P5.

```bash
./sync.sh push                                           # Push all to P5
./sync.sh pull                                           # Pull all from P5
./sync.sh pull foundations/01_ngram_language_models.ipynb  # Pull a single file
./sync.sh push foundations/02_neural_language_models.ipynb # Push a single file
```

Note: Local folder is `Foundations/` (capitalized), remote is `foundations/` (lowercase).

**SSH key regeneration (after Mac reboot):**
```bash
ssh-keygen -t ed25519 -f /tmp/temp_ec2_key -N "" -q
```

## Key Technical Context

- Target hardware: P5.48xlarge with 8x H100 80GB GPUs
- Remote workspace: `/home/ubuntu/mahajak-workspace/GPTFromScratch/` on the P5 instance
- The P5 instance is shared with other teams — only work inside the mahajak-workspace folder
