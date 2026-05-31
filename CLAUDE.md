# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational project building GPT from scratch, structured as Jupyter notebooks. The notebooks are designed to run on a **P5.48xlarge instance** (8x H100 GPUs) in ap-south-1.

## Development

There is no local build system. Notebooks are authored locally and synced to the P5 instance for execution.

To sync notebooks:
```bash
./sync.sh p5 push    # Push local → P5
./sync.sh p5 pull    # Pull P5 → local
```

## Key Technical Context

- Target hardware: P5.48xlarge with 8x H100 80GB GPUs
- Remote workspace: `/home/ubuntu/mahajak-workspace/GPTFromScratch/` on the P5 instance
- The P5 instance is shared with other teams — only work inside the mahajak-workspace folder
