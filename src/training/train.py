"""
Training entry point.

This file intentionally contains orchestration only.

The model lives in models/.
The optimizer lives in training/optimizer.py.
The LR schedule lives in training/lr_schedule.py.
DDP setup lives in distributed/.
The dataloader lives in data/.

This is the key repository-design idea:
the training loop coordinates components instead of implementing
every component itself.
"""

# TODO:
# 1. import config
# 2. setup distributed
# 3. create dataloader
# 4. create GPT
# 5. create optimizer
# 6. run training loop
# 7. evaluation/checkpointing
#
# Keep implementation here thin.
