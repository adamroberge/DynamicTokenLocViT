# Research scratch scripts

[← File guide](../docs/repository-guide.md)

This directory preserves small development probes and earlier research routines. They are standalone scripts with local settings; they do not form an automated test suite.

| File | Purpose |
| --- | --- |
| [`test_in1k.py`](test_in1k.py) | Inspect ImageNet dataset loading. |
| [`test_pack.py`](test_pack.py) | Explore tensor concatenation and sequence lengths. |
| [`test_summary_random_seed.py`](test_summary_random_seed.py) | Explore model summaries, parameter values, and random seeds. |
| [`train_one_epoch.py`](train_one_epoch.py) | Earlier training-epoch implementation. |
| [`visualize_only_cls.py`](visualize_only_cls.py) | Class-token attention visualization probe. |
| [`test_one_batch.py`](test_one_batch.py) | One-batch model/training exploration. |
| [`verify_pth.py`](verify_pth.py) | Checkpoint inspection using an original machine-specific path. |

Run these as modules from the repository root, for example `python -m research.test_pack`. Read their settings before use; several rely on original machine paths or earlier versions of the project. The five scripts formerly in `test/` and the two former root inspection scripts are collected here.
