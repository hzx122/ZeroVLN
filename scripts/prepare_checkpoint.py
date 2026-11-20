"""Expose the released learnable queries to existing ZeroVLN checkpoint readers."""
import argparse
import filecmp
from pathlib import Path


def prepare(root):
    source = root / "learnable_query" / "learnable_query_embeddings.pt"
    runtime_file = root / "dualvln_query" / "dualvln_query_embeddings.pt"
    if not source.is_file():
        if runtime_file.is_file():
            return  # Older checkpoints already use the runtime layout.
        raise FileNotFoundError(source)
    runtime_file.parent.mkdir(exist_ok=True)
    try:
        runtime_file.symlink_to(Path("..") / source.relative_to(root))
    except FileExistsError:
        if not runtime_file.is_file() or not filecmp.cmp(source, runtime_file, shallow=False):
            raise ValueError(f"Conflicting query weights at {runtime_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    prepare(parser.parse_args().checkpoint)
