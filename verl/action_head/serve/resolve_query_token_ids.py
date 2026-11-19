#!/usr/bin/env python3
"""Resolve the tokenizer IDs required by the four-query vLLM extension."""
from __future__ import annotations

import argparse

from transformers import AutoTokenizer


def csv(values):
    return ",".join(str(value) for value in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--query-token", required=True,
                        help="An existing single-ID special/reserved token.")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    query_ids = tokenizer.encode(args.query_token, add_special_tokens=False)
    if len(query_ids) != 1:
        raise SystemExit(
            f"query token must map to exactly one existing id, got {query_ids}")
    if args.query_token not in tokenizer.get_added_vocab():
        raise SystemExit(
            "query token must be an existing reserved/added token")
    if tokenizer.eos_token_id is None:
        raise SystemExit("tokenizer has no eos_token_id")
    if query_ids[0] == tokenizer.eos_token_id:
        raise SystemExit("query token must not be the tokenizer EOS token")
    values = {
        "ACTION_HEAD_QUERY_TOKEN_ID": query_ids,
        "ACTION_HEAD_QUERY_EOS_TOKEN_ID": [tokenizer.eos_token_id],
        "ACTION_HEAD_TOOL_END_TOKEN_IDS": tokenizer.encode(
            "</tool_call>", add_special_tokens=False),
        "ACTION_HEAD_RUN_TOOL_TOKEN_IDS": tokenizer.encode(
            "run_action_head", add_special_tokens=False),
    }
    for name, ids in values.items():
        if not ids:
            raise SystemExit(f"{name} resolved to an empty sequence")
        print(f"export {name}={csv(ids)}")


if __name__ == "__main__":
    main()
