import logging


class HidePromptTokenIds(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if (
            record.name == "vllm.entrypoints.logger"
            and isinstance(record.args, tuple)
            and "prompt_token_ids" in str(record.msg)
            and len(record.args) >= 3
        ):
            args = list(record.args)
            args[2] = "<hidden>"
            record.args = tuple(args)
        return True
