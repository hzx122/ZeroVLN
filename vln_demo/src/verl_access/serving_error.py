"""Compact infrastructure failures shared by tokenization and generation."""
import hashlib
import http.client
import json
import urllib.error


# Catch only around network I/O. ValueError, TypeError and local policy errors
# remain their own failures; a network error is not automatically a timeout.
TRANSPORT_ERRORS = (OSError, urllib.error.URLError, http.client.HTTPException)


class ServingGenerationError(RuntimeError):
    """A failed serving request is infrastructure failure, not a policy reply."""

    def __init__(self, audit):
        self.audit = audit
        super().__init__("serving_generation_error: " + json.dumps(audit, sort_keys=True))


def serving_error(payload, phase, exc=None, *, finish_reason=None, usage=None,
                  emitted_text_chars=0, stream_error=None, failure_kind=None):
    audit = dict(finish_reason=finish_reason, usage=dict(usage or {}),
                 emitted_text_chars=emitted_text_chars, stream_error=stream_error,
                 request_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True,
                     ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                 phase=phase)
    if failure_kind is not None:
        audit['failure_kind'] = failure_kind
    if exc is not None:
        chain, seen, current = [], set(), exc
        while isinstance(current, BaseException) and id(current) not in seen and len(chain) < 8:
            seen.add(id(current))
            chain.append(current)
            current = (current.reason if isinstance(current, urllib.error.URLError)
                       and isinstance(current.reason, BaseException)
                       else current.__cause__ or current.__context__)
        audit.update(error_type=type(exc).__name__,
                     exception_chain=[type(item).__name__ for item in chain],
                     failure_kind=(failure_kind or ('timeout' if any(isinstance(item, TimeoutError) for item in chain)
                                   else 'http' if isinstance(exc, urllib.error.HTTPError)
                                   else 'transport')))
        if isinstance(exc, urllib.error.HTTPError):
            audit['http_status'] = exc.code
        # Do not include HTTP bodies, prompts, images or exception messages
        # supplied by a server; type/phase/status identify the failure safely.
    return ServingGenerationError(audit)
