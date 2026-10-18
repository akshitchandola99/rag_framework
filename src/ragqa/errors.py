"""Phase 7: our own error types, so callers can tell expected failures from bugs."""


# A document we can't use (empty, corrupted, unsupported). The run skips it and goes on.
class DocumentLoadError(ValueError):
    pass


# The LLM call failed (Ollama down, too slow, or an error status). The question can't be answered.
class GenerationError(RuntimeError):
    pass
