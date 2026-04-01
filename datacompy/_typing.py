"""Arrow typing module."""
from typing import Protocol

# Defining Arrow streamable type compatible with pyarrow streamable objects.
class ArrowStreamable(Protocol):
    def __arrow_c_stream__(self) -> object: ...

class ArrowArrayLike(Protocol):
    def __arrow_c_array__(self) -> object: ...