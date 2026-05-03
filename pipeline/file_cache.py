# pipeline/file_cache.py

import logging
import pickle
from pathlib import Path

from datapizza.core.cache import Cache

log = logging.getLogger(__name__)


class FileCache(Cache):
    """File-based persistent cache using pickle.
    
    Stores values as pickled bytes in {cache_dir}/{key}.pkl. The key is
    already a SHA256 hex digest from datapizza-ai's @cacheable decorator,
    so we use it directly as the filename.
    
    Pickle is used because ClientResponse is a dataclass with nested
    TypedBlocks that don't have a JSON-friendly serializer. Pickle handles
    the full object graph transparently.
    
    Implements the Cache abstract class from datapizza.core.cache.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> object:
        path = self.cache_dir / f"{key}.pkl"
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, OSError, EOFError) as e:
            log.warning(f"Cache read failed for {key}: {e}")
            return None

    def set(self, key: str, value: object) -> None:
        path = self.cache_dir / f"{key}.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
        except (pickle.PicklingError, OSError, TypeError) as e:
            log.warning(f"Cache write failed for {key}: {e}")