import os
import urllib.request
from typing import Optional

from colorama import Fore, Style


def get_or_create_file(
    fname: str,
    url: str,
    cache_dir: str = "outputs/disco_rl/weights",
    filetype: Optional[str] = None,
) -> str:
    """Download a file if not cached and return local path.

    Args:
        fname: Target filename (e.g., 'disco_103.npz').
        url: Direct download URL.
        cache_dir: Directory to store cached weights.
        filetype: Optional expected extension (e.g., 'npz') for a sanity check.

    Returns:
        Local filesystem path to the file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, fname)
    if os.path.exists(path):
        print(f"{Fore.GREEN}{Style.BRIGHT}Using cached weights at {path}.{Style.RESET_ALL}")
        return path

    print(f"{Fore.YELLOW}{Style.BRIGHT}Downloading {fname} from {url}...{Style.RESET_ALL}")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

    if filetype is not None and not fname.endswith(f".{filetype}"):
        raise ValueError(f"Expected filetype .{filetype} for {fname}")

    print(f"{Fore.GREEN}{Style.BRIGHT}Saved weights to {path}.{Style.RESET_ALL}")
    return path
