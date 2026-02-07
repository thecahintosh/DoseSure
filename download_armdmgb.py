#!/usr/bin/env python3
"""
ARMD-MGB Dataset Downloader (Fixed)
Dataset: https://physionet.org/content/armd-mgb/1.0.0/
"""

import os
import json
import time
import fcntl
import hashlib
import shutil
import subprocess
import re
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class Config:
    base_dir: Path = field(default_factory=lambda: Path(os.getenv("ARMD_BASE", "/workspace/armd-mgb")))
    max_retries: int = 3
    lock_timeout: int = 300
    worker_id: str = field(default_factory=lambda: os.getenv("WORKER_ID", str(os.getpid())))
    files_per_batch: int = 50
    
    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
    
    @property
    def downloads(self): return self.base_dir / "downloads"
    @property
    def cache(self): return self.base_dir / "cache"
    @property
    def locks(self): return self.base_dir / "locks"
    @property
    def checksums_file(self): return self.cache / "checksums.json"
    @property
    def file_queue(self): return self.cache / "file_queue.json"
    @property
    def file_progress(self): return self.cache / "file_progress.json"
    @property
    def discovered_files(self): return self.cache / "discovered_files.json"


cfg = Config()

DATASET = {
    "name": "armd-mgb",
    "version": "1.0.0",
    "base_url": "https://physionet.org/files/armd-mgb/1.0.0/",
}


def setup():
    for d in [cfg.downloads, cfg.cache, cfg.locks]:
        d.mkdir(parents=True, exist_ok=True)


class LockTimeout(Exception):
    pass


@contextmanager
def file_lock(name: str, timeout: int = None):
    timeout = timeout or cfg.lock_timeout
    lock_path = cfg.locks / f"{name}.lock"
    lock_path.touch(exist_ok=True)
    
    fd = open(lock_path, 'w')
    start = time.time()
    
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - start > timeout:
                fd.close()
                raise LockTimeout(f"Lock timeout: {name}")
            time.sleep(0.5)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def get_credentials() -> Tuple[str, str]:
    if "PHYSIONET_USER" in os.environ and "PHYSIONET_PASS" in os.environ:
        return os.environ["PHYSIONET_USER"], os.environ["PHYSIONET_PASS"]
    
    cred_file = Path.home() / ".physionet.json"
    if cred_file.exists():
        with open(cred_file) as f:
            c = json.load(f)
            return c["username"], c["password"]
    
    raise RuntimeError(
        "Set PHYSIONET_USER and PHYSIONET_PASS environment variables"
    )


def file_checksum(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.sha256() if algorithm == "sha256" else hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


def wget_download(url: str, output: Path, user: str, password: str, 
                  quiet: bool = False, verbose: bool = False) -> Tuple[bool, str]:
    """Download file using wget. Returns (success, error_message)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "wget",
        "--user", user,
        "--password", password,
        "--no-check-certificate",
        "--tries", "3",
        "--timeout", "60",
        "--continue",
        "-O", str(output),
        url
    ]
    
    if quiet and not verbose:
        cmd.insert(1, "-q")
    
    if verbose:
        print(f"    CMD: wget -O {output} '{url}'")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else f"Return code {result.returncode}"
            return False, error_msg
        
        if not output.exists():
            return False, "Output file not created"
        
        if output.stat().st_size == 0:
            output.unlink()
            return False, "Empty file downloaded"
        
        return True, ""
        
    except subprocess.TimeoutExpired:
        return False, "Download timed out"
    except Exception as e:
        return False, str(e)


def download_file(url: str, output: Path, quiet: bool = False, verbose: bool = False) -> Tuple[bool, str]:
    user, password = get_credentials()
    return wget_download(url, output, user, password, quiet, verbose)


def check_url_exists(url: str, verbose: bool = False) -> Tuple[bool, int]:
    """Check if URL exists. Returns (exists, http_status_code)."""
    user, password = get_credentials()
    
    if verbose:
        print(f"    Checking: {url}")
    
    try:
        result = subprocess.run(
            ["wget", "--spider", "-S", "--user", user, "--password", password,
             "--no-check-certificate", url],
            capture_output=True, text=True, timeout=30
        )
        
        # Parse HTTP status from stderr
        status_code = 0
        for line in result.stderr.split('\n'):
            if 'HTTP/' in line:
                match = re.search(r'HTTP/\S+\s+(\d+)', line)
                if match:
                    status_code = int(match.group(1))
        
        if verbose:
            print(f"    Status: {status_code}, Return: {result.returncode}")
        
        return result.returncode == 0, status_code
        
    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return False, 0


def discover_files_from_index(url: str, verbose: bool = False) -> List[str]:
    """Get list of files/directories from a PhysioNet directory listing."""
    user, password = get_credentials()
    
    if verbose:
        print(f"  Fetching index: {url}")
    
    try:
        result = subprocess.run(
            ["wget", "-q", "-O", "-", "--user", user, "--password", password,
             "--no-check-certificate", url],
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode != 0:
            if verbose:
                print(f"    Failed to fetch index: {result.stderr}")
            return []
        
        html = result.stdout
        
        if verbose:
            print(f"    Got {len(html)} bytes of HTML")
        
        # Parse href links
        items = []
        pattern = r'href="([^"?][^"]*)"'
        matches = re.findall(pattern, html)
        
        for match in matches:
            # Skip navigation links
            if match.startswith('/') or match.startswith('..') or match.startswith('http'):
                continue
            # Skip PhysioNet UI elements
            if match in ['Name', 'Last modified', 'Size', 'Description']:
                continue
            items.append(match)
        
        if verbose:
            print(f"    Found {len(items)} items: {items[:5]}...")
        
        return list(set(items))  # Remove duplicates
        
    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return []


def discover_all_files(base_url: str, prefix: str = "", verbose: bool = False) -> List[dict]:
    """Recursively discover all files."""
    all_files = []
    
    current_url = f"{base_url}{prefix}"
    if verbose:
        print(f"\nScanning: {current_url}")
    
    items = discover_files_from_index(current_url, verbose)
    
    for item in items:
        # Skip empty or problematic items
        if not item or item.isspace():
            continue
            
        full_path = f"{prefix}{item}"
        full_url = f"{base_url}{full_path}"
        
        if item.endswith('/'):
            # It's a directory, recurse
            if verbose:
                print(f"  [DIR] {full_path}")
            subfiles = discover_all_files(base_url, full_path, verbose)
            all_files.extend(subfiles)
        else:
            # It's a file
            if verbose:
                print(f"  [FILE] {full_path}")
            all_files.append({
                "path": full_path,
                "url": full_url,
            })
    
    return all_files


def probe_dataset(verbose: bool = False):
    """Probe dataset to discover available files."""
    print("=" * 60)
    print("PROBING ARMD-MGB DATASET")
    print("=" * 60)
    
    base_url = DATASET["base_url"]
    print(f"\nBase URL: {base_url}")
    
    # First, check if we can access the base URL
    print("\nChecking access...")
    exists, status = check_url_exists(base_url, verbose)
    print(f"  Base URL accessible: {exists} (HTTP {status})")
    
    if not exists:
        print("\n[ERROR] Cannot access base URL. Check:")
        print("  1. Your credentials are correct")
        print("  2. You have signed the data use agreement")
        print("  3. The URL is correct")
        
        # Try alternative URLs
        alt_urls = [
            "https://physionet.org/files/armd-mgb/1.0.0/",
            "https://physionet.org/content/armd-mgb/1.0.0/",
        ]
        print("\nTrying alternative URLs...")
        for alt in alt_urls:
            exists, status = check_url_exists(alt, verbose)
            print(f"  {alt}: {exists} (HTTP {status})")
        return
    
    print("\nDiscovering files...")
    all_files = discover_all_files(base_url, "", verbose)
    
    if all_files:
        print(f"\n{'=' * 40}")
        print(f"Found {len(all_files)} files:")
        print('=' * 40)
        
        # Group by extension
        by_ext = {}
        for f in all_files:
            ext = Path(f["path"]).suffix.lower() or "[no extension]"
            by_ext.setdefault(ext, []).append(f)
        
        for ext, files in sorted(by_ext.items(), key=lambda x: -len(x[1])):
            print(f"  {ext}: {len(files)} files")
            if verbose:
                for f in files[:3]:
                    print(f"    - {f['path']}")
                if len(files) > 3:
                    print(f"    ... and {len(files) - 3} more")
        
        # Save discovered files
        save_json(cfg.discovered_files, {"files": all_files, "total": len(all_files)})
        print(f"\nSaved to: {cfg.discovered_files}")
        
        # List all files
        print(f"\nAll files:")
        for f in all_files:
            print(f"  {f['path']}")
    else:
        print("\n[WARNING] No files discovered!")
        print("Trying direct file check...")
        
        # Try common file names directly
        common_files = [
            "LICENSE.txt",
            "SHA256SUMS.txt", 
            "README",
            "RECORDS",
        ]
        
        found = []
        for fname in common_files:
            url = f"{base_url}{fname}"
            exists, status = check_url_exists(url, verbose)
            print(f"  {fname}: {exists} (HTTP {status})")
            if exists:
                found.append({"path": fname, "url": url})
        
        if found:
            save_json(cfg.discovered_files, {"files": found, "total": len(found)})
    
    return all_files


def download_all_discovered(verbose: bool = False):
    """Download all discovered files."""
    print("=" * 60)
    print("DOWNLOADING ALL FILES")
    print("=" * 60)
    
    discovered = load_json(cfg.discovered_files)
    files = discovered.get("files", [])
    
    if not files:
        print("\n[ERROR] No files discovered. Run 'probe' first.")
        return
    
    print(f"\nFiles to download: {len(files)}")
    
    succeeded = 0
    failed = 0
    skipped = 0
    
    for i, f in enumerate(files, 1):
        path = f["path"]
        url = f["url"]
        output = cfg.downloads / path
        
        print(f"\n[{i}/{len(files)}] {path}")
        
        # Skip if already exists
        if output.exists() and output.stat().st_size > 0:
            print(f"  [SKIP] Already exists ({output.stat().st_size:,} bytes)")
            skipped += 1
            continue
        
        # Download
        print(f"  URL: {url}")
        success, error = download_file(url, output, quiet=False, verbose=verbose)
        
        if success:
            size = output.stat().st_size
            print(f"  [OK] {size:,} bytes")
            succeeded += 1
        else:
            print(f"  [FAILED] {error}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"COMPLETE: {succeeded} OK, {failed} failed, {skipped} skipped")
    print("=" * 60)


def download_single_file(filepath: str, verbose: bool = False):
    """Download a single file by path."""
    base_url = DATASET["base_url"]
    url = f"{base_url}{filepath}"
    output = cfg.downloads / filepath
    
    print(f"Downloading: {filepath}")
    print(f"URL: {url}")
    print(f"Output: {output}")
    
    # Check if URL exists
    exists, status = check_url_exists(url, verbose)
    print(f"URL exists: {exists} (HTTP {status})")
    
    if not exists:
        print("[ERROR] URL not accessible")
        return False
    
    success, error = download_file(url, output, quiet=False, verbose=verbose)
    
    if success:
        print(f"[OK] Downloaded {output.stat().st_size:,} bytes")
        return True
    else:
        print(f"[FAILED] {error}")
        return False


def download_recursive():
    """Download using wget recursive mode."""
    print("=" * 60)
    print("RECURSIVE DOWNLOAD")
    print("=" * 60)
    
    user, password = get_credentials()
    base_url = DATASET["base_url"]
    
    print(f"\nURL: {base_url}")
    print(f"Output: {cfg.downloads}")
    
    cmd = [
        "wget",
        "-r",                    # Recursive
        "-np",                   # No parent
        "-nH",                   # No host directories
        "--cut-dirs=3",          # Remove /files/armd-mgb/1.0.0/ from path
        "-c",                    # Continue partial downloads
        "-N",                    # Timestamping
        "--user", user,
        "--password", password,
        "--no-check-certificate",
        "-P", str(cfg.downloads),
        "--tries", "3",
        "--timeout", "60",
        "-e", "robots=off",
        "--progress=bar:force",
        base_url
    ]
    
    print(f"\nCommand: wget -r -np -nH --cut-dirs=3 -P {cfg.downloads} {base_url}")
    print("\nStarting download...\n")
    
    try:
        result = subprocess.run(cmd, timeout=86400)
        if result.returncode == 0:
            print("\n[OK] Download complete")
        else:
            print(f"\n[WARNING] wget exited with code {result.returncode}")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] You can resume by running this command again")
    except Exception as e:
        print(f"\n[ERROR] {e}")


def print_status():
    print("=" * 60)
    print("STATUS")
    print("=" * 60)
    
    print(f"\nBase directory: {cfg.base_dir}")
    
    # Discovered files
    print("\nDiscovered:")
    if cfg.discovered_files.exists():
        discovered = load_json(cfg.discovered_files)
        files = discovered.get("files", [])
        print(f"  {len(files)} files found")
        
        by_ext = {}
        for f in files:
            ext = Path(f["path"]).suffix.lower() or "[none]"
            by_ext.setdefault(ext, []).append(f)
        
        for ext, ext_files in sorted(by_ext.items()):
            print(f"    {ext}: {len(ext_files)}")
    else:
        print("  [not probed yet]")
    
    # Downloaded files
    print("\nDownloaded:")
    if cfg.downloads.exists():
        files = [f for f in cfg.downloads.rglob("*") if f.is_file()]
        total_size = sum(f.stat().st_size for f in files)
        print(f"  {len(files)} files, {total_size / 1e6:.1f} MB")
        
        for f in files:
            rel = f.relative_to(cfg.downloads)
            print(f"    {rel} ({f.stat().st_size:,} bytes)")
    else:
        print("  [none]")


def test_download(verbose: bool = True):
    """Test download with a small file."""
    print("=" * 60)
    print("TEST DOWNLOAD")
    print("=" * 60)
    
    user, password = get_credentials()
    print(f"\nCredentials: {user} / {'*' * len(password)}")
    
    base_url = DATASET["base_url"]
    
    # Try to download LICENSE.txt as a test
    test_files = ["LICENSE.txt", "SHA256SUMS.txt", "README"]
    
    for test_file in test_files:
        url = f"{base_url}{test_file}"
        output = cfg.downloads / f"test_{test_file}"
        
        print(f"\n--- Testing: {test_file} ---")
        print(f"URL: {url}")
        
        # Check if exists
        exists, status = check_url_exists(url, verbose)
        print(f"Exists: {exists}, HTTP Status: {status}")
        
        if exists:
            print("Attempting download...")
            success, error = download_file(url, output, quiet=False, verbose=verbose)
            
            if success:
                print(f"[SUCCESS] Downloaded {output.stat().st_size} bytes")
                print(f"Content preview:")
                with open(output, 'r', errors='ignore') as f:
                    content = f.read(500)
                    print(content[:500])
                return True
            else:
                print(f"[FAILED] {error}")
    
    print("\n[ERROR] Could not download any test file")
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ARMD-MGB Downloader")
    parser.add_argument("command", 
        choices=["probe", "download", "recursive", "status", "test", "single"],
        help="Command to run")
    parser.add_argument("-v", "--verbose", action="store_true",
        help="Verbose output")
    parser.add_argument("-f", "--file", default=None,
        help="Single file to download (for 'single' command)")
    parser.add_argument("-d", "--base-dir", default=None,
        help="Base directory")
    
    args = parser.parse_args()
    
    if args.base_dir:
        cfg.base_dir = Path(args.base_dir)
    
    setup()
    
    if args.command == "status":
        print_status()
        return
    
    # Check credentials
    try:
        user, _ = get_credentials()
        print(f"PhysioNet user: {user}")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return
    
    if args.command == "test":
        test_download(args.verbose)
    
    elif args.command == "probe":
        probe_dataset(args.verbose)
    
    elif args.command == "download":
        download_all_discovered(args.verbose)
    
    elif args.command == "recursive":
        download_recursive()
    
    elif args.command == "single":
        if not args.file:
            print("Specify file with -f/--file")
            return
        download_single_file(args.file, args.verbose)


if __name__ == "__main__":
    main()