"""Document retrieval agent using GitHub API for project documentation."""

import os
import time
from pathlib import Path
from typing import List, Optional
import requests

class DocumentRetrievalAgent:
    """
    Fetch documentation directly from GitHub using the GitHub API (REST),
    with retry logic, subdirectory support, and terminal progress spinner.
    """
    def __init__(self, storage_env_var: str = "DOC_STORAGE_PATH"):
        self.base_storage_dir = Path(os.getenv(storage_env_var, "./docs_downloads"))
        self.base_storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir = self.base_storage_dir  # Will be set per query
        self.github_api = "https://api.github.com"

    def set_storage_subdir(self, subdir_name: Optional[str] = None):
        import datetime
        if not subdir_name:
            subdir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.storage_dir = self.base_storage_dir / subdir_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str) -> str:
        # For now, just return the query (expandable for web search)
        return query

    def download(self, url: str, dest_path: Path) -> bool:
        resp = requests.get(url)
        if resp.status_code == 200:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return True
        return False

    def fetch_github_docs(self, repo: str, path: str, ref: str = "main", subdir_name: Optional[str] = None, retry_limit: int = 3) -> List[str]:
        """
        Recursively fetch all files under a path in a GitHub repo at a given ref/tag.
        Saves to a subdirectory (named or date-based) under base_storage_dir.
        Shows a spinner/progress bar in the terminal.
        """
        self.set_storage_subdir(subdir_name)
        files_downloaded = []
        spinner = ['|', '/', '-', '\\']
        spin_idx = 0
        total_files = 0

        def _fetch_contents(api_url):
            for attempt in range(retry_limit):
                resp = requests.get(api_url)
                if resp.status_code == 200:
                    return resp.json()
                time.sleep(1)
            raise Exception(f"Failed to fetch {api_url} after {retry_limit} attempts.")

        def _recursive_fetch(repo, path, ref):
            nonlocal spin_idx, total_files
            api_url = f"{self.github_api}/repos/{repo}/contents/{path}?ref={ref}"
            entries = _fetch_contents(api_url)
            for entry in entries:
                if entry["type"] == "file":
                    file_url = entry["download_url"]
                    rel_path = Path(entry["path"]).relative_to(path)
                    dest = self.storage_dir / rel_path
                    errors = 0
                    while errors < retry_limit:
                        if self.download(file_url, dest):
                            files_downloaded.append(str(dest))
                            total_files += 1
                            print(f"\r{spinner[spin_idx % len(spinner)]} Downloaded: {total_files}", end="", flush=True)
                            spin_idx += 1
                            break
                        else:
                            errors += 1
                            time.sleep(1)
                    else:
                        print(f"\nFailed to download {entry['path']} after {retry_limit} attempts.")
                elif entry["type"] == "dir":
                    _recursive_fetch(repo, entry["path"], ref)

        _recursive_fetch(repo, path, ref)
        if total_files > 0:
            print(f"\rDownloaded: {total_files} files.{' ' * 20}")
        return files_downloaded

    def search(self, query: str) -> str:
        # For now, just return the query (expandable for web search)
        return query

    def download(self, url: str, dest_path: str) -> bool:
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return True
        return False

    def fetch_github_docs(self, repo: str, path: str, ref: str = "main") -> List[str]:
        """Recursively fetch all files under a path in a GitHub repo at a given ref/tag."""
        api_url = f"{self.github_api}/repos/{repo}/contents/{path}?ref={ref}"
        resp = requests.get(api_url)
        files_downloaded = []
        if resp.status_code == 200:
            for entry in resp.json():
                if entry["type"] == "file":
                    file_url = entry["download_url"]
                    rel_path = Path(entry["path"]).relative_to(path)
                    dest = self.storage_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if self.download(file_url, dest):
                        files_downloaded.append(str(dest))
                elif entry["type"] == "dir":
                    files_downloaded.extend(self.fetch_github_docs(repo, entry["path"], ref))
        return files_downloaded
