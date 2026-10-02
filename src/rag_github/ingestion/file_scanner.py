"""
Module for scanning cloned repositories and discovering source files.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from rag_github.config.settings import get_settings
from rag_github.parsing.models import FileInfo
from rag_github.ingestion.language_map import get_language_info

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", 
    ".tox", ".eggs", "vendor", ".venv", "venv", "dist", "build", ".idea", 
    ".vscode", ".next", "target"
}

def is_binary(file_path: Path) -> bool:
    """Check if a file is binary by looking for null bytes in the first 8KB."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except Exception:
        # If we can't read it, assume it's binary/unreadable
        return True

def scan_repository(repo_path: Path, repo_name: str) -> list[FileInfo]:
    """
    Walk a cloned repository directory and discover source files.
    
    Args:
        repo_path: Path to the cloned repository.
        repo_name: Name of the repository (e.g., 'owner/repo').
        
    Returns:
        List of FileInfo objects for discovered source files.
    """
    settings = get_settings()
    logger.info(f"Scanning repository {repo_name} at {repo_path}")
    
    if not repo_path.exists() or not repo_path.is_dir():
        logger.error(f"Repository path does not exist or is not a directory: {repo_path}")
        return []
        
    files: List[FileInfo] = []
    total_size = 0
    max_file_size = settings.max_file_size_bytes
    max_repo_size = settings.max_repo_size_bytes
    
    for root, dirs, filenames in os.walk(repo_path):
        # Modify dirs in-place to skip unwanted directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
        
        for filename in filenames:
            file_path = Path(root) / filename
            
            # Skip symlinks
            if file_path.is_symlink():
                continue
                
            try:
                file_size = file_path.stat().st_size
                
                # Skip files exceeding max size
                if file_size > max_file_size:
                    logger.debug(f"Skipping {file_path}: exceeds max file size ({file_size} > {max_file_size})")
                    continue
                    
                # Skip empty files
                if file_size == 0:
                    continue
                    
                # Skip binary files
                if is_binary(file_path):
                    continue
                    
                total_size += file_size
                
                # Check repo size limit
                if total_size > max_repo_size:
                    logger.warning(f"Repository size exceeded maximum ({settings.MAX_REPO_SIZE_MB}MB). Stopping scan.")
                    break
                    
                rel_path = file_path.relative_to(repo_path).as_posix()
                language, strategy = get_language_info(str(file_path))
                
                files.append(
                    FileInfo(
                        file_path=rel_path,
                        repo_name=repo_name,
                        language=language,
                        chunk_strategy=strategy,
                        size_bytes=file_size,
                    )
                )
                
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                
        # Break outer loop if repo size exceeded
        if total_size > max_repo_size:
            break
            
    logger.info(f"Scanned {len(files)} source files (total size: {total_size / 1024 / 1024:.2f} MB)")
    
    # Return sorted list for deterministic results
    return sorted(files, key=lambda f: f.file_path)
