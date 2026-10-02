"""
Module for cloning and updating GitHub repositories.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import git
from git.exc import GitCommandError

from rag_github.config.settings import get_settings

logger = logging.getLogger(__name__)


def clone_repo(repo_url: str, target_dir: str | None = None, branch: str = "main") -> Path:
    """
    Clone or update a public GitHub repository.
    
    Args:
        repo_url: URL of the GitHub repository.
        target_dir: Directory to clone into. If None, uses settings.CLONE_DIR / repo_name.
        branch: Branch to clone/checkout.
        
    Returns:
        Path to the cloned repository.
        
    Raises:
        ValueError: If the repository URL is invalid.
        RuntimeError: If cloning or updating fails.
    """
    settings = get_settings()
    
    # Parse repo URL to get owner and name
    parsed_url = urlparse(repo_url)
    if not parsed_url.netloc.endswith("github.com"):
        logger.warning(f"URL {repo_url} doesn't look like a github.com URL, but proceeding anyway.")
        
    path_parts = parsed_url.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL: {repo_url}")
        
    owner = path_parts[0]
    name = path_parts[1]
    if name.endswith(".git"):
        name = name[:-4]
        
    repo_name = f"{owner}/{name}"
    logger.info(f"Processing repository: {repo_name}")
    
    # Determine target directory
    if target_dir is None:
        base_dir = Path(settings.clone_dir)
        target_path = base_dir / owner / name
    else:
        target_path = Path(target_dir)
        
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if target_path.exists() and (target_path / ".git").exists():
            # Update existing repository
            logger.info(f"Repository already exists at {target_path}. Pulling latest changes...")
            repo = git.Repo(target_path)
            origin = repo.remotes.origin
            origin.fetch(depth=1)
            repo.git.checkout(branch)
            repo.git.pull("origin", branch)
            logger.info(f"Successfully updated repository at {target_path}")
        else:
            # Clone new repository
            logger.info(f"Cloning repository {repo_url} to {target_path}...")
            # Ensure URL ends with .git for cloning
            clone_url = repo_url if repo_url.endswith(".git") else f"{repo_url}.git"
            
            git.Repo.clone_from(
                clone_url,
                target_path,
                branch=branch,
                depth=1,  # Shallow clone
                single_branch=True
            )
            logger.info(f"Successfully cloned repository to {target_path}")
            
    except GitCommandError as e:
        logger.error(f"Git operation failed for {repo_url}: {e}")
        raise RuntimeError(f"Failed to clone or update repository: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error while processing {repo_url}: {e}")
        raise RuntimeError(f"Unexpected error processing repository: {e}") from e
        
    return target_path
