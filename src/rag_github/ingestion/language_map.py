"""
Maps file extensions to tree-sitter language names and chunking strategies.
"""
from __future__ import annotations

import os

# Map extension -> (tree_sitter_language, chunk_strategy)
# strategy is 'ast' for code, 'text' for markup/config
LANGUAGE_MAP: dict[str, tuple[str | None, str]] = {
    ".py": ("python", "ast"),
    ".js": ("javascript", "ast"),
    ".jsx": ("javascript", "ast"),
    ".ts": ("typescript", "ast"),
    ".tsx": ("tsx", "ast"),
    ".java": ("java", "ast"),
    ".cpp": ("cpp", "ast"),
    ".c": ("c", "ast"),
    ".cc": ("cpp", "ast"),
    ".cxx": ("cpp", "ast"),
    ".h": ("cpp", "ast"),
    ".hpp": ("cpp", "ast"),
    ".go": ("go", "ast"),
    ".rs": ("rust", "ast"),
    ".rb": ("ruby", "ast"),
    ".php": ("php", "ast"),
    ".cs": ("c_sharp", "ast"),
    ".swift": ("swift", "ast"),
    ".kt": ("kotlin", "ast"),
    ".scala": ("scala", "ast"),
    ".html": ("html", "text"),
    ".css": ("css", "text"),
    ".scss": ("scss", "text"),
    ".md": ("markdown", "text"),
    ".json": ("json", "text"),
    ".yaml": ("yaml", "text"),
    ".yml": ("yaml", "text"),
    ".toml": ("toml", "text"),
    ".xml": ("xml", "text"),
    ".sql": ("sql", "ast"),
    ".sh": ("bash", "ast"),
    ".bash": ("bash", "ast"),
    ".dockerfile": ("dockerfile", "text"),
    "dockerfile": ("dockerfile", "text"),
    ".r": ("r", "ast"),
    ".lua": ("lua", "ast"),
    ".dart": ("dart", "ast"),
    ".zig": ("zig", "ast"),
    ".vue": ("vue", "ast"),
    ".svelte": ("svelte", "ast"),
}


def get_language_info(file_path: str) -> tuple[str | None, str]:
    """
    Get the language name and chunk strategy for a given file path.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        A tuple of (language_name, chunk_strategy).
        If the language is unknown, returns (None, "text").
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == "" or ext == ".":
        # Check for files without extension like Dockerfile
        basename = os.path.basename(file_path).lower()
        if basename in LANGUAGE_MAP:
            return LANGUAGE_MAP[basename]
            
    if ext in LANGUAGE_MAP:
        return LANGUAGE_MAP[ext]
        
    return None, "text"
