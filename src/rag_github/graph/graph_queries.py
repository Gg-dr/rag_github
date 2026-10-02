from __future__ import annotations

from typing import List, Dict
from rag_github.graph.neo4j_client import Neo4jClient

class GraphQueries:
    """Pre-built Cypher queries for retrieval-time graph traversal."""

    @staticmethod
    def find_callers(client: Neo4jClient, function_name: str, repo_name: str) -> List[Dict]:
        cypher = '''
        MATCH (caller:Symbol)-[:CALLS]->(callee:Symbol {name: $func_name, repo_name: $repo_name})
        RETURN caller.qualified_name AS caller, caller.path AS path
        '''
        return client.run_query(cypher, {"func_name": function_name, "repo_name": repo_name})

    @staticmethod
    def find_callees(client: Neo4jClient, function_name: str, repo_name: str) -> List[Dict]:
        cypher = '''
        MATCH (caller:Symbol {name: $func_name, repo_name: $repo_name})-[:CALLS]->(callee:Symbol)
        RETURN callee.qualified_name AS callee, callee.path AS path
        '''
        return client.run_query(cypher, {"func_name": function_name, "repo_name": repo_name})

    @staticmethod
    def find_class_hierarchy(client: Neo4jClient, class_name: str, repo_name: str) -> List[Dict]:
        cypher = '''
        MATCH p=(child:Symbol {name: $cls_name, repo_name: $repo_name})-[:INHERITS*]->(parent:Symbol)
        RETURN nodes(p) AS hierarchy
        '''
        return client.run_query(cypher, {"cls_name": class_name, "repo_name": repo_name})

    @staticmethod
    def find_file_symbols(client: Neo4jClient, file_path: str, repo_name: str) -> List[Dict]:
        cypher = '''
        MATCH (f:File {file_path: $path, repo_name: $repo_name})-[:CONTAINS]->(s:Symbol)
        RETURN s.name AS name, s.kind AS type, s.start_line AS start_line
        '''
        return client.run_query(cypher, {"path": file_path, "repo_name": repo_name})

    @staticmethod
    def find_related_symbols(client: Neo4jClient, symbol_name: str, repo_name: str, max_depth: int = 2) -> List[Dict]:
        cypher = '''
        MATCH (s:Symbol {name: $name, repo_name: $repo_name})-[*1..2]-(related:Symbol)
        RETURN DISTINCT related.qualified_name AS related, related.kind AS type,
               related.file_path AS path, related.name AS name
        '''
        return client.run_query(cypher, {"name": symbol_name, "repo_name": repo_name})

    @staticmethod
    def find_co_referenced(client: Neo4jClient, symbol_name: str, repo_name: str) -> List[Dict]:
        cypher = '''
        MATCH (s1:Symbol {name: $name, repo_name: $repo_name})<-[:CALLS]-(common)-[:CALLS]->(s2:Symbol)
        WHERE s1 <> s2
        RETURN s2.qualified_name AS co_referenced, count(common) AS frequency
        ORDER BY frequency DESC
        '''
        return client.run_query(cypher, {"name": symbol_name, "repo_name": repo_name})

    @staticmethod
    def get_repo_stats(client: Neo4jClient, repo_name: str) -> Dict:
        cypher = '''
        MATCH (n {repo_name: $repo_name})
        RETURN labels(n) AS type, count(n) AS count
        '''
        results = client.run_query(cypher, {"repo_name": repo_name})
        stats = {}
        for r in results:
            labels = r.get("type", [])
            for label in labels:
                stats[label] = stats.get(label, 0) + r.get("count", 0)
        return stats
