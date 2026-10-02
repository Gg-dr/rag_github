from __future__ import annotations

import logging
from typing import Any

from neo4j import Driver, GraphDatabase

from rag_github.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.driver: Driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def verify_connectivity(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neo4j connectivity error: {e}")
            return False

    def run_query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        parameters = parameters or {}
        try:
            with self.driver.session() as session:
                result = session.run(cypher, parameters)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error running cypher read query: {e}")
            raise

    def run_write_query(self, cypher: str, parameters: dict[str, Any] | None = None) -> None:
        parameters = parameters or {}
        try:
            with self.driver.session() as session:
                session.run(cypher, parameters)
        except Exception as e:
            logger.error(f"Error running cypher write query: {e}")
            raise

    def batch_write(
        self, cypher: str, batch_params: list[dict[str, Any]], batch_size: int = 500
    ) -> int:
        if not batch_params:
            return 0
        total_processed = 0
        try:
            with self.driver.session() as session:
                for offset in range(0, len(batch_params), batch_size):
                    rows = batch_params[offset : offset + batch_size]
                    result = session.run(cypher, {"rows": rows})
                    summary = result.consume()
                    total_processed += (
                        summary.counters.nodes_created + summary.counters.relationships_created
                    )
            return total_processed
        except Exception:
            logger.exception("Error in Neo4j batch write")
            raise

    def clear_repo(self, repo_name: str) -> None:
        cypher = "MATCH (n {repo_name: $repo_name}) DETACH DELETE n"
        self.run_write_query(cypher, {"repo_name": repo_name})

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def keep_alive(self) -> bool:
        cypher = "MERGE (n:KeepAlive {id: 'ping'}) SET n.last_ping = datetime()"
        try:
            self.run_write_query(cypher)
            return True
        except Exception:
            return False
