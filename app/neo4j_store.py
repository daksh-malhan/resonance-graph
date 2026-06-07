from __future__ import annotations

import logging
from collections.abc import Iterable

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.config import AppConfig
from app.errors import AppError
from app.models import DownloadResult, RetrievedChunk, RoleCandidate, Transcript, TranscriptChunk
from app.roles import extract_role_candidates

logger = logging.getLogger(__name__)


class Neo4jStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_username, config.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def healthcheck(self) -> None:
        try:
            self.driver.verify_connectivity()
        except ServiceUnavailable as exc:
            raise AppError(
                f"Neo4j is not reachable at {self.config.neo4j_uri}. "
                "Start it with 'docker compose up -d'."
            ) from exc
        except Neo4jError as exc:
            raise AppError(f"Neo4j connectivity check failed: {exc}") from exc

    def setup_schema(self, embedding_dimension: int) -> None:
        self.healthcheck()
        statements = [
            "CREATE CONSTRAINT source_id_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
            (
                "CREATE CONSTRAINT episode_video_id_unique IF NOT EXISTS "
                "FOR (e:Episode) REQUIRE e.video_id IS UNIQUE"
            ),
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            (
                "CREATE CONSTRAINT segment_id_unique IF NOT EXISTS "
                "FOR (s:TranscriptSegment) REQUIRE s.segment_id IS UNIQUE"
            ),
            "CREATE CONSTRAINT person_name_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE INDEX source_url_index IF NOT EXISTS FOR (s:Source) ON (s.url)",
            "CREATE INDEX chunk_video_ordinal_index IF NOT EXISTS FOR (c:Chunk) ON (c.video_id, c.ordinal)",
            "CREATE INDEX role_candidate_role_index IF NOT EXISTS FOR (r:RoleCandidate) ON (r.role)",
        ]
        with self.driver.session() as session:
            for statement in statements:
                session.run(statement)
            session.run(
                f"""
                CREATE VECTOR INDEX {self.config.vector_index_name} IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: $dimensions,
                  `vector.similarity_function`: 'cosine'
                }}}}
                """,
                dimensions=embedding_dimension,
            )
        logger.info("Neo4j schema is ready")

    def ingest_episode(
        self,
        download: DownloadResult,
        transcript: Transcript,
        chunks: list[TranscriptChunk],
    ) -> None:
        if not all(chunk.embedding for chunk in chunks):
            raise AppError("Cannot ingest chunks before embeddings have been generated.")

        source = download.source.model_dump(mode="json")
        episode = download.episode.model_dump(mode="json")
        segments = [segment.model_dump(mode="json") for segment in transcript.segments]
        chunk_payload = [chunk.model_dump(mode="json") for chunk in chunks]
        role_candidates = [
            candidate.model_dump(mode="json")
            for candidate in extract_role_candidates(download, transcript)
        ]

        with self.driver.session() as session:
            session.execute_write(self._merge_source_episode, source, episode)
            session.execute_write(self._clear_episode_transcript, episode["video_id"])
            session.execute_write(self._clear_episode_role_candidates, episode["video_id"])
            session.execute_write(self._merge_segments, episode["video_id"], segments)
            session.execute_write(self._merge_chunks, episode["video_id"], chunk_payload)
            session.execute_write(self._merge_role_candidates, episode["video_id"], role_candidates)

    def upsert_episode_metadata(self, download: DownloadResult) -> None:
        source = download.source.model_dump(mode="json")
        episode = download.episode.model_dump(mode="json")
        role_candidates = [
            candidate.model_dump(mode="json")
            for candidate in extract_role_candidates(download, transcript=None)
        ]
        with self.driver.session() as session:
            session.execute_write(self._merge_source_episode, source, episode)
            session.execute_write(self._clear_episode_role_candidates, episode["video_id"])
            session.execute_write(self._merge_role_candidates, episode["video_id"], role_candidates)

    @staticmethod
    def _merge_source_episode(tx, source: dict, episode: dict) -> None:
        tx.run(
            """
            MERGE (s:Source {id: $source.id})
            SET s.url = $source.url,
                s.kind = $source.kind,
                s.updated_at = datetime()
            MERGE (e:Episode {video_id: $episode.video_id})
            SET e.title = $episode.title,
                e.channel = $episode.channel,
                e.channel_id = $episode.channel_id,
                e.channel_url = $episode.channel_url,
                e.uploader = $episode.uploader,
                e.uploader_id = $episode.uploader_id,
                e.uploader_url = $episode.uploader_url,
                e.creator = $episode.creator,
                e.description = $episode.description,
                e.source_url = $episode.source_url,
                e.duration = $episode.duration,
                e.upload_date = $episode.upload_date,
                e.local_video_path = $episode.local_video_path,
                e.info_json_path = $episode.info_json_path,
                e.transcript_source = $episode.transcript_source,
                e.transcript_status = $episode.transcript_status,
                e.updated_at = datetime()
            MERGE (s)-[:HAS_EPISODE]->(e)
            """,
            source=source,
            episode=episode,
        )

    @staticmethod
    def _clear_episode_transcript(tx, video_id: str) -> None:
        tx.run(
            """
            MATCH (e:Episode {video_id: $video_id})-[:HAS_CHUNK]->(c:Chunk)
            DETACH DELETE c
            """,
            video_id=video_id,
        )
        tx.run(
            """
            MATCH (e:Episode {video_id: $video_id})-[:HAS_SEGMENT]->(ts:TranscriptSegment)
            DETACH DELETE ts
            """,
            video_id=video_id,
        )

    @staticmethod
    def _clear_episode_role_candidates(tx, video_id: str) -> None:
        tx.run(
            """
            MATCH (e:Episode {video_id: $video_id})-[:HAS_ROLE_CANDIDATE]->(r:RoleCandidate)
            DETACH DELETE r
            """,
            video_id=video_id,
        )

    @staticmethod
    def _merge_segments(tx, video_id: str, segments: list[dict]) -> None:
        tx.run(
            """
            MATCH (e:Episode {video_id: $video_id})
            UNWIND $segments AS segment
            MERGE (ts:TranscriptSegment {segment_id: segment.segment_id})
            SET ts.video_id = segment.video_id,
                ts.start_time = segment.start_time,
                ts.end_time = segment.end_time,
                ts.text = segment.text,
                ts.source = segment.source
            MERGE (e)-[:HAS_SEGMENT]->(ts)
            """,
            video_id=video_id,
            segments=segments,
        )

    @staticmethod
    def _merge_chunks(tx, video_id: str, chunks: list[dict]) -> None:
        tx.run(
            """
            MATCH (e:Episode {video_id: $video_id})
            UNWIND $chunks AS chunk
            MERGE (c:Chunk {chunk_id: chunk.chunk_id})
            SET c.video_id = chunk.video_id,
                c.ordinal = chunk.ordinal,
                c.text = chunk.text,
                c.start_time = chunk.start_time,
                c.end_time = chunk.end_time,
                c.segment_ids = chunk.segment_ids,
                c.transcript_source = chunk.transcript_source,
                c.embedding = chunk.embedding,
                c.updated_at = datetime()
            MERGE (e)-[:HAS_CHUNK]->(c)
            WITH c, chunk
            UNWIND chunk.segment_ids AS segment_id
            MATCH (ts:TranscriptSegment {segment_id: segment_id})
            MERGE (c)-[:CONTAINS_SEGMENT]->(ts)
            """,
            video_id=video_id,
            chunks=chunks,
        )

    @staticmethod
    def _merge_role_candidates(tx, video_id: str, candidates: list[dict]) -> None:
        tx.run(
            """
            MATCH (e:Episode {video_id: $video_id})
            UNWIND $candidates AS candidate
            MERGE (p:Person {name: candidate.name})
            SET p.updated_at = datetime()
            MERGE (r:RoleCandidate {
                episode_video_id: $video_id,
                name: candidate.name,
                role: candidate.role,
                evidence_source: candidate.evidence_source
            })
            SET r.confidence = candidate.confidence,
                r.evidence_text = candidate.evidence_text,
                r.updated_at = datetime()
            MERGE (e)-[:HAS_ROLE_CANDIDATE]->(r)
            MERGE (r)-[:REFERS_TO]->(p)
            """,
            video_id=video_id,
            candidates=candidates,
        )

    def vector_search(
        self,
        question_embedding: list[float],
        top_k: int,
        neighbor_window: int = 0,
        video_id: str | None = None,
    ) -> list[RetrievedChunk]:
        with self.driver.session() as session:
            if video_id:
                records = self._vector_search_episode(session, question_embedding, top_k, video_id)
            else:
                records = session.run(
                    """
                    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                    YIELD node, score
                    MATCH (e:Episode)-[:HAS_CHUNK]->(node)
                    MATCH (s:Source)-[:HAS_EPISODE]->(e)
                    CALL {
                        WITH e
                        OPTIONAL MATCH (e)-[:HAS_ROLE_CANDIDATE]->(role:RoleCandidate)-[:REFERS_TO]->(person:Person)
                        WITH collect(CASE WHEN role IS NULL THEN NULL ELSE {
                            name: person.name,
                            role: role.role,
                            confidence: role.confidence,
                            evidence_source: role.evidence_source,
                            evidence_text: role.evidence_text
                        } END) AS raw_role_candidates
                        RETURN [candidate IN raw_role_candidates WHERE candidate.name IS NOT NULL] AS episode_role_candidates
                    }
                    RETURN node.chunk_id AS chunk_id,
                           node.video_id AS video_id,
                           e.title AS episode_title,
                           e.channel AS episode_channel,
                           e.uploader AS episode_uploader,
                           e.creator AS episode_creator,
                           episode_role_candidates AS episode_role_candidates,
                           s.url AS source_url,
                           node.text AS text,
                           node.start_time AS start_time,
                           node.end_time AS end_time,
                           score AS score,
                           node.ordinal AS ordinal,
                           node.transcript_source AS transcript_source
                    ORDER BY score DESC
                    """,
                    index_name=self.config.vector_index_name,
                    top_k=top_k,
                    embedding=question_embedding,
                ).data()

            if neighbor_window > 0 and records:
                records = self._expand_neighbors(session, records, neighbor_window)

        return [
            RetrievedChunk(
                chunk_id=record["chunk_id"],
                video_id=record["video_id"],
                episode_title=record["episode_title"],
                episode_channel=record.get("episode_channel"),
                episode_uploader=record.get("episode_uploader"),
                episode_creator=record.get("episode_creator"),
                episode_role_candidates=_role_candidates_from_record(record),
                source_url=record["source_url"],
                text=record["text"],
                start_time=float(record["start_time"]),
                end_time=float(record["end_time"]),
                score=float(record["score"]),
                transcript_source=record.get("transcript_source") or "local_whisper",
            )
            for record in _dedupe_records(records)
        ]

    @staticmethod
    def _vector_search_episode(
        session,
        question_embedding: list[float],
        top_k: int,
        video_id: str,
    ) -> list[dict]:
        return session.run(
            """
            MATCH (e:Episode {video_id: $video_id})-[:HAS_CHUNK]->(node:Chunk)
            MATCH (s:Source)-[:HAS_EPISODE]->(e)
            WITH e, s, node, vector.similarity.cosine(node.embedding, $embedding) AS score
            CALL {
                WITH e
                OPTIONAL MATCH (e)-[:HAS_ROLE_CANDIDATE]->(role:RoleCandidate)-[:REFERS_TO]->(person:Person)
                WITH collect(CASE WHEN role IS NULL THEN NULL ELSE {
                    name: person.name,
                    role: role.role,
                    confidence: role.confidence,
                    evidence_source: role.evidence_source,
                    evidence_text: role.evidence_text
                } END) AS raw_role_candidates
                RETURN [candidate IN raw_role_candidates WHERE candidate.name IS NOT NULL] AS episode_role_candidates
            }
            RETURN node.chunk_id AS chunk_id,
                   node.video_id AS video_id,
                   e.title AS episode_title,
                   e.channel AS episode_channel,
                   e.uploader AS episode_uploader,
                   e.creator AS episode_creator,
                   episode_role_candidates AS episode_role_candidates,
                   s.url AS source_url,
                   node.text AS text,
                   node.start_time AS start_time,
                   node.end_time AS end_time,
                   score AS score,
                   node.ordinal AS ordinal,
                   node.transcript_source AS transcript_source
            ORDER BY score DESC
            LIMIT $top_k
            """,
            video_id=video_id,
            embedding=question_embedding,
            top_k=top_k,
        ).data()

    def _expand_neighbors(self, session, records: list[dict], neighbor_window: int) -> list[dict]:
        expanded = list(records)
        for record in records:
            neighbor_records = session.run(
                """
                MATCH (e:Episode {video_id: $video_id})-[:HAS_CHUNK]->(c:Chunk)
                MATCH (s:Source)-[:HAS_EPISODE]->(e)
                WHERE c.ordinal >= $start AND c.ordinal <= $end
                CALL {
                    WITH e
                    OPTIONAL MATCH (e)-[:HAS_ROLE_CANDIDATE]->(role:RoleCandidate)-[:REFERS_TO]->(person:Person)
                    WITH collect(CASE WHEN role IS NULL THEN NULL ELSE {
                        name: person.name,
                        role: role.role,
                        confidence: role.confidence,
                        evidence_source: role.evidence_source,
                        evidence_text: role.evidence_text
                    } END) AS raw_role_candidates
                    RETURN [candidate IN raw_role_candidates WHERE candidate.name IS NOT NULL] AS episode_role_candidates
                }
                RETURN c.chunk_id AS chunk_id,
                       c.video_id AS video_id,
                       e.title AS episode_title,
                       e.channel AS episode_channel,
                       e.uploader AS episode_uploader,
                       e.creator AS episode_creator,
                       episode_role_candidates AS episode_role_candidates,
                       s.url AS source_url,
                       c.text AS text,
                       c.start_time AS start_time,
                       c.end_time AS end_time,
                       0.0 AS score,
                       c.ordinal AS ordinal,
                       c.transcript_source AS transcript_source
                ORDER BY c.ordinal
                """,
                video_id=record["video_id"],
                start=max(0, int(record["ordinal"]) - neighbor_window),
                end=int(record["ordinal"]) + neighbor_window,
            ).data()
            expanded.extend(neighbor_records)
        return sorted(expanded, key=lambda item: (item["video_id"], item["ordinal"]))

    def list_episodes(self) -> list[dict]:
        with self.driver.session() as session:
            return session.run(
                """
                MATCH (e:Episode)
                OPTIONAL MATCH (e)-[:HAS_CHUNK]->(c:Chunk)
                RETURN e.video_id AS video_id,
                       e.title AS title,
                       e.channel AS channel,
                       e.channel_id AS channel_id,
                       e.channel_url AS channel_url,
                       e.uploader AS uploader,
                       e.uploader_id AS uploader_id,
                       e.uploader_url AS uploader_url,
                       e.creator AS creator,
                       e.description AS description,
                       e.duration AS duration,
                       e.source_url AS source_url,
                       e.transcript_source AS transcript_source,
                       e.transcript_status AS transcript_status,
                       e.updated_at AS updated_at,
                       count(c) AS chunk_count
                ORDER BY updated_at DESC, title ASC
                """
            ).data()

    def inspect_episode(self, video_id: str) -> dict | None:
        with self.driver.session() as session:
            record = session.run(
                """
                MATCH (e:Episode {video_id: $video_id})
                CALL {
                    WITH e
                    OPTIONAL MATCH (e)-[:HAS_SEGMENT]->(ts:TranscriptSegment)
                    RETURN count(ts) AS segment_count
                }
                CALL {
                    WITH e
                    OPTIONAL MATCH (e)-[:HAS_CHUNK]->(c:Chunk)
                    RETURN count(c) AS chunk_count,
                           min(c.start_time) AS first_chunk_start,
                           max(c.end_time) AS last_chunk_end
                }
                CALL {
                    WITH e
                    OPTIONAL MATCH (e)-[:HAS_ROLE_CANDIDATE]->(role:RoleCandidate)-[:REFERS_TO]->(person:Person)
                    WITH collect(CASE WHEN role IS NULL THEN NULL ELSE {
                        name: person.name,
                        role: role.role,
                        confidence: role.confidence,
                        evidence_source: role.evidence_source,
                        evidence_text: role.evidence_text
                    } END) AS raw_role_candidates
                    RETURN [candidate IN raw_role_candidates WHERE candidate.name IS NOT NULL] AS role_candidates
                }
                RETURN e.video_id AS video_id,
                       e.title AS title,
                       e.channel AS channel,
                       e.channel_id AS channel_id,
                       e.channel_url AS channel_url,
                       e.uploader AS uploader,
                       e.uploader_id AS uploader_id,
                       e.uploader_url AS uploader_url,
                       e.creator AS creator,
                       e.description AS description,
                       e.duration AS duration,
                       e.upload_date AS upload_date,
                       e.source_url AS source_url,
                       e.local_video_path AS local_video_path,
                       e.info_json_path AS info_json_path,
                       e.transcript_source AS transcript_source,
                       e.transcript_status AS transcript_status,
                       role_candidates AS role_candidates,
                       segment_count AS segment_count,
                       chunk_count AS chunk_count,
                       first_chunk_start AS first_chunk_start,
                       last_chunk_end AS last_chunk_end
                """,
                video_id=video_id,
            ).single()
        return dict(record) if record else None

    def reset_database(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Deleted all graph data")

    def graph_overview(self, limit: int = 250, video_id: str | None = None) -> dict:
        limit = max(1, min(limit, 1000))
        if video_id:
            query = """
            MATCH (s:Source)-[:HAS_EPISODE]->(e:Episode {video_id: $video_id})
            OPTIONAL MATCH (e)-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (e)-[:HAS_ROLE_CANDIDATE]->(r:RoleCandidate)-[:REFERS_TO]->(p:Person)
            RETURN s, e, c, r, p
            ORDER BY c.ordinal ASC
            LIMIT $limit
            """
            params = {"video_id": video_id, "limit": limit}
        else:
            query = """
            MATCH (s:Source)-[:HAS_EPISODE]->(e:Episode)
            OPTIONAL MATCH (e)-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (e)-[:HAS_ROLE_CANDIDATE]->(r:RoleCandidate)-[:REFERS_TO]->(p:Person)
            RETURN s, e, c, r, p
            ORDER BY e.updated_at DESC, c.ordinal ASC
            LIMIT $limit
            """
            params = {"limit": limit}

        nodes: dict[str, dict] = {}
        links: dict[tuple[str, str, str], dict] = {}
        with self.driver.session() as session:
            for record in session.run(query, **params):
                source = record.get("s")
                episode = record.get("e")
                chunk = record.get("c")
                role = record.get("r")
                person = record.get("p")
                if source:
                    source_id = f"Source:{source.get('id')}"
                    nodes[source_id] = {
                        "id": source_id,
                        "label": "Source",
                        "title": source.get("url") or source.get("id"),
                        "properties": dict(source),
                    }
                if episode:
                    episode_id = f"Episode:{episode.get('video_id')}"
                    nodes[episode_id] = {
                        "id": episode_id,
                        "label": "Episode",
                        "title": episode.get("title") or episode.get("video_id"),
                        "properties": dict(episode),
                    }
                    if source:
                        links[(source_id, episode_id, "HAS_EPISODE")] = {
                            "source": source_id,
                            "target": episode_id,
                            "type": "HAS_EPISODE",
                        }
                if chunk:
                    chunk_id = f"Chunk:{chunk.get('chunk_id')}"
                    text = chunk.get("text") or ""
                    nodes[chunk_id] = {
                        "id": chunk_id,
                        "label": "Chunk",
                        "title": f"Chunk {chunk.get('ordinal')}: {text[:90]}",
                        "properties": {
                            key: value
                            for key, value in dict(chunk).items()
                            if key != "embedding"
                        },
                    }
                    if episode:
                        links[(episode_id, chunk_id, "HAS_CHUNK")] = {
                            "source": episode_id,
                            "target": chunk_id,
                            "type": "HAS_CHUNK",
                        }
                if role:
                    role_id = (
                        f"RoleCandidate:{role.get('episode_video_id')}:"
                        f"{role.get('role')}:{role.get('name')}:"
                        f"{role.get('evidence_source')}"
                    )
                    nodes[role_id] = {
                        "id": role_id,
                        "label": "RoleCandidate",
                        "title": f"{role.get('role')}: {role.get('name')}",
                        "properties": dict(role),
                    }
                    if episode:
                        links[(episode_id, role_id, "HAS_ROLE_CANDIDATE")] = {
                            "source": episode_id,
                            "target": role_id,
                            "type": "HAS_ROLE_CANDIDATE",
                        }
                if person:
                    person_id = f"Person:{person.get('name')}"
                    nodes[person_id] = {
                        "id": person_id,
                        "label": "Person",
                        "title": person.get("name"),
                        "properties": dict(person),
                    }
                    if role:
                        links[(role_id, person_id, "REFERS_TO")] = {
                            "source": role_id,
                            "target": person_id,
                            "type": "REFERS_TO",
                        }

        return {"nodes": list(nodes.values()), "links": list(links.values())}


def _dedupe_records(records: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    output: list[dict] = []
    for record in records:
        chunk_id = record["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        output.append(record)
    return output


def _role_candidates_from_record(record: dict) -> list[RoleCandidate]:
    candidates: list[RoleCandidate] = []
    for item in record.get("episode_role_candidates") or []:
        if not item or not item.get("name"):
            continue
        candidates.append(RoleCandidate.model_validate(item))
    return candidates
