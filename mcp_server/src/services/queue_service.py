"""Queue service factory for DB-neutral episode processing.

This module provides the factory function to create the appropriate queue backend
based on configuration, and exports QueueService as an alias for backward compatibility.

Backend selection logic:
1. If queue.backend == "redis" → Redis Streams backend
2. If queue.backend == "memory" → In-Memory backend
3. If queue.backend == "auto" (default):
   - If queue.redis_url is set → Redis Streams
   - If database.provider == "falkordb" → Redis Streams (uses FalkorDB's Redis)
   - Otherwise → In-Memory
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from .queue_backend import QueueBackend
from .queue_memory import InMemoryBackend, InMemoryQueueConfig
from .queue_redis import RedisQueueConfig, RedisStreamsBackend

if TYPE_CHECKING:
    from config.schema import GraphitiConfig

logger = logging.getLogger(__name__)


@dataclass
class QueueConfig:
    """Configuration for queue service.

    This is the unified config that maps to the config.yaml queue section.
    """

    backend: Literal['auto', 'redis', 'memory'] = 'auto'
    redis_url: str | None = None
    consumer_group: str = 'graphiti_workers'
    block_ms: int = 5000
    claim_min_idle_ms: int = 60000
    max_retries: int = 3
    shutdown_timeout: float = 30.0


def create_queue_backend(
    config: 'GraphitiConfig',
    queue_config: QueueConfig | None = None,
) -> QueueBackend:
    """Create the appropriate queue backend based on configuration.

    Args:
        config: Full GraphitiConfig for database provider detection
        queue_config: Optional explicit queue config (uses config.queue if not provided)

    Returns:
        QueueBackend instance (Redis Streams or In-Memory)
    """
    # Use provided queue_config or build from main config
    if queue_config is None:
        queue_config = _extract_queue_config(config)

    backend_type = queue_config.backend
    redis_url: str | None = queue_config.redis_url

    # Auto-detection logic
    if backend_type == 'auto':
        if redis_url:
            backend_type = 'redis'
            logger.info(f'Auto-detected Redis backend from queue.redis_url')
        elif config.database.provider == 'falkordb' and config.database.providers.falkordb:
            # FalkorDB is Redis-based, reuse it for queue
            redis_url = _build_redis_url_from_falkordb(config)
            backend_type = 'redis'
            logger.info(f'Auto-detected Redis backend from FalkorDB database')
        else:
            backend_type = 'memory'
            logger.info(f'Auto-detected In-Memory backend (no Redis available)')

    # Create the appropriate backend
    if backend_type == 'redis':
        if not redis_url:
            raise ValueError('Redis backend requires redis_url configuration')
        redis_cfg = RedisQueueConfig(
            redis_url=redis_url,
            consumer_group=queue_config.consumer_group,
            block_ms=queue_config.block_ms,
            claim_min_idle_ms=queue_config.claim_min_idle_ms,
            max_retries=queue_config.max_retries,
            shutdown_timeout=queue_config.shutdown_timeout,
        )
        logger.info(f'Using Redis Streams backend')
        return RedisStreamsBackend(config=redis_cfg)

    # Default: In-Memory
    memory_cfg = InMemoryQueueConfig(
        shutdown_timeout=queue_config.shutdown_timeout,
    )
    logger.info(f'Using In-Memory backend')
    return InMemoryBackend(config=memory_cfg)


def _extract_queue_config(config: 'GraphitiConfig') -> QueueConfig:
    """Extract QueueConfig from GraphitiConfig.

    Handles both old-style (no queue section) and new-style (with queue section) configs.
    """
    # Check if config has queue attribute (new-style)
    if hasattr(config, 'queue') and config.queue is not None:
        q = config.queue
        return QueueConfig(
            backend=getattr(q, 'backend', 'auto'),
            redis_url=getattr(q, 'redis_url', None),
            consumer_group=getattr(q, 'consumer_group', 'graphiti_workers'),
            block_ms=getattr(q, 'block_ms', 5000),
            claim_min_idle_ms=getattr(q, 'claim_min_idle_ms', 60000),
            max_retries=getattr(q, 'max_retries', 3),
            shutdown_timeout=getattr(q, 'shutdown_timeout', 30.0),
        )

    # Old-style config: no queue section, default to auto
    return QueueConfig(backend='auto')


def _build_redis_url_from_falkordb(config: 'GraphitiConfig') -> str:
    """Build Redis URL from FalkorDB configuration.

    FalkorDB uses Redis as its backend, so we can reuse the connection for queue.
    Handles password authentication properly (Redis uses :password@ format).
    """
    falkor_cfg = config.database.providers.falkordb
    if falkor_cfg is None:
        return 'redis://localhost:6379'

    parsed = urlparse(falkor_cfg.uri)

    # Only add password if not already in URI and password is configured
    if falkor_cfg.password and not parsed.password:
        host = parsed.hostname or 'localhost'
        port = parsed.port or 6379
        return f'redis://:{falkor_cfg.password}@{host}:{port}'

    return falkor_cfg.uri


# Backward compatibility: QueueService as alias for RedisStreamsBackend
# This allows existing code using QueueService to continue working
QueueService = RedisStreamsBackend

# Export all public symbols
__all__ = [
    'QueueConfig',
    'QueueBackend',
    'QueueService',
    'RedisStreamsBackend',
    'InMemoryBackend',
    'create_queue_backend',
]
