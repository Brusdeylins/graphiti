"""Entity Type Service for managing entity types in Redis."""

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import redis.asyncio as redis

from config.schema import EntityTypeConfig, EntityTypeFieldConfig, GraphitiConfig

logger = logging.getLogger(__name__)

# Redis key for storing entity types
ENTITY_TYPES_KEY = 'graphiti:entity_types'


class EntityTypeData:
    """Data class for entity type stored in Redis."""

    def __init__(
        self,
        name: str,
        description: str,
        fields: list[dict[str, Any]] | None = None,
        uuid: str | None = None,
        source: str = 'user',
        created_at: str | None = None,
        modified_at: str | None = None,
    ):
        self.uuid = uuid or str(uuid4())
        self.name = name
        self.description = description
        self.fields = fields or []
        self.source = source  # 'config' or 'user'
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.modified_at = modified_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'uuid': self.uuid,
            'name': self.name,
            'description': self.description,
            'fields': self.fields,
            'source': self.source,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'EntityTypeData':
        """Create from dictionary."""
        return cls(
            uuid=data.get('uuid'),
            name=data['name'],
            description=data['description'],
            fields=data.get('fields', []),
            source=data.get('source', 'user'),
            created_at=data.get('created_at'),
            modified_at=data.get('modified_at'),
        )

    @classmethod
    def from_config(cls, config: EntityTypeConfig) -> 'EntityTypeData':
        """Create from config schema."""
        fields = []
        if config.fields:
            for field in config.fields:
                fields.append({
                    'name': field.name,
                    'type': field.type,
                    'required': field.required,
                    'description': field.description,
                })
        return cls(
            name=config.name,
            description=config.description,
            fields=fields,
            source='config',
        )


class EntityTypeService:
    """Service for managing entity types in Redis."""

    def __init__(self):
        """Initialize the entity type service."""
        self._redis: redis.Redis | None = None
        self._config: GraphitiConfig | None = None

    async def initialize(
        self,
        host: str,
        port: int,
        password: str | None = None,
        config: GraphitiConfig | None = None,
    ) -> None:
        """Initialize connection to Redis and seed from config if needed.

        Args:
            host: Redis host
            port: Redis port
            password: Redis password (optional)
            config: GraphitiConfig for seeding entity types
        """
        self._config = config

        # Connect to Redis
        self._redis = redis.Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=True,
        )

        # Test connection
        await self._redis.ping()
        logger.info(f'EntityTypeService connected to Redis at {host}:{port}')

        # Seed from config if entity types exist
        if config and config.graphiti.entity_types:
            await self._seed_from_config(config.graphiti.entity_types)

    async def _seed_from_config(self, config_types: list[EntityTypeConfig]) -> None:
        """Seed entity types from config, adding only new ones.

        Args:
            config_types: List of entity types from config
        """
        existing_types = await self.get_all()
        existing_names = {et.name for et in existing_types}

        added_count = 0
        for config_type in config_types:
            if config_type.name not in existing_names:
                entity_type = EntityTypeData.from_config(config_type)
                await self._save(entity_type)
                added_count += 1
                logger.info(f'Seeded entity type from config: {config_type.name}')

        if added_count > 0:
            logger.info(f'Seeded {added_count} new entity types from config')
        else:
            logger.info('No new entity types to seed from config')

    async def _save(self, entity_type: EntityTypeData) -> None:
        """Save a single entity type to Redis."""
        if not self._redis:
            raise RuntimeError('EntityTypeService not initialized')

        # Get all types, update/add this one, save back
        all_types = await self._load_all_raw()

        # Find and update or append
        found = False
        for i, et in enumerate(all_types):
            if et.get('name') == entity_type.name:
                all_types[i] = entity_type.to_dict()
                found = True
                break

        if not found:
            all_types.append(entity_type.to_dict())

        await self._redis.set(ENTITY_TYPES_KEY, json.dumps(all_types))

    async def _load_all_raw(self) -> list[dict[str, Any]]:
        """Load all entity types as raw dicts from Redis."""
        if not self._redis:
            raise RuntimeError('EntityTypeService not initialized')

        data = await self._redis.get(ENTITY_TYPES_KEY)
        if not data:
            return []
        return json.loads(data)

    async def get_all(self) -> list[EntityTypeData]:
        """Get all entity types.

        Returns:
            List of all entity types
        """
        raw_types = await self._load_all_raw()
        return [EntityTypeData.from_dict(et) for et in raw_types]

    async def get_by_name(self, name: str) -> EntityTypeData | None:
        """Get an entity type by name.

        Args:
            name: Entity type name

        Returns:
            EntityTypeData if found, None otherwise
        """
        all_types = await self.get_all()
        for et in all_types:
            if et.name == name:
                return et
        return None

    async def create(
        self,
        name: str,
        description: str,
        fields: list[dict[str, Any]] | None = None,
    ) -> EntityTypeData:
        """Create a new entity type.

        Args:
            name: Entity type name
            description: Entity type description
            fields: Optional list of field definitions

        Returns:
            The created entity type

        Raises:
            ValueError: If entity type with this name already exists
        """
        existing = await self.get_by_name(name)
        if existing:
            raise ValueError(f'Entity type "{name}" already exists')

        entity_type = EntityTypeData(
            name=name,
            description=description,
            fields=fields or [],
            source='user',
        )
        await self._save(entity_type)
        logger.info(f'Created entity type: {name}')
        return entity_type

    async def update(
        self,
        name: str,
        description: str | None = None,
        fields: list[dict[str, Any]] | None = None,
    ) -> EntityTypeData:
        """Update an existing entity type.

        Args:
            name: Entity type name
            description: New description (optional)
            fields: New fields (optional)

        Returns:
            The updated entity type

        Raises:
            ValueError: If entity type not found
        """
        entity_type = await self.get_by_name(name)
        if not entity_type:
            raise ValueError(f'Entity type "{name}" not found')

        if description is not None:
            entity_type.description = description
        if fields is not None:
            entity_type.fields = fields

        entity_type.modified_at = datetime.now(timezone.utc).isoformat()

        await self._save(entity_type)
        logger.info(f'Updated entity type: {name}')
        return entity_type

    async def delete(self, name: str) -> bool:
        """Delete an entity type.

        Args:
            name: Entity type name

        Returns:
            True if deleted, False if not found
        """
        if not self._redis:
            raise RuntimeError('EntityTypeService not initialized')

        all_types = await self._load_all_raw()
        original_count = len(all_types)

        all_types = [et for et in all_types if et.get('name') != name]

        if len(all_types) == original_count:
            return False

        await self._redis.set(ENTITY_TYPES_KEY, json.dumps(all_types))
        logger.info(f'Deleted entity type: {name}')
        return True

    async def get_as_pydantic_models(self) -> dict[str, type] | None:
        """Get entity types as Pydantic models for Graphiti.

        Returns:
            Dict mapping type names to Pydantic model classes, or None if no types
        """
        from pydantic import Field, create_model

        all_types = await self.get_all()
        if not all_types:
            return None

        custom_types = {}
        for et in all_types:
            field_definitions: dict[str, Any] = {}

            for field in et.fields:
                field_name = field.get('name', '')
                field_type = field.get('type', 'str')
                field_required = field.get('required', False)
                field_desc = field.get('description', '')

                # Map type string to Python type
                python_type = str  # default
                if field_type == 'int':
                    python_type = int
                elif field_type == 'float':
                    python_type = float
                elif field_type == 'bool':
                    python_type = bool

                if field_required:
                    field_definitions[field_name] = (
                        python_type,
                        Field(description=field_desc),
                    )
                else:
                    field_definitions[field_name] = (
                        python_type | None,
                        Field(default=None, description=field_desc),
                    )

            # Create dynamic Pydantic model
            model = create_model(
                et.name,
                __doc__=et.description,
                **field_definitions,
            )
            custom_types[et.name] = model

        return custom_types if custom_types else None

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            logger.info('EntityTypeService connection closed')
