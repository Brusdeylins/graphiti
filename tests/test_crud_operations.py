"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Integration tests for Graphiti CRUD operations.

Tests the new CRUD methods on the Graphiti class:
- Entity: create, get, update, remove
- Edge: create, get, update, remove
- Episode: get, get_episodes_by_group_id
- Group: get_groups, rename_group, remove_group, get_graph_stats

Run with: pytest tests/test_crud_operations.py -v
"""

import os

import pytest
import pytest_asyncio

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError

pytestmark = pytest.mark.integration
pytest_plugins = ('pytest_asyncio',)

# Test configuration - uses running FalkorDB container
FALKORDB_HOST = os.environ.get('FALKORDB_HOST', 'localhost')
FALKORDB_PORT = int(os.environ.get('FALKORDB_PORT', '6379'))
FALKORDB_PASSWORD = os.environ.get('FALKORDB_PASSWORD', '')
OPENAI_API_URL = os.environ.get('OPENAI_API_URL', 'http://localhost:11434/v1')
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text:latest')
TEST_GROUP_ID = 'crud_test'
RENAME_GROUP_ID = 'crud_test_renamed'

# Store UUIDs between tests
_test_data = {}


@pytest_asyncio.fixture(loop_scope='module', scope='module')
async def graphiti_client():
    """Create Graphiti client for tests."""
    driver = FalkorDriver(
        host=FALKORDB_HOST,
        port=FALKORDB_PORT,
        password=FALKORDB_PASSWORD or None,
        database=TEST_GROUP_ID,
    )

    # Configure embedder with custom endpoint
    api_key = os.environ.get('OPENAI_API_KEY', 'not-needed')
    embedder_config = OpenAIEmbedderConfig(
        api_key=api_key,
        base_url=OPENAI_API_URL,
        embedding_model=EMBEDDING_MODEL,
    )
    embedder = OpenAIEmbedder(embedder_config)

    client = Graphiti(graph_driver=driver, embedder=embedder)

    # Build indices
    await client.build_indices_and_constraints()

    yield client

    # Cleanup: remove test groups
    for group_id in [TEST_GROUP_ID, RENAME_GROUP_ID]:
        try:
            await client.remove_group(group_id)
        except Exception:
            pass

    await client.close()


@pytest.mark.asyncio(loop_scope='module')
class TestEntityCRUD:
    """Tests for entity CRUD operations."""

    async def test_create_entity(self, graphiti_client: Graphiti):
        """Test creating an entity."""
        entity = await graphiti_client.create_entity(
            name='Test Person',
            group_id=TEST_GROUP_ID,
            entity_type='Person',
            summary='A test person for CRUD testing',
            attributes={'role': 'tester'},
        )

        assert entity is not None
        assert entity.uuid is not None
        assert entity.name == 'Test Person'
        assert entity.summary == 'A test person for CRUD testing'
        assert entity.group_id == TEST_GROUP_ID
        assert 'Person' in entity.labels
        assert entity.attributes.get('role') == 'tester'
        assert entity.name_embedding is not None
        assert entity.summary_embedding is not None

        # Store UUID for later tests
        _test_data['entity_uuid'] = entity.uuid

    async def test_create_entity_invalid_type(self, graphiti_client: Graphiti):
        """Test that invalid entity_type raises ValueError."""
        with pytest.raises(ValueError, match='Invalid entity_type'):
            await graphiti_client.create_entity(
                name='Invalid Entity',
                group_id=TEST_GROUP_ID,
                entity_type='Invalid:Type',  # Contains colon - invalid
            )

        with pytest.raises(ValueError, match='Invalid entity_type'):
            await graphiti_client.create_entity(
                name='Invalid Entity',
                group_id=TEST_GROUP_ID,
                entity_type='123StartWithNumber',  # Starts with number - invalid
            )

    async def test_get_entity(self, graphiti_client: Graphiti):
        """Test retrieving an entity."""
        entity = await graphiti_client.get_entity(_test_data['entity_uuid'])

        assert entity is not None
        assert entity.uuid == _test_data['entity_uuid']
        assert entity.name == 'Test Person'

    async def test_update_entity_name(self, graphiti_client: Graphiti):
        """Test updating entity name (should regenerate embedding)."""
        entity = await graphiti_client.update_entity(
            uuid=_test_data['entity_uuid'],
            name='Updated Person',
        )

        assert entity.name == 'Updated Person'
        assert entity.name_embedding is not None

    async def test_update_entity_summary(self, graphiti_client: Graphiti):
        """Test updating entity summary (should regenerate embedding)."""
        entity = await graphiti_client.update_entity(
            uuid=_test_data['entity_uuid'],
            summary='Updated summary for testing',
        )

        assert entity.summary == 'Updated summary for testing'
        assert entity.summary_embedding is not None

    async def test_update_entity_attributes(self, graphiti_client: Graphiti):
        """Test updating entity attributes."""
        entity = await graphiti_client.update_entity(
            uuid=_test_data['entity_uuid'],
            attributes={'role': 'senior_tester', 'level': '5'},
        )

        assert entity.attributes.get('role') == 'senior_tester'
        assert entity.attributes.get('level') == '5'

    async def test_update_entity_type(self, graphiti_client: Graphiti):
        """Test updating entity type (labels should change in database)."""
        # First check current type is Person
        entity = await graphiti_client.get_entity(_test_data['entity_uuid'])
        assert 'Person' in entity.labels

        # Change to Organization
        entity = await graphiti_client.update_entity(
            uuid=_test_data['entity_uuid'],
            entity_type='Organization',
        )

        assert 'Organization' in entity.labels
        assert 'Person' not in entity.labels

        # Verify in database by fetching again
        entity = await graphiti_client.get_entity(_test_data['entity_uuid'])
        assert 'Organization' in entity.labels
        assert 'Person' not in entity.labels

        # Change back to Person for other tests
        await graphiti_client.update_entity(
            uuid=_test_data['entity_uuid'],
            entity_type='Person',
        )

    async def test_update_entity_invalid_type(self, graphiti_client: Graphiti):
        """Test that invalid entity_type in update raises ValueError."""
        with pytest.raises(ValueError, match='Invalid entity_type'):
            await graphiti_client.update_entity(
                uuid=_test_data['entity_uuid'],
                entity_type='Invalid-Type',  # Contains hyphen - invalid
            )

    async def test_get_entities_by_group_id(self, graphiti_client: Graphiti):
        """Test listing entities by group ID."""
        entities = await graphiti_client.get_entities_by_group_id(TEST_GROUP_ID)

        assert len(entities) >= 1
        assert any(e.uuid == _test_data['entity_uuid'] for e in entities)


@pytest.mark.asyncio(loop_scope='module')
class TestEdgeCRUD:
    """Tests for edge CRUD operations."""

    async def test_create_second_entity_for_edge(self, graphiti_client: Graphiti):
        """Create a second entity to use as edge target."""
        entity = await graphiti_client.create_entity(
            name='Test Project',
            group_id=TEST_GROUP_ID,
            entity_type='Project',
            summary='A test project',
        )
        _test_data['target_entity_uuid'] = entity.uuid

    async def test_create_edge(self, graphiti_client: Graphiti):
        """Test creating an edge with auto-created episode."""
        edge = await graphiti_client.create_edge(
            source_node_uuid=_test_data['entity_uuid'],
            target_node_uuid=_test_data['target_entity_uuid'],
            name='WORKS_ON',
            fact='Updated Person works on Test Project',
            group_id=TEST_GROUP_ID,
        )

        assert edge is not None
        assert edge.uuid is not None
        assert edge.name == 'WORKS_ON'
        assert edge.fact == 'Updated Person works on Test Project'
        assert edge.source_node_uuid == _test_data['entity_uuid']
        assert edge.target_node_uuid == _test_data['target_entity_uuid']
        assert edge.fact_embedding is not None

        # Edge should have an episode (auto-created)
        assert edge.episodes is not None
        assert len(edge.episodes) == 1

        _test_data['edge_uuid'] = edge.uuid
        _test_data['episode_uuid'] = edge.episodes[0]

    async def test_edge_episode_exists(self, graphiti_client: Graphiti):
        """Test that the auto-created episode exists and has correct content."""
        episode = await graphiti_client.get_episode(_test_data['episode_uuid'])

        assert episode is not None
        assert episode.content == 'Updated Person works on Test Project'
        assert episode.source_description == 'Manual entry'
        assert _test_data['edge_uuid'] in episode.entity_edges

    async def test_get_edge(self, graphiti_client: Graphiti):
        """Test retrieving an edge."""
        edge = await graphiti_client.get_edge(_test_data['edge_uuid'])

        assert edge is not None
        assert edge.uuid == _test_data['edge_uuid']
        assert edge.name == 'WORKS_ON'

    async def test_update_edge_name(self, graphiti_client: Graphiti):
        """Test updating edge name."""
        edge = await graphiti_client.update_edge(
            uuid=_test_data['edge_uuid'],
            name='CONTRIBUTES_TO',
        )

        assert edge.name == 'CONTRIBUTES_TO'

    async def test_update_edge_fact(self, graphiti_client: Graphiti):
        """Test updating edge fact (should regenerate embedding)."""
        edge = await graphiti_client.update_edge(
            uuid=_test_data['edge_uuid'],
            fact='Updated Person contributes to Test Project as lead',
        )

        assert edge.fact == 'Updated Person contributes to Test Project as lead'
        assert edge.fact_embedding is not None

    async def test_get_edges_by_group_id(self, graphiti_client: Graphiti):
        """Test listing edges by group ID."""
        edges = await graphiti_client.get_edges_by_group_id(TEST_GROUP_ID)

        assert len(edges) >= 1
        assert any(e.uuid == _test_data['edge_uuid'] for e in edges)


@pytest.mark.asyncio(loop_scope='module')
class TestEpisodeOperations:
    """Tests for episode operations."""

    async def test_get_episodes_by_group_id(self, graphiti_client: Graphiti):
        """Test listing episodes by group ID."""
        episodes = await graphiti_client.get_episodes_by_group_id(TEST_GROUP_ID)

        assert len(episodes) >= 1
        assert any(e.uuid == _test_data['episode_uuid'] for e in episodes)

    async def test_get_episodes_by_group_id_empty(self, graphiti_client: Graphiti):
        """Test listing episodes for non-existent group returns empty list."""
        episodes = await graphiti_client.get_episodes_by_group_id('nonexistent_group')

        assert episodes == []


@pytest.mark.asyncio(loop_scope='module')
class TestGroupOperations:
    """Tests for group-level operations."""

    async def test_get_groups(self, graphiti_client: Graphiti):
        """Test getting all group IDs."""
        groups = await graphiti_client.get_groups()

        assert isinstance(groups, list)
        assert TEST_GROUP_ID in groups

    async def test_get_graph_stats(self, graphiti_client: Graphiti):
        """Test getting graph statistics."""
        stats = await graphiti_client.get_graph_stats(group_id=TEST_GROUP_ID)

        assert 'node_count' in stats
        assert 'edge_count' in stats
        assert 'episode_count' in stats
        assert 'episode_edge_count' in stats

        # We should have at least 2 entities (Person, Project), 1 edge, 1 episode
        assert stats['node_count'] >= 2
        assert stats['edge_count'] >= 1
        assert stats['episode_count'] >= 1

    async def test_execute_query(self, graphiti_client: Graphiti):
        """Test executing a raw Cypher query."""
        result, _, _ = await graphiti_client.execute_query(
            'MATCH (n:Entity {group_id: $group_id}) RETURN count(n) AS count',
            group_id=TEST_GROUP_ID,
        )

        assert len(result) == 1
        assert result[0]['count'] >= 2

    async def test_rename_group(self, graphiti_client: Graphiti):
        """Test renaming a group."""
        # Create a temporary group with some data
        temp_entity = await graphiti_client.create_entity(
            name='Temp Entity',
            group_id=RENAME_GROUP_ID,
            entity_type='TempType',
        )

        # Verify it exists
        groups_before = await graphiti_client.get_groups()
        assert RENAME_GROUP_ID in groups_before

        # Rename the group
        new_name = f'{RENAME_GROUP_ID}_new'
        await graphiti_client.rename_group(RENAME_GROUP_ID, new_name)

        # Verify rename worked
        groups_after = await graphiti_client.get_groups()
        assert RENAME_GROUP_ID not in groups_after
        assert new_name in groups_after

        # Cleanup
        await graphiti_client.remove_group(new_name)

    async def test_rename_group_same_name_error(self, graphiti_client: Graphiti):
        """Test that renaming to same name raises ValueError."""
        with pytest.raises(ValueError, match='must be different'):
            await graphiti_client.rename_group(TEST_GROUP_ID, TEST_GROUP_ID)


@pytest.mark.asyncio(loop_scope='module')
class TestCleanup:
    """Cleanup tests - run last."""

    async def test_remove_edge(self, graphiti_client: Graphiti):
        """Test removing an edge."""
        await graphiti_client.remove_edge(_test_data['edge_uuid'])

        with pytest.raises(EdgeNotFoundError):
            await graphiti_client.get_edge(_test_data['edge_uuid'])

    async def test_remove_entities(self, graphiti_client: Graphiti):
        """Test removing entities."""
        await graphiti_client.remove_entity(_test_data['entity_uuid'])
        await graphiti_client.remove_entity(_test_data['target_entity_uuid'])

        with pytest.raises(NodeNotFoundError):
            await graphiti_client.get_entity(_test_data['entity_uuid'])

    async def test_remove_group(self, graphiti_client: Graphiti):
        """Test removing an entire group."""
        # Create a temporary group
        temp_group = 'temp_group_for_deletion'
        await graphiti_client.create_entity(
            name='Entity to delete',
            group_id=temp_group,
            entity_type='TempEntity',
        )

        # Verify it exists
        groups = await graphiti_client.get_groups()
        assert temp_group in groups

        # Remove the group
        await graphiti_client.remove_group(temp_group)

        # Verify it's gone
        groups = await graphiti_client.get_groups()
        assert temp_group not in groups


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
