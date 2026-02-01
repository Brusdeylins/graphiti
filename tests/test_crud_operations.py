"""
Integration tests for Graphiti CRUD operations.

Tests the new CRUD methods on the Graphiti class:
- Entity: create, get, update, delete
- Edge: create, get, update, delete
- Episode: get, delete
- Group: get_groups, get_entities_by_group_id, etc.

Run with: pytest tests/test_crud_operations.py -v
"""

import os
import pytest
import pytest_asyncio

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.errors import NodeNotFoundError, EdgeNotFoundError


# Test configuration - uses running FalkorDB container
FALKORDB_HOST = os.environ.get('FALKORDB_HOST', 'localhost')
FALKORDB_PORT = int(os.environ.get('FALKORDB_PORT', '6379'))
FALKORDB_PASSWORD = os.environ.get('FALKORDB_PASSWORD', 'password4FalkorDB!')
OPENAI_API_URL = os.environ.get('OPENAI_API_URL', 'https://openai.brusdeylins.info/v1')
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text:latest')
TEST_GROUP_ID = 'crud_test'

# Store UUIDs between tests
_test_data = {}


@pytest_asyncio.fixture(loop_scope='module', scope='module')
async def graphiti_client():
    """Create Graphiti client for tests."""
    driver = FalkorDriver(
        host=FALKORDB_HOST,
        port=FALKORDB_PORT,
        password=FALKORDB_PASSWORD,
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

    # Cleanup: delete test group
    try:
        await client.delete_group(TEST_GROUP_ID)
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
class TestGroupOperations:
    """Tests for group-level operations."""

    async def test_get_groups(self, graphiti_client: Graphiti):
        """Test getting all group IDs."""
        groups = await graphiti_client.get_groups()

        assert isinstance(groups, list)
        # Test group should be in list (if data exists)

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


@pytest.mark.asyncio(loop_scope='module')
class TestCleanup:
    """Cleanup tests - run last."""

    async def test_delete_edge(self, graphiti_client: Graphiti):
        """Test deleting an edge."""
        await graphiti_client.delete_edge(_test_data['edge_uuid'])

        with pytest.raises(EdgeNotFoundError):
            await graphiti_client.get_edge(_test_data['edge_uuid'])

    async def test_delete_entities(self, graphiti_client: Graphiti):
        """Test deleting entities."""
        await graphiti_client.delete_entity(_test_data['entity_uuid'])
        await graphiti_client.delete_entity(_test_data['target_entity_uuid'])

        with pytest.raises(NodeNotFoundError):
            await graphiti_client.get_entity(_test_data['entity_uuid'])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
