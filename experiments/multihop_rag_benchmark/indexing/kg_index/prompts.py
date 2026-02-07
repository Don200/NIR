"""Prompt templates for GraphRAG v2 entity/relationship extraction and community summarization."""

KG_EXTRACTION_PROMPT = """Given the following text, extract all entities and relationships.

For each entity, provide:
- name: The entity name (capitalize properly)
- type: The entity type (Person, Organization, Location, Event, Technology, Product, Concept, etc.)
- description: A comprehensive description of the entity based on the text

For each relationship, provide:
- source: The source entity name
- target: The target entity name
- relation: A short label for the relationship
- description: A description of how these entities are related

Return the result as JSON with two arrays: "entities" and "relationships".

Example output:
{{
  "entities": [
    {{"name": "OpenAI", "type": "Organization", "description": "An AI research company that developed ChatGPT and GPT-4."}},
    {{"name": "Sam Altman", "type": "Person", "description": "CEO of OpenAI who leads the company's strategic direction."}}
  ],
  "relationships": [
    {{"source": "Sam Altman", "target": "OpenAI", "relation": "CEO_OF", "description": "Sam Altman serves as the CEO of OpenAI, leading its operations and strategy."}}
  ]
}}

Text:
{text}

Extract entities and relationships as JSON:"""

COMMUNITY_SUMMARY_PROMPT = """You are given a set of entities and their relationships within a community in a knowledge graph.
Write a comprehensive summary of this community that captures the key entities, their roles, and how they relate to each other.
The summary should be useful for answering questions about the topics covered by this community.

Community relationships:
{edges_text}

Write a coherent summary paragraph that covers the key information in this community:"""

QUERY_ENTITY_EXTRACT_PROMPT = """Extract the key entity names from the following question.
Return only the entity names as a JSON array of strings.
Focus on proper nouns, organization names, person names, and specific concepts.

Question: {query}

Entity names (JSON array):"""
