"""Prompt templates for GraphRAG v2 — based on LlamaIndex GraphRAG v2 cookbook."""

# ---------------------------------------------------------------------------
# Entity/relationship extraction — from the notebook's KG_TRIPLET_EXTRACT_TMPL
# ---------------------------------------------------------------------------

KG_TRIPLET_EXTRACT_TMPL = """\
-Goal-
Given a text document, identify all entities and their entity types from the text \
and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, \
target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity \
and the target entity are related to each other

3. Output Formatting:
- Return the result in valid JSON format with two keys: 'entities' (list of entity \
objects) and 'relationships' (list of relationship objects).
- Exclude any text outside the JSON structure (e.g., no explanations or comments).
- If no entities or relationships are identified, return empty lists: \
{{ "entities": [], "relationships": [] }}.

-An Output Example-
{{
  "entities": [
    {{
      "entity_name": "Albert Einstein",
      "entity_type": "Person",
      "entity_description": "Albert Einstein was a theoretical physicist who developed \
the theory of relativity and made significant contributions to physics."
    }},
    {{
      "entity_name": "Theory of Relativity",
      "entity_type": "Scientific Theory",
      "entity_description": "A scientific theory developed by Albert Einstein, describing \
the laws of physics in relation to observers in different frames of reference."
    }},
    {{
      "entity_name": "Nobel Prize in Physics",
      "entity_type": "Award",
      "entity_description": "A prestigious international award in the field of physics, \
awarded annually by the Royal Swedish Academy of Sciences."
    }}
  ],
  "relationships": [
    {{
      "source_entity": "Albert Einstein",
      "target_entity": "Theory of Relativity",
      "relation": "developed",
      "relationship_description": "Albert Einstein is the developer of the theory \
of relativity."
    }},
    {{
      "source_entity": "Albert Einstein",
      "target_entity": "Nobel Prize in Physics",
      "relation": "won",
      "relationship_description": "Albert Einstein won the Nobel Prize in Physics \
in 1921."
    }}
  ]
}}

-Real Data-
######################
text: {text}
######################
output:"""

# ---------------------------------------------------------------------------
# Community summarization — from the notebook's GraphRAGStore
# ---------------------------------------------------------------------------

COMMUNITY_SUMMARY_SYSTEM_PROMPT = """\
You are provided with a set of relationships from a knowledge graph, each represented as \
entity1->entity2->relation->relationship_description. Your task is to create a summary \
of these relationships. The summary should include the names of the entities involved \
and a concise synthesis of the relationship descriptions. The goal is to capture the most \
critical and relevant details that highlight the nature and significance of each \
relationship. Ensure that the summary is coherent and integrates the information in a way \
that emphasizes the key aspects of the relationships."""

# ---------------------------------------------------------------------------
# Query entity extraction
# ---------------------------------------------------------------------------

QUERY_ENTITY_EXTRACT_PROMPT = """\
Extract the key entity names from the following question.
Return only the entity names as a JSON array of strings.
Focus on proper nouns, organization names, person names, and specific concepts.

Question: {query}

Entity names (JSON array):"""
