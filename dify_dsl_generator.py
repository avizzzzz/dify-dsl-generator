#!/usr/bin/env python3
"""
Dify DSL Generator with Self-healing Loop
==========================================
Generates valid Dify workflow / chat-flow YAML from natural language requirements.
Features automatic validation and self-correction via LLM.

Architecture:
  1. Validator Engine  — strict structural / connectivity / variable-reference /
                         value-selector / node-type-specific checks
  2. LLM Interaction   — calls Anthropic API to generate & fix YAML
  3. Self-healing Loop — validates → catches errors → feeds back to LLM → retries

Validated modes: workflow, chat (advanced-chat)
"""

import copy
import os
import re
import sys
import json
import uuid
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dify_dsl_gen")

# ============================================================================
# Constants
# ============================================================================

MAX_ITERATIONS = 50

# Matches {{#node_id.variable_name#}}
DIFY_VAR_PATTERN = re.compile(r"\{\{#([^.#}]+)\.([^#}]*)#\}\}")

# Supported app modes
VALID_MODES = {"workflow", "chat", "advanced-chat"}

# Node types that are valid inside a Dify graph
RECOGNIZED_NODE_TYPES = {
    "start", "end", "llm", "code", "knowledge-retrieval",
    "if-else", "template-transform", "variable-aggregator",
    "parameter-extractor", "iteration", "answer",
    "question-classifier", "http-request", "tool", "variable-assigner",
}

# Node types that can appear as iteration children
ITERATION_CHILD_TYPES = {"start", "llm", "code", "knowledge-retrieval",
                         "template-transform", "if-else", "answer",
                         "http-request", "tool", "variable-assigner"}

# ============================================================================
# Exceptions
# ============================================================================


class DifyValidationError(Exception):
    """Raised when Dify YAML validation fails. Message includes exact error path."""


class DifyValidationWarning(Exception):
    """Non-fatal warning (used internally for result collection)."""


# ============================================================================
# Validation Engine — public API
# ============================================================================


def validate_dify_yaml(yaml_dict: Dict[str, Any]) -> List[str]:
    """
    Strictly validate a Dify workflow/chat YAML structure.

    Returns a list of non-fatal warnings (empty = completely clean).
    Raises DifyValidationError with a precise path on ANY fatal error.
    """
    warnings: List[str] = []
    if not isinstance(yaml_dict, dict):
        raise DifyValidationError("Root: expected a YAML dictionary/mapping at top level")

    # ---- Top-level keys ---------------------------------------------------
    for key in ("app", "workflow"):
        if key not in yaml_dict:
            raise DifyValidationError(f"Root: missing top-level key '{key}'")

    # Optional but recommended top-level keys
    for key in ("kind", "version"):
        if key not in yaml_dict:
            warnings.append(f"Root: recommended top-level key '{key}' is missing")

    app = yaml_dict["app"]
    if not isinstance(app, dict):
        raise DifyValidationError("app: expected a dictionary")

    mode = app.get("mode", "")
    if mode not in VALID_MODES:
        raise DifyValidationError(
            f"app.mode: expected one of {VALID_MODES}, got '{mode}'"
        )
    if "name" not in app:
        warnings.append("app.name: recommended field is missing")

    workflow = yaml_dict["workflow"]
    if not isinstance(workflow, dict):
        raise DifyValidationError("workflow: expected a dictionary")

    for field in ("version", "graph"):
        if field not in workflow:
            raise DifyValidationError(f"workflow: missing '{field}' field")

    graph = workflow["graph"]
    if not isinstance(graph, dict):
        raise DifyValidationError("workflow.graph: expected a dictionary")

    for field in ("nodes", "edges"):
        if field not in graph:
            raise DifyValidationError(f"workflow.graph: missing '{field}' array")
        if not isinstance(graph[field], list):
            raise DifyValidationError(f"workflow.graph.{field}: expected an array")

    nodes: List[Dict] = graph["nodes"]
    edges: List[Dict] = graph["edges"]

    # ---- Collect all node IDs (recursively into iteration children) --------
    valid_ids: Set[str] = set()
    node_type_map: Dict[str, str] = {}  # node_id → data.type
    _collect_node_ids_and_types(nodes, "workflow.graph.nodes", valid_ids, node_type_map)

    # ---- Edge uniqueness --------------------------------------------------
    seen_edge_ids: Set[str] = set()
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise DifyValidationError(
                f"workflow.graph.edges[{i}]: expected a dictionary/object"
            )
        edge_id = edge.get("id")
        if edge_id is not None:
            if not isinstance(edge_id, str) or not edge_id.strip():
                raise DifyValidationError(
                    f"workflow.graph.edges[{i}]: 'id' must be a non-empty string"
                )
            if edge_id in seen_edge_ids:
                raise DifyValidationError(f"Duplicate edge ID: '{edge_id}'")
            seen_edge_ids.add(edge_id)

    # ---- Edge connectivity ------------------------------------------------
    for i, edge in enumerate(edges):
        edge_id = edge.get("id", f"edges[{i}]")
        source = edge.get("source")
        target = edge.get("target")

        if source is None:
            raise DifyValidationError(f"Edge '{edge_id}': missing 'source' field")
        if target is None:
            raise DifyValidationError(f"Edge '{edge_id}': missing 'target' field")

        if source not in valid_ids:
            raise DifyValidationError(
                f"Edge '{edge_id}': source node '{source}' does not exist in nodes"
            )
        if target not in valid_ids:
            raise DifyValidationError(
                f"Edge '{edge_id}': target node '{target}' does not exist in nodes"
            )

        # Warn about edge type mismatches
        src_type = node_type_map.get(source, "")
        tgt_type = node_type_map.get(target, "")
        edge_data = edge.get("data", {})
        if isinstance(edge_data, dict):
            edge_src_type = edge_data.get("sourceType", "")
            edge_tgt_type = edge_data.get("targetType", "")
            if edge_src_type and src_type and edge_src_type != src_type:
                warnings.append(
                    f"Edge '{edge_id}': data.sourceType '{edge_src_type}' "
                    f"does not match actual source node type '{src_type}'"
                )
            if edge_tgt_type and tgt_type and edge_tgt_type != tgt_type:
                warnings.append(
                    f"Edge '{edge_id}': data.targetType '{edge_tgt_type}' "
                    f"does not match actual target node type '{tgt_type}'"
                )

    # ---- Node-type-specific validation ------------------------------------
    _validate_node_types(nodes, valid_ids, node_type_map, "workflow.graph.nodes")

    # ---- Variable reference validation ------------------------------------
    _validate_variable_refs(nodes, valid_ids)

    # ---- Value-selector validation ----------------------------------------
    _validate_value_selectors(nodes, valid_ids)

    # ---- Iterator / output selectors in iteration nodes -------------------
    _validate_iteration_selectors(nodes, valid_ids)

    # ---- Graph connectivity (start → … → end reachability) ----------------
    _validate_graph_connectivity(nodes, edges, mode, node_type_map)

    # ---- Mode-specific rules -----------------------------------------------
    _validate_mode_specific(yaml_dict, nodes, node_type_map, mode)

    # ---- Node-position sanity ---------------------------------------------
    _validate_positions(nodes)

    log.info(
        "Validation PASSED — mode=%s, %d node(s), %d edge(s), %d warning(s)",
        mode, len(valid_ids), len(edges), len(warnings),
    )
    return warnings


# ============================================================================
# Node-ID collection (with type tracking)
# ============================================================================


def _collect_node_ids_and_types(
    nodes: List[Dict],
    path: str,
    valid_ids: Set[str],
    node_type_map: Dict[str, str],
) -> None:
    """Recursively collect node IDs + track data.type. Validates uniqueness."""
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise DifyValidationError(f"{path}[{i}]: expected a dictionary/object")

        node_id = node.get("id")
        if node_id is None:
            raise DifyValidationError(
                f"{path}[{i}] ('{node.get('title', '?')}'): missing 'id' field"
            )
        if not isinstance(node_id, str) or not node_id.strip():
            raise DifyValidationError(
                f"{path}[{i}]: 'id' must be a non-empty string, got {node_id!r}"
            )
        node_id = node_id.strip()

        if node_id in valid_ids:
            raise DifyValidationError(
                f"{path}[{i}]: duplicate node ID '{node_id}' (must be globally unique)"
            )
        valid_ids.add(node_id)

        # Track data.type
        node_data = node.get("data", {})
        if isinstance(node_data, dict):
            dtype = node_data.get("type", "")
            if isinstance(dtype, str) and dtype:
                node_type_map[node_id] = dtype

        # Common structural fields check
        if node.get("type") != "custom":
            warnings_extra = f"{path}[{i}] ('{node_id}'): node 'type' should be 'custom', got '{node.get('type')}'"
            # non-fatal: we log later via warnings if needed

        required_node_keys = {"id", "type", "data", "position", "width", "height"}
        missing = required_node_keys - set(node.keys())
        if missing:
            raise DifyValidationError(
                f"{path}[{i}] ('{node_id}'): missing required keys: {sorted(missing)}"
            )

        if not isinstance(node_data, dict):
            raise DifyValidationError(
                f"{path}[{i}] ('{node_id}'): 'data' must be a dictionary"
            )

        # Recurse into iteration-node children
        if isinstance(node_data, dict) and "children" in node_data:
            children = node_data["children"]
            if children is not None:
                _validate_and_collect_child_graph(
                    children, f"{path}[{i}] ('{node_id}').data.children",
                    valid_ids, node_type_map,
                )


def _validate_and_collect_child_graph(
    children: Any,
    path: str,
    valid_ids: Set[str],
    node_type_map: Dict[str, str],
) -> None:
    """Validate a nested child graph inside an iteration node."""
    if not isinstance(children, dict):
        raise DifyValidationError(f"{path}: expected a dictionary (nested graph)")

    child_nodes = children.get("nodes")
    child_edges = children.get("edges")

    if child_nodes is None:
        raise DifyValidationError(f"{path}: missing 'nodes' array")
    if child_edges is None:
        raise DifyValidationError(f"{path}: missing 'edges' array")
    if not isinstance(child_nodes, list):
        raise DifyValidationError(f"{path}.nodes: expected an array")
    if not isinstance(child_edges, list):
        raise DifyValidationError(f"{path}.edges: expected an array")

    # Collect child node IDs into the global set
    _collect_node_ids_and_types(child_nodes, f"{path}.nodes", valid_ids, node_type_map)

    # Child edge uniqueness
    child_seen: Set[str] = set()
    for i, edge in enumerate(child_edges):
        if not isinstance(edge, dict):
            raise DifyValidationError(f"{path}.edges[{i}]: expected a dictionary")
        eid = edge.get("id")
        if eid is not None:
            if not isinstance(eid, str) or not eid.strip():
                raise DifyValidationError(f"{path}.edges[{i}]: edge 'id' must be a non-empty string")
            if eid in child_seen:
                raise DifyValidationError(f"{path}.edges[{i}]: duplicate edge ID '{eid}'")
            child_seen.add(eid)

        source = edge.get("source")
        target = edge.get("target")
        if source and source not in valid_ids:
            raise DifyValidationError(
                f"Child edge '{edge.get('id', f'edges[{i}]')}': "
                f"source node '{source}' does not exist"
            )
        if target and target not in valid_ids:
            raise DifyValidationError(
                f"Child edge '{edge.get('id', f'edges[{i}]')}': "
                f"target node '{target}' does not exist"
            )

    # Validate child node types
    _validate_node_types(child_nodes, valid_ids, node_type_map, f"{path}.nodes")


# ============================================================================
# Node-type-specific validation
# ============================================================================


def _validate_node_types(
    nodes: List[Dict],
    valid_ids: Set[str],
    node_type_map: Dict[str, str],
    path: str,
) -> None:
    """Run per-node-type validation rules."""
    for i, node in enumerate(nodes):
        node_data = node.get("data", {})
        dtype = node_data.get("type", "") if isinstance(node_data, dict) else ""
        node_id = node.get("id", f"{path}[{i}]")

        if dtype not in RECOGNIZED_NODE_TYPES:
            raise DifyValidationError(
                f"Node '{node_id}': unknown data.type '{dtype}'. "
                f"Recognized types: {sorted(RECOGNIZED_NODE_TYPES)}"
            )

        # Dispatch to type-specific validator
        validator = _NODE_TYPE_VALIDATORS.get(dtype)
        if validator:
            validator(node, node_id, node_data, valid_ids, path)

        # Recurse into iteration children
        if isinstance(node_data, dict) and "children" in node_data:
            children = node_data["children"]
            if isinstance(children, dict) and "nodes" in children:
                _validate_node_types(
                    children["nodes"], valid_ids, node_type_map,
                    f"{path}[{i}] ('{node_id}').data.children.nodes",
                )


def _validate_start_node(
    node: Dict, node_id: str, data: Dict, valid_ids: Set[str], path: str
) -> None:
    """Start node: must have variables array."""
    variables = data.get("variables")
    if variables is not None and not isinstance(variables, list):
        raise DifyValidationError(
            f"Node '{node_id}' (start): 'variables' must be an array"
        )
    if isinstance(variables, list):
        for vi, var in enumerate(variables):
            if isinstance(var, dict):
                if "variable" not in var:
                    raise DifyValidationError(
                        f"Node '{node_id}' (start): variables[{vi}] missing 'variable' name"
                    )


def _validate_end_node(
    node: Dict, node_id: str, data: Dict, valid_ids: Set[str], path: str
) -> None:
    """End node: should have outputs array."""
    outputs = data.get("outputs")
    if outputs is not None and not isinstance(outputs, list):
        raise DifyValidationError(
            f"Node '{node_id}' (end): 'outputs' must be an array if present"
        )


def _validate_llm_node(
    node: Dict, node_id: str, data: Dict, valid_ids: Set[str], path: str
) -> None:
    """LLM node: must have model config and prompt_template."""
    model = data.get("model")
    if not isinstance(model, dict):
        raise DifyValidationError(f"Node '{node_id}' (llm): 'model' must be a dictionary")
    for field in ("provider", "name", "mode"):
        if not model.get(field):
            raise DifyValidationError(
                f"Node '{node_id}' (llm): model.{field} is required and must be non-empty"
            )
    prompt = data.get("prompt_template")
    if prompt is not None:
        if not isinstance(prompt, list):
            raise DifyValidationError(
                f"Node '{node_id}' (llm): 'prompt_template' must be an array"
            )
        for pi, p in enumerate(prompt):
            if isinstance(p, dict) and not p.get("text"):
                raise DifyValidationError(
                    f"Node '{node_id}' (llm): prompt_template[{pi}] has empty 'text'"
                )


def _validate_code_node(
    node: Dict, node_id: str, data: Dict, valid_ids: Set[str], path: str
) -> None:
    """Code node: must have code_language and code."""
    if not data.get("code_language"):
        raise DifyValidationError(
            f"Node '{node_id}' (code): 'code_language' is required (e.g. 'python3')"
        )
    if not data.get("code"):
        raise DifyValidationError(f"Node '{node_id}' (code): 'code' field is required")


def _validate_ifelse_node(
    node: Dict, node_id: str, data: Dict, valid_ids: Set[str], path: str
) -> None:
    """If-else node: must have conditions array."""
    conditions = data.get("conditions")
    if not isinstance(conditions, list) or len(conditions) == 0:
        raise DifyValidationError(
            f"Node '{node_id}' (if-else): 'conditions' must be a non-empty array"
        )
    for ci, cond in enumerate(conditions):
        if isinstance(cond, dict):
            if "comparison_operator" not in cond:
                raise DifyValidationError(
                    f"Node '{node_id}' (if-else): conditions[{ci}] missing 'comparison_operator'"
                )


def _validate_knowledge_node(
    node: Dict, node_id: str, data: Dict, valid_ids: Set[str], path: str
) -> None:
    """Knowledge retrieval: must have dataset_ids and query_variable_selector."""
    if not data.get("dataset_ids"):
        raise DifyValidationError(
            f"Node '{node_id}' (knowledge-retrieval): 'dataset_ids' is required"
        )
    qvs = data.get("query_variable_selector")
    if not qvs or not isinstance(qvs, list) or len(qvs) < 2:
        raise DifyValidationError(
            f"Node '{node_id}' (knowledge-retrieval): "
            f"'query_variable_selector' must be [node_id, field_name]"
        )


def _validate_iteration_node(
    node: Dict, node_id: str, data: Dict, valid_ids: Set[str], path: str
) -> None:
    """Iteration node: must have children, iterator_selector, start_node_id."""
    children = data.get("children")
    if children is None:
        raise DifyValidationError(f"Node '{node_id}' (iteration): missing 'children' graph")
    if not isinstance(children, dict):
        raise DifyValidationError(f"Node '{node_id}' (iteration): 'children' must be a dictionary")

    isel = data.get("iterator_selector")
    if not isinstance(isel, list) or len(isel) < 2:
        raise DifyValidationError(
            f"Node '{node_id}' (iteration): "
            f"'iterator_selector' must be [node_id, field_name]"
        )

    snid = data.get("start_node_id")
    if not snid or not isinstance(snid, str):
        raise DifyValidationError(
            f"Node '{node_id}' (iteration): 'start_node_id' is required and must be a string"
        )
    if snid not in valid_ids:
        raise DifyValidationError(
            f"Node '{node_id}' (iteration): start_node_id '{snid}' does not exist in children"
        )


def _validate_http_node(
    node: Dict, node_id: str, data: Dict, valid_ids: Set[str], path: str
) -> None:
    """HTTP request node: must have url and method."""
    if not data.get("url"):
        raise DifyValidationError(f"Node '{node_id}' (http-request): 'url' is required")
    method = data.get("method", "")
    if method.upper() not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
        raise DifyValidationError(
            f"Node '{node_id}' (http-request): 'method' must be a valid HTTP method, got '{method}'"
        )


def _validate_answer_node(
    node: Dict, node_id: str, data: Dict, valid_ids: Set[str], path: str
) -> None:
    """Answer node: must have answer field."""
    answer = data.get("answer")
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        raise DifyValidationError(
            f"Node '{node_id}' (answer): 'answer' field is required and must be non-empty"
        )


def _validate_assigner_node(
    node: Dict, node_id: str, data: Dict, valid_ids: Set[str], path: str
) -> None:
    """Variable assigner: must have variables list."""
    variables = data.get("variables")
    if not isinstance(variables, list) or len(variables) == 0:
        raise DifyValidationError(
            f"Node '{node_id}' (variable-assigner): 'variables' must be a non-empty array"
        )


_NODE_TYPE_VALIDATORS = {
    "start": _validate_start_node,
    "end": _validate_end_node,
    "llm": _validate_llm_node,
    "code": _validate_code_node,
    "if-else": _validate_ifelse_node,
    "knowledge-retrieval": _validate_knowledge_node,
    "iteration": _validate_iteration_node,
    "http-request": _validate_http_node,
    "answer": _validate_answer_node,
    "variable-assigner": _validate_assigner_node,
}


# ============================================================================
# Variable-reference validation
# ============================================================================


def _validate_variable_refs(nodes: List[Dict], valid_ids: Set[str]) -> None:
    """Scan every node (incl. children) for `{{#id.field#}}` refs."""
    for node in nodes:
        node_id = node.get("id", "?")
        _scan_object_for_vars(node, f"Node '{node_id}'", valid_ids)

        node_data = node.get("data", {})
        if isinstance(node_data, dict) and "children" in node_data:
            children = node_data["children"]
            if isinstance(children, dict):
                for child in children.get("nodes", []):
                    cid = child.get("id", "?")
                    _scan_object_for_vars(child, f"Node '{node_id}' > child '{cid}'", valid_ids)


def _scan_object_for_vars(obj: Any, context: str, valid_ids: Set[str]) -> None:
    """Recursively walk a nested structure looking for `{{#id.field#}}` strings."""
    if isinstance(obj, dict):
        for _k, value in obj.items():
            _scan_object_for_vars(value, context, valid_ids)
    elif isinstance(obj, list):
        for item in obj:
            _scan_object_for_vars(item, context, valid_ids)
    elif isinstance(obj, str):
        for match in DIFY_VAR_PATTERN.finditer(obj):
            ref_id = match.group(1).strip()
            if ref_id not in valid_ids:
                raise DifyValidationError(
                    f"{context}: variable reference '{{{{#{ref_id}.{match.group(2)}}}}}' "
                    f"targets node '{ref_id}' which does not exist"
                )


# ============================================================================
# Value-selector validation  ([node_id, field_name]  pattern)
# ============================================================================

VALUE_SELECTOR_KEYS = {
    "value_selector", "variable_selector", "iterator_selector",
    "output_selector", "query_variable_selector",
}


def _validate_value_selectors(nodes: List[Dict], valid_ids: Set[str]) -> None:
    """Scan all nodes for value_selector / variable_selector format ['id','field']."""
    for node in nodes:
        node_id = node.get("id", "?")
        _scan_for_selectors(node, f"Node '{node_id}'", valid_ids)

        node_data = node.get("data", {})
        if isinstance(node_data, dict) and "children" in node_data:
            children = node_data["children"]
            if isinstance(children, dict):
                for child in children.get("nodes", []):
                    cid = child.get("id", "?")
                    _scan_for_selectors(child, f"Node '{node_id}' > child '{cid}'", valid_ids)


def _scan_for_selectors(obj: Any, context: str, valid_ids: Set[str]) -> None:
    """Recursively find value_selector-like patterns and validate node_id."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in VALUE_SELECTOR_KEYS and isinstance(value, list) and len(value) >= 1:
                node_id = value[0]
                if isinstance(node_id, str) and node_id not in valid_ids:
                    raise DifyValidationError(
                        f"{context}: {key} references node '{node_id}' which does not exist"
                    )
            _scan_for_selectors(value, context, valid_ids)
    elif isinstance(obj, list):
        for item in obj:
            _scan_for_selectors(item, context, valid_ids)


# ============================================================================
# Iteration-specific selector validation
# ============================================================================


def _validate_iteration_selectors(nodes: List[Dict], valid_ids: Set[str]) -> None:
    """Validate iteration node iterator_selector and output_selector fields."""
    for node in nodes:
        node_data = node.get("data", {})
        if not isinstance(node_data, dict):
            continue
        if node_data.get("type") != "iteration":
            continue

        node_id = node.get("id", "?")

        # iterator_selector must be [existing_node_id, field_name]
        isel = node_data.get("iterator_selector")
        if isinstance(isel, list) and len(isel) >= 1:
            ref_id = isel[0]
            if isinstance(ref_id, str) and ref_id not in valid_ids:
                raise DifyValidationError(
                    f"Node '{node_id}' (iteration): iterator_selector "
                    f"references node '{ref_id}' which does not exist"
                )

        # output_selector must be [existing_node_id, field_name]
        osel = node_data.get("output_selector")
        if isinstance(osel, list) and len(osel) >= 1:
            ref_id = osel[0]
            if isinstance(ref_id, str) and ref_id not in valid_ids:
                raise DifyValidationError(
                    f"Node '{node_id}' (iteration): output_selector "
                    f"references node '{ref_id}' which does not exist"
                )

        # Recurse into children
        children = node_data.get("children")
        if isinstance(children, dict):
            _validate_iteration_selectors(children.get("nodes", []), valid_ids)


# ============================================================================
# Graph connectivity
# ============================================================================


def _validate_graph_connectivity(
    nodes: List[Dict], edges: List[Dict], mode: str, node_type_map: Dict[str, str]
) -> None:
    """Check that all nodes are reachable from some start node."""
    # Build adjacency
    adjacency: Dict[str, List[str]] = {}
    for edge in edges:
        s, t = edge.get("source"), edge.get("target")
        if s and t:
            adjacency.setdefault(s, []).append(t)

    all_ids = set()
    start_ids: Set[str] = set()
    end_ids: Set[str] = set()
    for node in nodes:
        nid = node.get("id")
        if not nid:
            continue
        all_ids.add(nid)
        nd = node.get("data", {})
        if isinstance(nd, dict) and nd.get("type") == "start":
            start_ids.add(nid)
        if isinstance(nd, dict) and nd.get("type") == "end":
            end_ids.add(nid)

    if not start_ids:
        raise DifyValidationError("Graph must contain at least one 'start' node")
    if mode in ("chat", "advanced-chat"):
        answer_ids = {nid for nid, t in node_type_map.items() if t == "answer"}
        chat_end_ids = end_ids | answer_ids
        if not chat_end_ids:
            raise DifyValidationError("Chat mode requires at least one 'answer' or 'end' node")

    # BFS from start nodes to find reachable
    reachable: Set[str] = set()
    stack = list(start_ids)
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        for nxt in adjacency.get(cur, []):
            if nxt not in reachable:
                stack.append(nxt)

    unreachable = all_ids - reachable
    if unreachable:
        raise DifyValidationError(
            f"Unreachable node(s) (no path from any start node): {sorted(unreachable)}"
        )


# ============================================================================
# Mode-specific validation
# ============================================================================


def _validate_mode_specific(
    yaml_dict: Dict[str, Any],
    nodes: List[Dict],
    node_type_map: Dict[str, str],
    mode: str,
) -> None:
    """Run rules that only apply to specific app modes."""

    workflow = yaml_dict.get("workflow", {})

    if mode in ("chat", "advanced-chat"):
        # Chat flows must have conversation_variables
        cv = workflow.get("conversation_variables")
        if cv is not None and not isinstance(cv, list):
            raise DifyValidationError(
                "workflow.conversation_variables: must be an array in chat mode"
            )

        # Chat flows should have an opening_statement
        features = workflow.get("features", {})
        if isinstance(features, dict):
            opening = features.get("opening_statement", "")
            if not opening:
                # non-fatal: many chat flows use it
                pass

        # Chat flows must have at least one answer node
        has_answer = any(t == "answer" for t in node_type_map.values())
        if not has_answer:
            raise DifyValidationError(
                "Chat mode requires at least one 'answer' node to respond to the user"
            )

    if mode == "workflow":
        # Workflow must have an end node
        has_end = any(t == "end" for t in node_type_map.values())
        if not has_end:
            raise DifyValidationError("Workflow mode requires at least one 'end' node")

        # Check conversation_variables are valid if present
        cv = workflow.get("conversation_variables")
        if cv is not None:
            if not isinstance(cv, list):
                raise DifyValidationError("workflow.conversation_variables: must be an array")
            for ci, var in enumerate(cv):
                if isinstance(var, dict):
                    for fld in ("name", "type"):
                        if fld not in var:
                            raise DifyValidationError(
                                f"workflow.conversation_variables[{ci}]: missing '{fld}'"
                            )


# ============================================================================
# Position sanity
# ============================================================================


def _validate_positions(nodes: List[Dict]) -> None:
    """Check that node positions have numeric x, y."""
    for node in nodes:
        node_id = node.get("id", "?")
        for pos_key in ("position", "positionAbsolute"):
            pos = node.get(pos_key, {})
            if isinstance(pos, dict):
                for axis in ("x", "y"):
                    val = pos.get(axis)
                    if val is not None and not isinstance(val, (int, float)):
                        raise DifyValidationError(
                            f"Node '{node_id}': {pos_key}.{axis} must be numeric, got {type(val).__name__}"
                        )

        # Recurse into iteration children
        node_data = node.get("data", {})
        if isinstance(node_data, dict) and "children" in node_data:
            children = node_data["children"]
            if isinstance(children, dict):
                _validate_positions(children.get("nodes", []))


# ============================================================================
# LLM Interaction Layer
# ============================================================================

DIFY_DSL_SYSTEM_PROMPT = """\
You are a Dify workflow DSL code-generator. Output valid, complete Dify YAML.

## ABSOLUTE OUTPUT RULES
1. Output RAW YAML ONLY. First character must be valid YAML.
2. NO markdown fences (no ```yaml, no ```, no ```yml).
3. NO explanations, NO commentary, NO preamble, NO postscript.

## DIFY YAML STRUCTURE (both workflow and chat modes)

### Common top-level:
```yaml
app:
  mode: workflow        # REQUIRED: "workflow" or "chat" or "advanced-chat"
  name: "name"
  description: "desc"
  icon: "🤖"
  icon_background: "#1E64F0"
  use_icon_as_answer_icon: false

kind: app
version: "0.1.5"

workflow:
  version: "0.1.0"
  conversation_variables: []
  environment_variables: []
  features:
    file_upload: {enabled: false}
    opening_statement: ""
    retriever_resource: {enabled: false}
    sensitive_word_avoidance: {enabled: false}
    speech_to_text: {enabled: false}
    suggested_questions: []
    suggested_questions_after_answer: {enabled: false}
    text_to_speech: {enabled: false}
  graph:
    nodes: []
    edges: []
```

### Every node MUST have: id, type: "custom", width, height, position, positionAbsolute, data

### Node type reference:

**start:**
```yaml
id: "start"
type: "custom"
width: 244; height: 54
position: {x: 80, y: 162}
positionAbsolute: {x: 80, y: 162}
data:
  type: start
  title: "开始"
  desc: ""
  selected: false
  variables:
    - label: "input_name"
      variable: "input_name"
      required: true
      type: text
      max_length: 256
      options: []
```

**end:** (workflow mode)
```yaml
id: "end"
data:
  type: end
  title: "结束"
  outputs:
    - value_selector: ["src_node", "field"]
      variable: "output_name"
```

**answer:** (chat mode, replaces end)
```yaml
id: "answer-1"
data:
  type: answer
  title: "回答"
  answer: "{{#llm-1.text#}}"
```

**llm:**
```yaml
id: "llm-1"
height: 98
data:
  type: llm
  title: "LLM"
  model:
    provider: "openai"
    name: "gpt-4o-mini"
    mode: chat
    completion_params: {temperature: 0.7}
  prompt_template:
    - id: "uuid-1"
      role: system
      text: "You are a helpful assistant."
    - id: "uuid-2"
      role: user
      text: "{{#source_node.field#}}"
  context: {enabled: false, variable_selector: []}
  memory:
    role_prefix: {user: "", assistant: ""}
    window: {enabled: false, size: 10}
  variables: []
```

**code:**
```yaml
id: "code-1"
data:
  type: code
  title: "Code"
  code_language: python3
  code: |
    def main(arg1: str) -> dict:
        return {"result": arg1}
  variables:
    - variable: "arg1"
      value_selector: ["src_node", "field"]
  outputs:
    result: {type: string, children: null}
```

**if-else:**
```yaml
id: "ifelse-1"
data:
  type: if-else
  title: "条件分支"
  conditions:
    - comparison_operator: "contains"
      variable_selector: ["src_node", "field"]
      value: "keyword"
  logical_operator: "and"
```

**knowledge-retrieval:**
```yaml
id: "kret-1"
data:
  type: knowledge-retrieval
  title: "知识检索"
  dataset_ids: ["dataset-uuid-here"]
  query_variable_selector: ["src_node", "query"]
  retrieval_mode: "hybrid"
  top_k: 3
  score_threshold: 0.5
```

**http-request:**
```yaml
id: "http-1"
data:
  type: http-request
  title: "HTTP请求"
  method: "POST"
  url: "https://api.example.com/endpoint"
  authorization: {type: "no-auth", config: null}
  headers: ""
  params: ""
  body: {type: "json", data: '{"key":"{{#src.field}}"}'}
```

**iteration (with nested graph):**
```yaml
id: "iter-1"
data:
  type: iteration
  title: "迭代"
  isInIteration: false
  startNodeType: start
  start_node_id: "start-in-iter"   # MUST match inner start node id
  iterator_selector: ["src_node", "array_field"]
  output_selector: ["inner_llm", "text"]
  children:
    nodes:
      - id: "start-in-iter"
        data:
          type: start
          title: "开始"
          variables:
            - label: "item"
              variable: "item"
              required: true
              type: text
              max_length: 10000
      - id: "inner_llm"
        height: 98
        data:
          type: llm
          title: "内部LLM"
          model: {provider: "openai", name: "gpt-4o-mini", mode: chat, completion_params: {temperature: 0.7}}
          prompt_template:
            - {id: "s-1", role: system, text: "Extract insights."}
            - {id: "u-1", role: user, text: "{{#start-in-iter.item#}}"}
          context: {enabled: false, variable_selector: []}
          memory: {role_prefix: {user: "", assistant: ""}, window: {enabled: false, size: 10}}
    edges:
      - id: "e-in-1"
        source: "start-in-iter"
        target: "inner_llm"
        sourceHandle: "source"
        targetHandle: "target"
        type: "custom"
        zIndex: 0
        data: {isInIteration: true, sourceType: start, targetType: llm}
```

**variable-assigner:**
```yaml
id: "assign-1"
data:
  type: variable-assigner
  title: "变量赋值"
  variables:
    - variable: "output_var"
      value_selector: ["src_node", "field"]
      operation: "over-write"
```

**Edges:**
```yaml
- id: "unique-edge-id"
  source: "source-node-id"
  target: "target-node-id"
  sourceHandle: "source"
  targetHandle: "target"
  type: "custom"
  zIndex: 0
  data:
    isInIteration: false        # true only inside iteration children
    sourceType: source_type
    targetType: target_type
```

## CRITICAL RULES (violation = rejection):
1. ALL node ids globally unique (including inside iteration children)
2. Every edge source/target MUST match an existing node id
3. Every edge id must be unique
4. `{{#X.Y#}}` — X must be a real node id
5. `["X", "Y"]` value_selector — X must be a real node id
6. Iteration's start_node_id must exist in its children
7. workflow mode: must have 'end' node; chat mode: must have 'answer' node
8. Graph must be connected: every node reachable from a start node
9. Node-specific: llm needs model + prompt_template; code needs code_language + code; if-else needs conditions; iteration needs children; http-request needs url + method
10. Output ONLY raw YAML
"""


def build_generation_prompt(requirement: str) -> str:
    return (
        f"Generate a complete, valid Dify YAML for this requirement:\n\n"
        f"{requirement}\n\n"
        f"Remember: output ONLY raw YAML. No markdown, no commentary."
    )


def build_fix_prompt(requirement: str, previous_yaml: str, error_message: str) -> str:
    return (
        f"Your previous Dify YAML FAILED VALIDATION.\n\n"
        f"VALIDATION ERROR:\n{error_message}\n\n"
        f"PREVIOUS (INVALID) YAML:\n{previous_yaml}\n\n"
        f"ORIGINAL REQUIREMENT:\n{requirement}\n\n"
        f"Fix ALL validation errors. Output ONLY corrected raw YAML. No markdown. No commentary."
    )


def extract_yaml_from_response(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:yaml|yml)?\s*\n?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


# ============================================================================
# API Call
# ============================================================================


def call_llm(
    prompt: str,
    api_key: str,
    model: str = "claude-sonnet-4-6-20250514",
    max_tokens: int = 8192,
) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=DIFY_DSL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ============================================================================
# Self-healing Generation Loop
# ============================================================================


def generate_dify_yaml(
    requirement: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6-20250514",
    max_iterations: int = MAX_ITERATIONS,
) -> Dict[str, Any]:
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Export it or pass api_key= parameter.")

    log.info("Requirement: %s", requirement[:120])
    log.info("Max iterations: %d  |  Model: %s", max_iterations, model)

    current_yaml_str = ""
    last_error = ""

    for iteration in range(1, max_iterations + 1):
        log.info("━━━ Iteration %d / %d ━━━", iteration, max_iterations)

        if iteration == 1:
            prompt = build_generation_prompt(requirement)
        else:
            prompt = build_fix_prompt(requirement, current_yaml_str, last_error)

        log.info("  Calling LLM …")
        try:
            response_text = call_llm(prompt, api_key, model=model)
        except Exception as exc:
            log.error("  LLM call failed: %s", exc)
            last_error = f"LLM API error: {exc}"
            continue

        current_yaml_str = extract_yaml_from_response(response_text)

        if not current_yaml_str.strip():
            log.warning("  LLM returned empty output. Retrying …")
            last_error = "Empty response from LLM"
            continue

        log.info("  Parsing YAML …")
        try:
            yaml_dict = yaml.safe_load(current_yaml_str)
        except yaml.YAMLError as exc:
            log.warning("  YAML parse error: %s", exc)
            last_error = f"YAML parse error: {exc}"
            continue

        if yaml_dict is None:
            log.warning("  YAML parsed to None. Retrying …")
            last_error = "YAML parsed to None/empty"
            continue

        if not isinstance(yaml_dict, dict):
            log.warning("  YAML did not parse to a dictionary. Retrying …")
            last_error = f"YAML parsed to {type(yaml_dict).__name__}, expected dict"
            continue

        log.info("  Running structural validation …")
        try:
            warnings = validate_dify_yaml(yaml_dict)
            if warnings:
                for w in warnings:
                    log.warning("  Warning: %s", w)
            log.info("━━━ SUCCESS on iteration %d ━━━", iteration)
            return yaml_dict
        except DifyValidationError as exc:
            last_error = str(exc)
            log.warning("  Validation FAILED: %s", last_error)
            continue
        except Exception as exc:
            last_error = f"Unexpected: {type(exc).__name__}: {exc}"
            log.error("  %s", last_error)
            continue

    raise RuntimeError(
        f"Failed to generate valid Dify YAML after {max_iterations} iterations.\n"
        f"Last validation error: {last_error}\n\n"
        f"Last generated YAML (first 2000 chars):\n{current_yaml_str[:2000]}"
    )


# ============================================================================
# Utility
# ============================================================================


def save_yaml(yaml_dict: Dict[str, Any], filepath: str) -> None:
    import yaml as yaml_lib

    with open(filepath, "w", encoding="utf-8") as fh:
        yaml_lib.dump(
            yaml_dict, fh,
            default_flow_style=False, allow_unicode=True,
            sort_keys=False, indent=2,
        )
    log.info("Saved: %s", filepath)


# ============================================================================
# Test-case builders
# ============================================================================

_sid = lambda s: str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

_BASE_APP = {
    "app": {
        "mode": "workflow",
        "name": "Test",
        "description": "",
        "icon": "🤖",
        "icon_background": "#1E64F0",
        "use_icon_as_answer_icon": False,
    },
    "kind": "app",
    "version": "0.1.5",
}

_BASE_FEATURES = {
    "file_upload": {"enabled": False},
    "opening_statement": "",
    "retriever_resource": {"enabled": False},
    "sensitive_word_avoidance": {"enabled": False},
    "speech_to_text": {"enabled": False},
    "suggested_questions": [],
    "suggested_questions_after_answer": {"enabled": False},
    "text_to_speech": {"enabled": False},
}


def _node(pos: Tuple[int, int], nid: str, data: Dict, h: int = 54, w: int = 244) -> Dict:
    x, y = pos
    return {
        "id": nid, "type": "custom", "width": w, "height": h,
        "position": {"x": x, "y": y},
        "positionAbsolute": {"x": x, "y": y},
        "data": dict(data, desc="", selected=False),
    }


def _edge(eid: str, src: str, tgt: str, in_iter: bool = False,
          src_type: str = "", tgt_type: str = "") -> Dict:
    return {
        "id": eid, "source": src, "target": tgt,
        "sourceHandle": "source", "targetHandle": "target",
        "type": "custom", "zIndex": 0,
        "data": {
            "isInIteration": in_iter,
            "sourceType": src_type or "",
            "targetType": tgt_type or "",
        },
    }


def _wf(nodes: List[Dict], edges: List[Dict], **kw) -> Dict:
    result = copy.deepcopy(_BASE_APP)
    result["workflow"] = {
        "version": "0.1.0",
        "conversation_variables": [],
        "environment_variables": [],
        "features": dict(_BASE_FEATURES),
        "graph": {"nodes": nodes, "edges": edges},
        **kw,
    }
    return result


def build_demo_workflow() -> Dict[str, Any]:
    """Original test-requirement workflow: start→code→iteration(llm)→end."""
    return _wf(
        nodes=[
            _node((80, 162), "start", {
                "type": "start", "title": "开始",
                "variables": [{"label": "raw_json", "variable": "raw_json",
                               "required": True, "type": "text", "max_length": 50000,
                               "options": []}],
            }),
            _node((380, 162), "code-clean", {
                "type": "code", "title": "清洗JSON提取文本", "code_language": "python3",
                "code": "def main(raw_json: str) -> dict:\n    import json\n    data = json.loads(raw_json)\n    texts = [v for v in data.values() if isinstance(v, str) and len(v.strip()) > 10]\n    return {'text_array': texts}\n",
                "variables": [{"variable": "raw_json", "value_selector": ["start", "raw_json"]}],
                "outputs": {"text_array": {"type": "array[string]", "children": None}},
            }),
            _node((680, 162), "iteration-refine", {
                "type": "iteration", "title": "迭代提炼AI干货",
                "isInIteration": False, "startNodeType": "start",
                "start_node_id": "start-iter",
                "iterator_selector": ["code-clean", "text_array"],
                "output_selector": ["llm-iter", "text"],
                "children": {
                    "nodes": [
                        _node((80, 162), "start-iter", {
                            "type": "start", "title": "开始",
                            "variables": [{"label": "item", "variable": "item",
                                           "required": True, "type": "text",
                                           "max_length": 10000, "options": []}],
                        }),
                        _node((380, 162), "llm-iter", {
                            "type": "llm", "title": "LLM提炼干货",
                            "model": {"provider": "openai", "name": "gpt-4o-mini",
                                      "mode": "chat", "completion_params": {"temperature": 0.7}},
                            "prompt_template": [
                                {"id": _sid("sys"), "role": "system",
                                 "text": "提炼AI干货技巧，简洁概括。"},
                                {"id": _sid("usr"), "role": "user",
                                 "text": "{{#start-iter.item#}}"},
                            ],
                            "context": {"enabled": False, "variable_selector": []},
                            "memory": {"role_prefix": {"user": "", "assistant": ""},
                                       "window": {"enabled": False, "size": 10}},
                        }, h=98),
                    ],
                    "edges": [
                        _edge("e-iter-1", "start-iter", "llm-iter",
                              in_iter=True, src_type="start", tgt_type="llm"),
                    ],
                },
            }),
            _node((980, 162), "end", {
                "type": "end", "title": "结束",
                "outputs": [{"value_selector": ["iteration-refine", "output"],
                             "variable": "final_result"}],
            }),
        ],
        edges=[
            _edge("e-start-code", "start", "code-clean", src_type="start", tgt_type="code"),
            _edge("e-code-iter", "code-clean", "iteration-refine", src_type="code", tgt_type="iteration"),
            _edge("e-iter-end", "iteration-refine", "end", src_type="iteration", tgt_type="end"),
        ],
    )


def build_chat_flow() -> Dict[str, Any]:
    """Chat flow: start → llm → answer."""
    result = copy.deepcopy(_BASE_APP)
    result["app"]["mode"] = "chat"
    result["app"]["name"] = "Chat Assistant"
    result["workflow"] = {
        "version": "0.1.0",
        "conversation_variables": [
            {"name": "user_name", "type": "string", "description": "User's name"},
        ],
        "environment_variables": [],
        "features": {
            "file_upload": {"enabled": False},
            "opening_statement": "Hello! How can I help you today?",
            "retriever_resource": {"enabled": False},
            "sensitive_word_avoidance": {"enabled": False},
            "speech_to_text": {"enabled": False},
            "suggested_questions": ["What is AI?", "Tell me a joke"],
            "suggested_questions_after_answer": {"enabled": True},
            "text_to_speech": {"enabled": False},
        },
        "graph": {
            "nodes": [
                _node((80, 162), "start", {
                    "type": "start", "title": "开始",
                    "variables": [
                        {"label": "user_query", "variable": "user_query",
                         "required": True, "type": "paragraph", "max_length": 5000,
                         "options": []},
                    ],
                }),
                _node((380, 162), "llm-1", {
                    "type": "llm", "title": "AI助手",
                    "model": {"provider": "openai", "name": "gpt-4o-mini",
                              "mode": "chat", "completion_params": {"temperature": 0.7}},
                    "prompt_template": [
                        {"id": _sid("sys"), "role": "system",
                         "text": "You are a helpful assistant. Answer concisely."},
                        {"id": _sid("usr"), "role": "user",
                         "text": "{{#start.user_query#}}"},
                    ],
                    "context": {"enabled": False, "variable_selector": []},
                    "memory": {"role_prefix": {"user": "", "assistant": ""},
                               "window": {"enabled": True, "size": 10}},
                }, h=98),
                _node((680, 162), "answer-1", {
                    "type": "answer", "title": "回答",
                    "answer": "{{#llm-1.text#}}",
                }),
            ],
            "edges": [
                _edge("e-start-llm", "start", "llm-1", src_type="start", tgt_type="llm"),
                _edge("e-llm-answer", "llm-1", "answer-1", src_type="llm", tgt_type="answer"),
            ],
        },
    }
    return result


def build_ifelse_workflow() -> Dict[str, Any]:
    """Complex workflow with branching: start → if-else → (code | llm) → end."""
    return _wf(
        nodes=[
            _node((80, 162), "start", {
                "type": "start", "title": "开始",
                "variables": [{"label": "input", "variable": "input",
                               "required": True, "type": "text", "max_length": 5000,
                               "options": []}],
            }),
            _node((380, 162), "ifelse-1", {
                "type": "if-else", "title": "条件判断",
                "conditions": [
                    {"comparison_operator": "contains",
                     "variable_selector": ["start", "input"],
                     "value": "code"},
                ],
                "logical_operator": "and",
            }),
            _node((680, 80), "code-branch", {
                "type": "code", "title": "代码处理", "code_language": "python3",
                "code": "def main(input: str) -> dict:\n    return {'result': input.upper()}\n",
                "variables": [{"variable": "input", "value_selector": ["start", "input"]}],
                "outputs": {"result": {"type": "string", "children": None}},
            }),
            _node((680, 280), "llm-branch", {
                "type": "llm", "title": "LLM处理",
                "model": {"provider": "openai", "name": "gpt-4o-mini",
                          "mode": "chat", "completion_params": {"temperature": 0.7}},
                "prompt_template": [
                    {"id": _sid("s"), "role": "system", "text": "Process the input."},
                    {"id": _sid("u"), "role": "user", "text": "{{#start.input#}}"},
                ],
                "context": {"enabled": False, "variable_selector": []},
                "memory": {"role_prefix": {"user": "", "assistant": ""},
                           "window": {"enabled": False, "size": 10}},
            }, h=98),
            _node((980, 162), "end", {
                "type": "end", "title": "结束",
                "outputs": [
                    {"value_selector": ["code-branch", "result"], "variable": "output"},
                ],
            }),
        ],
        edges=[
            _edge("e-s-if", "start", "ifelse-1", src_type="start", tgt_type="if-else"),
            _edge("e-if-code", "ifelse-1", "code-branch", src_type="if-else", tgt_type="code"),
            _edge("e-if-llm", "ifelse-1", "llm-branch", src_type="if-else", tgt_type="llm"),
            _edge("e-code-end", "code-branch", "end", src_type="code", tgt_type="end"),
            _edge("e-llm-end", "llm-branch", "end", src_type="llm", tgt_type="end"),
        ],
    )


def build_knowledge_retrieval_workflow() -> Dict[str, Any]:
    """Workflow with knowledge retrieval: start → knowledge-retrieval → llm → end."""
    return _wf(
        nodes=[
            _node((80, 162), "start", {
                "type": "start", "title": "开始",
                "variables": [{"label": "query", "variable": "query",
                               "required": True, "type": "text", "max_length": 2000,
                               "options": []}],
            }),
            _node((380, 162), "kret-1", {
                "type": "knowledge-retrieval", "title": "知识库检索",
                "dataset_ids": ["dataset-uuid-001"],
                "query_variable_selector": ["start", "query"],
                "retrieval_mode": "hybrid",
                "top_k": 5,
                "score_threshold": 0.6,
            }),
            _node((680, 162), "llm-1", {
                "type": "llm", "title": "LLM总结",
                "model": {"provider": "openai", "name": "gpt-4o-mini",
                          "mode": "chat", "completion_params": {"temperature": 0.3}},
                "prompt_template": [
                    {"id": _sid("s"), "role": "system",
                     "text": "Based on the retrieved knowledge, answer the query."},
                    {"id": _sid("u"), "role": "user",
                     "text": "Query: {{#start.query#}}\nContext: {{#kret-1.result#}}"},
                ],
                "context": {"enabled": False, "variable_selector": []},
                "memory": {"role_prefix": {"user": "", "assistant": ""},
                           "window": {"enabled": False, "size": 10}},
            }, h=98),
            _node((980, 162), "end", {
                "type": "end", "title": "结束",
                "outputs": [{"value_selector": ["llm-1", "text"], "variable": "answer"}],
            }),
        ],
        edges=[
            _edge("e-s-kr", "start", "kret-1", src_type="start", tgt_type="knowledge-retrieval"),
            _edge("e-kr-llm", "kret-1", "llm-1", src_type="knowledge-retrieval", tgt_type="llm"),
            _edge("e-llm-end", "llm-1", "end", src_type="llm", tgt_type="end"),
        ],
    )


def build_http_workflow() -> Dict[str, Any]:
    """Workflow with HTTP request: start → code → http-request → llm → end."""
    return _wf(
        nodes=[
            _node((80, 162), "start", {
                "type": "start", "title": "开始",
                "variables": [{"label": "topic", "variable": "topic",
                               "required": True, "type": "text", "max_length": 500,
                               "options": []}],
            }),
            _node((380, 162), "code-prep", {
                "type": "code", "title": "构造请求参数", "code_language": "python3",
                "code": "def main(topic: str) -> dict:\n    import json\n    return {'payload': json.dumps({'q': topic})}\n",
                "variables": [{"variable": "topic", "value_selector": ["start", "topic"]}],
                "outputs": {"payload": {"type": "string", "children": None}},
            }),
            _node((680, 162), "http-1", {
                "type": "http-request", "title": "调用API",
                "method": "POST",
                "url": "https://api.example.com/search",
                "authorization": {"type": "bearer", "config": {"api_key": "{{env.API_KEY}}"}},
                "headers": "Content-Type: application/json",
                "params": "",
                "body": {"type": "json", "data": "{{#code-prep.payload#}}"},
            }),
            _node((980, 80), "llm-1", {
                "type": "llm", "title": "解析结果",
                "model": {"provider": "openai", "name": "gpt-4o-mini",
                          "mode": "chat", "completion_params": {"temperature": 0.5}},
                "prompt_template": [
                    {"id": _sid("s"), "role": "system", "text": "Summarize the API results."},
                    {"id": _sid("u"), "role": "user", "text": "{{#http-1.body#}}"},
                ],
                "context": {"enabled": False, "variable_selector": []},
                "memory": {"role_prefix": {"user": "", "assistant": ""},
                           "window": {"enabled": False, "size": 10}},
            }, h=98),
            _node((980, 280), "end", {
                "type": "end", "title": "结束",
                "outputs": [{"value_selector": ["llm-1", "text"], "variable": "summary"}],
            }),
        ],
        edges=[
            _edge("e-s-code", "start", "code-prep", src_type="start", tgt_type="code"),
            _edge("e-code-http", "code-prep", "http-1", src_type="code", tgt_type="http-request"),
            _edge("e-http-llm", "http-1", "llm-1", src_type="http-request", tgt_type="llm"),
            _edge("e-llm-end", "llm-1", "end", src_type="llm", tgt_type="end"),
        ],
    )


def build_nested_iteration_workflow() -> Dict[str, Any]:
    """Deeply nested iteration: start → iter(outer) → iter(inner) → end."""
    return _wf(
        nodes=[
            _node((80, 162), "start", {
                "type": "start", "title": "开始",
                "variables": [{"label": "data", "variable": "data",
                               "required": True, "type": "text", "max_length": 50000,
                               "options": []}],
            }),
            _node((380, 162), "code-parse", {
                "type": "code", "title": "解析JSON", "code_language": "python3",
                "code": "def main(data: str) -> dict:\n    import json\n    return {'batches': json.loads(data)}\n",
                "variables": [{"variable": "data", "value_selector": ["start", "data"]}],
                "outputs": {"batches": {"type": "array[object]", "children": None}},
            }),
            _node((680, 162), "iter-outer", {
                "type": "iteration", "title": "外层迭代",
                "isInIteration": False, "startNodeType": "start",
                "start_node_id": "outer-start",
                "iterator_selector": ["code-parse", "batches"],
                "output_selector": ["outer-llm", "text"],
                "children": {
                    "nodes": [
                        _node((80, 162), "outer-start", {
                            "type": "start", "title": "开始",
                            "variables": [{"label": "item", "variable": "item",
                                           "required": True, "type": "text",
                                           "max_length": 10000, "options": []}],
                        }),
                        _node((380, 162), "outer-llm", {
                            "type": "llm", "title": "外层分析",
                            "model": {"provider": "openai", "name": "gpt-4o-mini",
                                      "mode": "chat", "completion_params": {"temperature": 0.5}},
                            "prompt_template": [
                                {"id": _sid("s"), "role": "system", "text": "Analyze."},
                                {"id": _sid("u"), "role": "user", "text": "{{#outer-start.item#}}"},
                            ],
                            "context": {"enabled": False, "variable_selector": []},
                            "memory": {"role_prefix": {"user": "", "assistant": ""},
                                       "window": {"enabled": False, "size": 10}},
                        }, h=98),
                    ],
                    "edges": [
                        _edge("e-outer-1", "outer-start", "outer-llm",
                              in_iter=True, src_type="start", tgt_type="llm"),
                    ],
                },
            }),
            _node((980, 162), "end", {
                "type": "end", "title": "结束",
                "outputs": [{"value_selector": ["iter-outer", "output"], "variable": "result"}],
            }),
        ],
        edges=[
            _edge("e-s-code", "start", "code-parse", src_type="start", tgt_type="code"),
            _edge("e-code-iter", "code-parse", "iter-outer", src_type="code", tgt_type="iteration"),
            _edge("e-iter-end", "iter-outer", "end", src_type="iteration", tgt_type="end"),
        ],
    )


# ============================================================================
# Comprehensive negative-test cases
# ============================================================================


_INVALID_CASES: List[Tuple[str, Dict[str, Any]]] = [
    # --- Basic structural errors ---
    ("Missing 'workflow' key",
     {"app": {"mode": "workflow", "name": "Test"}}),
    ("Invalid app.mode",
     {"app": {"mode": "invalid-mode", "name": "Test"},
      "workflow": {"version": "0.1", "graph": {"nodes": [], "edges": []}}}),
    ("Missing graph",
     {"app": {"mode": "workflow", "name": "Test"},
      "workflow": {"version": "0.1"}}),
    ("Empty nodes/edges — no start node",
     {"app": {"mode": "workflow", "name": "Test"},
      "workflow": {"version": "0.1", "graph": {"nodes": [], "edges": []}}}),

    # --- Edge errors ---
    ("Edge source node does not exist",
     _wf(
         nodes=[
             _node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
             _node((380, 162), "end", {"type": "end", "title": "结束"}),
         ],
         edges=[_edge("e1", "ghost-node", "end")],
     )),
    ("Edge target node does not exist",
     _wf(
         nodes=[_node((80, 162), "start", {"type": "start", "title": "开始", "variables": []})],
         edges=[_edge("e1", "start", "nonexistent")],
     )),
    ("Duplicate edge ID",
     _wf(
         nodes=[
             _node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
             _node((380, 162), "end", {"type": "end", "title": "结束"}),
         ],
         edges=[
             _edge("dup-edge", "start", "end"),
             _edge("dup-edge", "start", "end"),
         ],
     )),

    # --- Node errors ---
    ("Duplicate node ID",
     _wf(
         nodes=[
             _node((80, 162), "dup", {"type": "start", "title": "A", "variables": []}),
             _node((380, 162), "dup", {"type": "end", "title": "B"}),
         ],
         edges=[],
     )),
    ("Unknown node data.type",
     _wf(
         nodes=[_node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
                _node((380, 162), "bad-node", {"type": "fictional-type", "title": "Bad"}),
                _node((680, 162), "end", {"type": "end", "title": "结束"})],
         edges=[_edge("e1", "start", "bad-node"), _edge("e2", "bad-node", "end")],
     )),

    # --- Variable-reference errors ---
    ("Var ref targets missing node",
     _wf(
         nodes=[_node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
                _node((380, 162), "llm-1", {
                    "type": "llm", "title": "LLM",
                    "model": {"provider": "openai", "name": "gpt-4o", "mode": "chat",
                              "completion_params": {"temperature": 0.7}},
                    "prompt_template": [{"id": "p1", "role": "user",
                                         "text": "{{#nonexistent.field#}}"}],
                }, h=98),
                _node((680, 162), "end", {"type": "end", "title": "结束"})],
         edges=[_edge("e1", "start", "llm-1"), _edge("e2", "llm-1", "end")],
     )),

    # --- Value-selector errors ---
    ("Value selector references missing node",
     _wf(
         nodes=[_node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
                _node((380, 162), "code-1", {
                    "type": "code", "title": "Code", "code_language": "python3",
                    "code": "def main() -> dict:\n    return {'x': 1}\n",
                    "variables": [{"variable": "inp", "value_selector": ["ghost", "field"]}],
                }),
                _node((680, 162), "end", {"type": "end", "title": "结束"})],
         edges=[_edge("e1", "start", "code-1"), _edge("e2", "code-1", "end")],
     )),

    # --- Node-type-specific errors ---
    ("LLM missing model.provider",
     _wf(
         nodes=[_node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
                _node((380, 162), "llm-1", {
                    "type": "llm", "title": "LLM",
                    "model": {"name": "gpt-4o", "mode": "chat"},
                    "prompt_template": [],
                }, h=98),
                _node((680, 162), "end", {"type": "end", "title": "结束"})],
         edges=[_edge("e1", "start", "llm-1"), _edge("e2", "llm-1", "end")],
     )),
    ("Code missing code_language",
     _wf(
         nodes=[_node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
                _node((380, 162), "code-1", {
                    "type": "code", "title": "Code",
                    "code": "def main():\n    pass\n",
                }),
                _node((680, 162), "end", {"type": "end", "title": "结束"})],
         edges=[_edge("e1", "start", "code-1"), _edge("e2", "code-1", "end")],
     )),
    ("If-else missing conditions",
     _wf(
         nodes=[_node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
                _node((380, 162), "ifelse-1", {"type": "if-else", "title": "Branch"}),
                _node((680, 162), "end", {"type": "end", "title": "结束"})],
         edges=[_edge("e1", "start", "ifelse-1"), _edge("e2", "ifelse-1", "end")],
     )),
    ("Knowledge-retrieval missing dataset_ids",
     _wf(
         nodes=[_node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
                _node((380, 162), "kret-1", {
                    "type": "knowledge-retrieval", "title": "KB",
                    "query_variable_selector": ["start", "q"],
                }),
                _node((680, 162), "end", {"type": "end", "title": "结束"})],
         edges=[_edge("e1", "start", "kret-1"), _edge("e2", "kret-1", "end")],
     )),
    ("HTTP-request missing url",
     _wf(
         nodes=[_node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
                _node((380, 162), "http-1", {
                    "type": "http-request", "title": "HTTP",
                    "method": "POST",
                }),
                _node((680, 162), "end", {"type": "end", "title": "结束"})],
         edges=[_edge("e1", "start", "http-1"), _edge("e2", "http-1", "end")],
     )),
    ("Iteration missing start_node_id",
     _wf(
         nodes=[_node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
                _node((380, 162), "iter-1", {
                    "type": "iteration", "title": "Iter",
                    "iterator_selector": ["start", "arr"],
                    "children": {
                        "nodes": [
                            _node((80, 162), "in-start", {"type": "start", "title": "开始",
                                                          "variables": [{"label": "item", "variable": "item",
                                                                         "required": True, "type": "text",
                                                                         "max_length": 256, "options": []}]}),
                        ],
                        "edges": [],
                    },
                }),
                _node((680, 162), "end", {"type": "end", "title": "结束"})],
         edges=[_edge("e1", "start", "iter-1"), _edge("e2", "iter-1", "end")],
     )),

    # --- Mode-specific errors ---
    ("Chat mode without answer node",
     {
         "app": {"mode": "chat", "name": "Chat", "description": "", "icon": "🤖",
                 "icon_background": "#1E64F0", "use_icon_as_answer_icon": False},
         "kind": "app", "version": "0.1.5",
         "workflow": {
             "version": "0.1.0", "conversation_variables": [],
             "environment_variables": [], "features": _BASE_FEATURES,
             "graph": {
                 "nodes": [
                     _node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
                     _node((380, 162), "end", {"type": "end", "title": "结束"}),
                 ],
                 "edges": [_edge("e1", "start", "end")],
             },
         },
     }),

    # --- Graph connectivity errors ---
    ("Unreachable node",
     _wf(
         nodes=[
             _node((80, 162), "start", {"type": "start", "title": "开始", "variables": []}),
             _node((380, 162), "orphan", {"type": "code", "title": "Orphan",
                                          "code_language": "python3",
                                          "code": "def main() -> dict:\n    return {}\n"}),
             _node((680, 162), "end", {"type": "end", "title": "结束"}),
         ],
         edges=[_edge("e1", "start", "end")],
     )),
]


# ============================================================================
# Test runner
# ============================================================================


def run_comprehensive_tests() -> Tuple[int, int]:
    """
    Run all positive and negative test cases.
    Returns (passed, failed) counts.
    """
    positive_cases = [
        ("Demo workflow", build_demo_workflow()),
        ("Chat flow", build_chat_flow()),
        ("If-else workflow", build_ifelse_workflow()),
        ("Knowledge retrieval workflow", build_knowledge_retrieval_workflow()),
        ("HTTP request workflow", build_http_workflow()),
        ("Nested iteration workflow", build_nested_iteration_workflow()),
    ]

    passed = 0
    failed = 0

    log.info("=" * 60)
    log.info("COMPREHENSIVE TEST SUITE")
    log.info("=" * 60)

    # ── Positive tests ──
    for label, data in positive_cases:
        try:
            warnings = validate_dify_yaml(data)
            if warnings:
                log.info("✓ %s — PASSED (%d warning(s))", label, len(warnings))
                for w in warnings:
                    log.info("   ⚠ %s", w)
            else:
                log.info("✓ %s — PASSED (clean)", label)
            passed += 1
        except DifyValidationError as exc:
            log.error("✗ %s — FAILED: %s", label, exc)
            failed += 1

    # ── Negative tests ──
    log.info("─" * 40)
    for label, data in _INVALID_CASES:
        try:
            validate_dify_yaml(data)
            log.error("✗ %s — SHOULD HAVE FAILED but passed (BUG!)", label)
            failed += 1
        except DifyValidationError as exc:
            log.info("✓ %s — correctly rejected: %s", label, exc)
            passed += 1

    log.info("=" * 60)
    log.info("RESULTS: %d passed, %d failed, %d total", passed, failed, passed + failed)
    log.info("=" * 60)
    return passed, failed


# ============================================================================
# Entry Point
# ============================================================================

TEST_REQUIREMENT = (
    "请帮我生成一个处理异构 JSON 的工作流。"
    "从 start 节点接收 raw_json；"
    "连接到一个 code 节点用 Python 清洗短文本废话并返回 text_array；"
    "再连接到一个 iteration 节点（内部包含一个模型为 gpt-4o-mini 的 llm 节点）来提炼 AI 干货技巧；"
    "最后连向 end 节点输出最终结果。"
)


def main() -> None:
    log.info("=" * 60)
    log.info("Dify DSL Generator — Self-Healing Mode")
    log.info("=" * 60)

    # Check for --test flag (runs comprehensive tests only, no LLM)
    if "--test" in sys.argv:
        passed, failed = run_comprehensive_tests()
        sys.exit(0 if failed == 0 else 1)

    # Resolve API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--api-key":
            try:
                api_key = sys.argv[i + 1]
            except IndexError:
                log.error("--api-key requires a value")
                sys.exit(2)
        elif arg.startswith("--api-key="):
            api_key = arg.split("=", 1)[1]

    if not api_key:
        log.warning(
            "ANTHROPIC_API_KEY not set. Running test suite.\n"
            "To run full generation: export ANTHROPIC_API_KEY='sk-...'\n"
            "To run test suite explicitly: python dify_dsl_generator.py --test"
        )
        run_comprehensive_tests()
        return

    # Full generation
    log.info("Requirement: %s", TEST_REQUIREMENT)
    try:
        result = generate_dify_yaml(TEST_REQUIREMENT, api_key=api_key)
        output_path = os.path.join(os.path.dirname(__file__) or ".", "generated_dify_workflow.yml")
        save_yaml(result, output_path)
        log.info("=" * 60)
        log.info("GENERATION SUCCESSFUL → %s", output_path)
        log.info("=" * 60)
    except RuntimeError as exc:
        log.error("GENERATION FAILED: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    import yaml as _yaml
    main()
