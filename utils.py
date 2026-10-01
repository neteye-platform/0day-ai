import networkx as nx
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage


def build_networkx_graph(graph_path: str) -> nx.DiGraph:
    """
    Reads the Graphify JSON output and builds a NetworkX Directed Graph.
    """
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)

    # Initialize a Directed Graph
    G = nx.DiGraph()

    # Add Nodes with their attributes (community, type, file_path, etc.)
    for node in graph_data.get('nodes', []):
        node_id = node.get('id')
        if not node_id:
            continue

        # Copy all other key-value pairs as node attributes
        attributes = {k: v for k, v in node.items() if k != 'id'}
        G.add_node(node_id, **attributes)

    # Add Edges with their attributes (e.g., relationship type like 'calls')
    for edge in graph_data.get('links', []):
        source = edge.get('source')
        target = edge.get('target')
        if not source or not target:
            continue

        # Copy all other key-value pairs as edge attributes
        attributes = {k: v for k, v in edge.items() if k not in ['source', 'target']}
        G.add_edge(source, target, **attributes)

    print(f"Loaded Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def extract_subgraph(G: nx.DiGraph, target_communities: list) -> nx.DiGraph:
    """
    Creates a subgraph containing only the nodes in the target communities,
    plus a 1-hop perimeter of incoming/outgoing connections.
    """
    target_nodes = set()

    # Find all nodes belonging to the assigned communities
    for node_id, data in G.nodes(data=True):
        if str(data.get('community')) in target_communities:
            target_nodes.add(node_id)

    # Include 1-hop neighbors to provide boundary context
    perimeter_nodes = set(target_nodes)
    for node in target_nodes:
        # Add nodes that call into our target community
        perimeter_nodes.update(G.predecessors(node))
        # Add nodes that our target community calls
        perimeter_nodes.update(G.successors(node))

    # Create and return the isolated subgraph
    subgraph = G.subgraph(perimeter_nodes).copy()
    return subgraph


def serialize_and_truncate(obj, max_length=400):
    """
    Recursively parses objects and shortens long strings for clean terminal logging.
    Ensures anything a node outputs is JSON serializable for pretty-printing.
    """
    if isinstance(obj, dict):
        return {k: serialize_and_truncate(v, max_length) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_and_truncate(item, max_length) for item in obj]
    elif isinstance(obj, str):
        # Truncate giant source code dumps or massive summaries
        return obj if len(obj) <= max_length else obj[:max_length] + \
            f" ... [TRUNCATED (showing {max_length}/{len(obj)} bytes)]"
    elif hasattr(obj, "model_dump"):
        # Beautifully unpack Pydantic models (like your ExpertTask)
        return serialize_and_truncate(obj.model_dump(), max_length)
    elif hasattr(obj, "content"):
        # Extract meaningful data from Langchain AIMessage/ToolMessage objects
        rep = {"content": obj.content}
        if hasattr(obj, "tool_calls") and obj.tool_calls:
            rep["tool_calls"] = obj.tool_calls
        return serialize_and_truncate(rep, max_length)
    elif type(obj) in (int, float, bool, type(None)):
        return obj
    else:
        return str(obj)


def run_stream(app, inputs, config=None, vv=False):
    print("\n\033[95m[System]\033[0m Initializing Multi-Agent Analysis...\n")

    # We still need to manually accumulate the reports to return to main.py
    accumulated_state = {"vulnerability_reports": []}

    for event in app.stream(inputs, stream_mode="updates", subgraphs=True, config=config):

        namespace, chunk = event

        for node_name, state_update in chunk.items():
            if vv and node_name == "execute_tools":
                continue

            # 1. Print the Node Header
            print(f"\n" + "-"*60)
            print(f"\033[94m[NODE EXECUTED: {node_name}]\033[0m")
            print("-" * 60)

            # 2. Parse, truncate, and dynamically format the output
            clean_data = serialize_and_truncate(state_update)
            formatted_json = json.dumps(clean_data, indent=2)

            # 3. Print the Output to terminal
            print(f"\033[96m-> State Update (Output):\033[0m")
            print(f"\033[90m{formatted_json}\033[0m")

            # 4. Track vulnerability reports for the final output in main.py
            if "vulnerability_reports" in state_update and not namespace:
                accumulated_state["vulnerability_reports"].extend(state_update["vulnerability_reports"])

    print("\n\033[95m[System]\033[0m Execution Finished.")
    return accumulated_state


def get_agent_logger(role_name: str, target_communities: list) -> logging.Logger:
    """
    Creates or retrieves a logger specific to an agent and its assigned communities.
    Outputs to a dedicated file in the agent_logs/ folder.
    """
    # Create a unique name, e.g., "WebSurfaceAuditor_Comm_1_3"
    comm_str = "_".join(target_communities)
    logger_name = f"{role_name}_Comm_{comm_str}"

    logger = logging.getLogger(logger_name)

    # Prevent adding duplicate handlers if the logger already exists
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Write to a specific file
        file_handler = logging.FileHandler(f"agent_logs/{logger_name}.log", mode='w')
        file_handler.setLevel(logging.DEBUG)

        # Format the log output nicely
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


def get_master_logger() -> logging.Logger:
    """A logger for the Manager and Preprocessor that prints to the console."""
    logger = logging.getLogger("Manager")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('\n\033[94m[%(name)s]\033[0m %(message)s') # Blue text
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
