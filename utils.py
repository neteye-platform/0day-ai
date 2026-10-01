import networkx as nx
import json
import logging


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


async def run_stream(app, inputs):
    final_state = None
    
    # Request BOTH messages (for token streaming) and values (for state updates)
    async for event_type, data in app.astream(
        inputs,
        stream_mode=["messages", "values"]
    ):
        if event_type == "messages":
            # Unpack the chunk and metadata
            chunk, metadata = data
            
            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                # Print reasoning tokens in gray (using ANSI escape codes)
                print(f"\033[90m{reasoning}\033[0m", end="", flush=True)
            elif chunk.content:
                # Print actual answer in default terminal color
                print(chunk.content, end="", flush=True)
            elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    tool_name = tool_call.get("name", "UnknownTool")
                    tool_args = tool_call.get("args", {})

                    # Print the tool name in cyan
                    print(f"\n\033[96m[Tool Call Intercepted] -> {tool_name}\033[0m")
                    # Pretty-print the structured arguments
                    print(f"\033[96m{json.dumps(tool_args, indent=2)}\033[0m\n", flush=True)
                    
        elif event_type == "values":
            # The "values" stream yields the entire state dictionary every time a node finishes.
            # We continuously overwrite final_state so that when the loop finishes, 
            # it holds the absolute final state of the graph.
            final_state = data

    return final_state


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
