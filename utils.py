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


def serialize_for_json(obj):
    """
    Recursively parses objects to create a clean, JSON structure.
    Strips heavy LangChain metadata but preserves full content and tool IDs.
    """
    if isinstance(obj, dict):
        # Strip out noisy metadata
        return {
            k: serialize_for_json(v)
            for k, v in obj.items()
            if k not in ["usage_metadata", "response_metadata"]
        }

    elif isinstance(obj, list):
        # Filter out hidden system messages
        parsed_list = [serialize_for_json(item) for item in obj]
        return parsed_list
        return [item for item in parsed_list if item != "[SYSTEM PROMPT HIDDEN]"]

    elif hasattr(obj, "model_dump"):
        # Unpack Pydantic models
        return serialize_for_json(obj.model_dump())

    elif hasattr(obj, "content") and hasattr(obj, "type"):
        if obj.type == "system":
            return "[SYSTEM PROMPT HIDDEN]"

        # Structure the message exactly how you requested
        msg_data = {"type": obj.type}

        if obj.content:
            msg_data["content"] = obj.content

        # If the AI calls a tool, include the call details and ID
        if hasattr(obj, "tool_calls") and obj.tool_calls:
            msg_data["tool_calls"] = obj.tool_calls

        # If this is a tool responding, include its name and the ID it's answering
        if obj.type == "tool":
            if hasattr(obj, "name"):
                msg_data["tool_name"] = obj.name
            if hasattr(obj, "tool_call_id"):
                msg_data["responds_to_id"] = obj.tool_call_id

        return msg_data

    elif type(obj) in (int, float, bool, type(None), str):
        return obj
    else:
        return str(obj)

def run_stream(app, inputs, config=None, output_file="trace.json"):
    print(f"\n[System] Running Multi-Agent Analysis. Assembling state to {output_file}...")

    assembled_states = {}
    raw_main_state = {}

    # Token usage stats
    tracked_msg_ids = set()
    agent_token_stats = {}
    token_stats = {"input": 0, "output": 0, "total": 0}

    for event in app.stream(inputs, stream_mode="values", subgraphs=True, config=config):
        namespace, state = event

        # Format the namespace so it's readable in the JSON
        # The root graph has an empty namespace tuple ()
        if not namespace:
            graph_name = "Main_Graph"
            raw_main_state = state
        else:
            # Subgraphs/Agents have namespaces like ('manager', 'expert', 'b3f1...')
            graph_name = ' -> '.join(namespace)

        if hasattr(state.get("task"), "agent_role"):
            graph_name = graph_name[:17] + f" ({state.get("task").agent_role})"

        # Overwrite the key with the most recent full state.
        assembled_states[graph_name] = serialize_for_json(state)

        # --- TERMINAL PROGRESS INDICATOR ---
        if "messages" in state and state["messages"]:
            last_msg = state["messages"][-1]
            msg_type = getattr(last_msg, "type", "unknown")
            msg_id = getattr(last_msg, "id")

            # Extract content safely, even if it's nested
            content = getattr(last_msg, "content", "")
            if isinstance(content, list):
                content = str(content)

            # Create a clean, single-line snippet
            snippet = (content[:100] + "...") if len(content) > 100 else content
            snippet = snippet.replace('\n', ' ').strip()

            if msg_type == "ai":
                # Track token usgae
                if msg_id and msg_id not in tracked_msg_ids:
                    tracked_msg_ids.add(msg_id)
                    usage = getattr(last_msg, "usage_metadata", {})
                    if usage:
                        in_tok = usage.get("input_tokens", 0)
                        out_tok = usage.get("output_tokens", 0)
                        tot_tok = usage.get("total_tokens", 0)

                        # Initialize agent in stats dictionary if not present
                        if graph_name not in agent_token_stats:
                            agent_token_stats[graph_name] = {"input": 0, "output": 0, "total": 0}

                        agent_token_stats[graph_name]["input"] += in_tok
                        agent_token_stats[graph_name]["output"] += out_tok
                        agent_token_stats[graph_name]["total"] += tot_tok

                        token_stats["input"] += in_tok
                        token_stats["output"] += out_tok
                        token_stats["total"] += tot_tok

                # Print the AI's thought process (Chain of Thought)
                if snippet:
                    print(f"[{graph_name}] \033[96m🧠 AI: {snippet}\033[0m")

                # Print the Tool Call (if it decided to act)
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_strings = []
                    for tc in last_msg.tool_calls:
                        name = tc.get("name", "unknown")
                        args = tc.get("args", {})
                        if name == "SubmitReport":
                            # Omit descriptio to keep logs clean
                            args_str = [a.get("vulnerability_type", "") for a in args.get("findings", [])]
                        else:
                            args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                        tool_strings.append(f"{name}({args_str})")

                    print(f"[{graph_name}] \033[93m🛠️  Calling tool: {' | '.join(tool_strings)}\033[0m")

            elif msg_type == "tool":
                print(f"[{graph_name}] \033[92m✅ Tool executed: {getattr(last_msg, 'name', 'unknown')}\033[0m")

            elif msg_type == "human":
                print(f"[{graph_name}] \033[94m👤 Human: {snippet}\033[0m")

            else:
                print(f"[{graph_name}] \033[90m⚙️  {msg_type.capitalize()} message\033[0m")

        else:
            # Tell us exactly WHICH state keys were updated in the background
            state_keys = ", ".join([k for k in state.keys() if k != "messages"])
            print(f"[{graph_name}] \033[90mState updated: [{state_keys}]\033[0m")
        # -----------------------------------

    # Dump the cohesive final states to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(assembled_states, f, indent=2)

    print("[System] Execution Finished.")
    # --- Print Token Usage Summary ---
    print("\n" + "="*50)
    print("📊 \033[1mToken Usage Summary by Agent\033[0m")
    print("-" * 50)
    
    # Sort agents alphabetically for a cleaner readout (optional, but nice)
    for agent_name in sorted(agent_token_stats.keys()):
        stats = agent_token_stats[agent_name]
        print(f"🔹 \033[96m{agent_name}\033[0m")
        print(f"   In: {stats['input']:,}  |  Out: {stats['output']:,}  |  Total: {stats['total']:,}")
    
    print("-" * 50)
    print("🏆 \033[1mGrand Totals\033[0m")
    print(f"   Input Tokens:  {token_stats['input']:,}")
    print(f"   Output Tokens: {token_stats['output']:,}")
    print(f"   Total Tokens:  \033[95m{token_stats['total']:,}\033[0m")
    print("="*50 + "\n")

    # Extract the vulnerability reports from the main graph to return
    main_state = assembled_states.get("Main_Graph", {})
    return {"vulnerability_reports": raw_main_state.get("vulnerability_reports", [])}
