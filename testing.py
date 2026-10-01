import argparse
from pprint import pprint
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import settings
import sqlite3

from main import build_graph
from utils import run_stream

TEST_THREAD_ID = "dev_testing_2"
DB_PATH = "testing_db.sqlite"

def setup_memory():
    return SqliteSaver.from_conn_string(DB_PATH)

def list_checkpoints(args):
    """Command: python testing.py list"""
    with SqliteSaver.from_conn_string(DB_PATH) as memory:
        # We need the compiled app to parse the history intelligently
        app = build_graph(checkpointer=memory)
        config = {"configurable": {"thread_id": TEST_THREAD_ID}}
        
        # get_state_history returns snapshots sorted newest to oldest
        history = list(app.get_state_history(config))
        
        print(f"\n--- Found {len(history)} Checkpoints for '{TEST_THREAD_ID}' ---")
        
        for snapshot in history:
            chk_id = snapshot.config["configurable"]["checkpoint_id"]
            
            # .next tells you what node(s) are scheduled to run NEXT.
            # If the tuple is empty, it means the graph reached END.
            if snapshot.next:
                status = f"Paused before: {snapshot.next}"
            else:
                status = "Graph finished (END)"
                
            print(f"Checkpoint: {chk_id} | {status}")
            
        print("-" * 60)

def get_state(args):
    """Command: python testing.py get <checkpoint_id>"""
    with SqliteSaver.from_conn_string(DB_PATH) as memory:
        app = build_graph(checkpointer=memory)
        
        full_id = resolve_short_id(app, TEST_THREAD_ID, args.checkpoint)
        
        config = {
            "configurable": {
                "thread_id": TEST_THREAD_ID,
                "checkpoint_id": full_id
            }
        }
        
        snapshot = app.get_state(config)
        print(f"\n--- State for Checkpoint: {full_id} ---")
        pprint(snapshot.values)
        print("\n--- Next Scheduled Nodes ---")
        print(snapshot.next)
            
def run_graph(args):
    """Command: python testing.py run [--stop-before <node>] [--checkpoint <id>]"""
    current_thread_id = args.thread if hasattr(args, 'thread') else TEST_THREAD_ID
    
    with SqliteSaver.from_conn_string(DB_PATH) as memory:
        interrupt_list = [args.stop_before] if args.stop_before else None
        app = build_graph(checkpointer=memory, interrupt_before=interrupt_list)
        
        config = {"configurable": {"thread_id": current_thread_id}, "max_concurrency": 3}
        
        # ⚠️ RESOLVE THE SHORT ID HERE
        if args.checkpoint:
            full_id = resolve_short_id(app, current_thread_id, args.checkpoint)
            print(f"\n🚀 Time-Traveling! Resuming from: {full_id}")
            config["configurable"]["checkpoint_id"] = full_id
            input_data = None
        else:
            print(f"\n🚀 Starting fresh run for thread: {current_thread_id}")
            input_data = {
                "graph_path": Path(settings.app_path) / "graphify-out" / "graph.json",
                "app_summary": "",
                "communities_map": {},
                "expert_tasks": [],
                "vulnerability_reports": [],
                "filtered_reports": [],
                "messages": []
            }

        if args.isolate_task is not None:
            # Grab the state we are about to run
            snapshot = app.get_state(config)
            current_tasks = snapshot.values.get("expert_tasks", [])
            
            if not current_tasks:
                print("⚠️  No 'expert_tasks' found in the current state to isolate!")
                return
            elif args.isolate_task < 0 or args.isolate_task >= len(current_tasks):
                print(f"⚠️  Task index {args.isolate_task} is out of bounds. Only {len(current_tasks)} tasks exist (0 to {len(current_tasks)-1}).")
                return
                
            target_task = current_tasks[args.isolate_task]
            print(f"✂️  Isolating task [{args.isolate_task}] out of {len(current_tasks)} total tasks.")
            
            update_config = snapshot.config
            if "checkpoint_ns" not in update_config.get("configurable", {}):
                update_config.setdefault("configurable", {})["checkpoint_ns"] = ""
            
            # Update the state to ONLY contain the targeted task using the patched config
            app.update_state(update_config, {"expert_tasks": [target_task]})
            
            # CRITICAL: update_state creates a new checkpoint. 
            # We must fetch the absolute latest state and use ITS config for the invocation.
            new_snapshot = app.get_state({"configurable": {"thread_id": current_thread_id}})
            
            # Overwrite our main config with the fully hydrated one from the new snapshot
            config = new_snapshot.config
            
            # Ensure input_data is None since we are technically resuming from our state update
            input_data = None

        if interrupt_list:
            print(f"⏸️ Graph will pause BEFORE node: '{args.stop_before}'")

        # result = app.invoke(input_data, config)
        result = run_stream(app, input_data, config=config)
        print("\n--- Execution Complete (or Paused) ---")
        
        new_state = app.get_state({"configurable": {"thread_id": current_thread_id}})
        print(f"\n📍 LATEST Checkpoint ID: {new_state.config['configurable']['checkpoint_id']}")
        print(f"Next to execute: {new_state.next}")
            
def tree_checkpoints(args):
    """Command: python testing.py tree"""
    with SqliteSaver.from_conn_string(DB_PATH) as memory:
        app = build_graph(checkpointer=memory)
        config = {"configurable": {"thread_id": TEST_THREAD_ID}}
        
        # History is returned newest-to-oldest
        history = list(app.get_state_history(config))
        
        if not history:
            print(f"No history found for thread '{TEST_THREAD_ID}'.")
            return
            
        # 1. Build the family tree
        nodes = {}
        for snap in history:
            cid = snap.config["configurable"]["checkpoint_id"]
            raw_ts = str(snap.created_at)
            timestamp = raw_ts[:19].replace("T", " ") if "T" in raw_ts else raw_ts[:19]

            # Safely get the parent ID (root nodes won't have one)
            pid = None
            if snap.parent_config:
                pid = snap.parent_config["configurable"].get("checkpoint_id")
                
            # Determine the label based on what is scheduled next
            next_node = snap.next[0] if snap.next else "END"
            
            nodes[cid] = {
                "id": cid,
                "parent": pid,
                "next_node": next_node,
                "timestamp": timestamp,
                "children": []
            }
            
        # 2. Link children to their parents
        roots = []
        for cid, data in nodes.items():
            pid = data["parent"]
            # If the parent exists in our current thread history, link it
            if pid and pid in nodes:
                nodes[pid]["children"].append(cid)
            else:
                # If there's no parent, this is the start of a branch/thread
                roots.append(cid)
                
        print(f"\n--- Checkpoint Tree for '{TEST_THREAD_ID}' ---")
        
        # 3. Recursive function to print the tree with indentation
        def print_tree(node_id, prefix=""):
            node = nodes[node_id]
            display_id = node["id"] if args.full else node["id"][:8]
            
            print(f"{prefix}└── [{display_id}] Paused before: {node['next_node']} - {node['timestamp']}")
            
            # Since history is newest-first, children were appended newest-first. 
            # We reverse them here so the tree prints top-down chronologically.
            for child_id in reversed(node["children"]):
                # Add 4 spaces of indentation for each level of depth
                print_tree(child_id, prefix + "    ")
        
        # Print all root nodes (usually just one, unless you wiped the DB but kept the thread ID)
        for root_id in reversed(roots):
            print_tree(root_id, "")
            
        print("-" * 60)

def resolve_short_id(app, thread_id, provided_id):
    """Takes a short ID (or full ID) and resolves it to the full Checkpoint UUID."""
    if not provided_id:
        return None
        
    # If they pasted a full UUID, just return it immediately
    if len(provided_id) == 36:
        return provided_id
        
    # Fetch the history for this thread to search for the short ID
    config = {"configurable": {"thread_id": thread_id}}
    history = list(app.get_state_history(config))
    
    matches = []
    for snap in history:
        cid = snap.config["configurable"]["checkpoint_id"]
        if cid.startswith(provided_id):
            matches.append(cid)
            
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(f"Ambiguous short ID '{provided_id}'. Found multiple matches.")
    else:
        raise ValueError(f"No checkpoint found starting with '{provided_id}'.")

def delete_checkpoint(args):
    """Command: python testing.py delete <checkpoint_id>"""
    current_thread_id = args.thread if hasattr(args, 'thread') else TEST_THREAD_ID
    
    with SqliteSaver.from_conn_string(DB_PATH) as memory:
        app = build_graph(checkpointer=memory)
        
        try:
            # 1. Resolve the short ID
            full_id = resolve_short_id(app, current_thread_id, args.checkpoint)
        except Exception as e:
            print(f"Error: {e}")
            return
            
        # 2. Fetch history to figure out the parent/child relationships
        config = {"configurable": {"thread_id": current_thread_id}}
        history = list(app.get_state_history(config))
        
        # Build a map of parent_id -> list of child_ids
        children_map = {}
        for snap in history:
            pid = snap.parent_config["configurable"].get("checkpoint_id") if snap.parent_config else None
            cid = snap.config["configurable"]["checkpoint_id"]
            if pid not in children_map:
                children_map[pid] = []
            children_map[pid].append(cid)
            
        # 3. Recursively find all descendants of the target ID
        to_delete = set([full_id])
        
        def find_descendants(node_id):
            if node_id in children_map:
                for child in children_map[node_id]:
                    to_delete.add(child)
                    find_descendants(child)
                    
        find_descendants(full_id)
        
    # 4. Connect directly to SQLite to nuke the records
    print(f"🗑️  Preparing to delete {len(to_delete)} checkpoints (including children)...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    placeholders = ','.join(['?'] * len(to_delete))
    delete_params = tuple(to_delete)
    
    # Depending on the LangGraph version, it uses a combination of these tables.
    # We attempt to clean all of them to be safe.
    tables = ["checkpoints", "checkpoint_writes", "checkpoint_blobs"]
    
    for table in tables:
        try:
            query = f"DELETE FROM {table} WHERE checkpoint_id IN ({placeholders})"
            cursor.execute(query, delete_params)
        except sqlite3.OperationalError:
            # Table might not exist in this specific version of LangGraph, skip it.
            pass 
            
    conn.commit()
    conn.close()
    
    print(f"✅ Successfully deleted {len(to_delete)} records from the database.")

def main():
    parser = argparse.ArgumentParser(description="LangGraph Testing CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # 'list' command
    parser_list = subparsers.add_parser("list", help="List all checkpoints")
    parser_list.add_argument("-f", "--full", action="store_true", help="Print the full UUID")
    parser_list.set_defaults(func=tree_checkpoints)
    # parser_list.set_defaults(func=list_checkpoints)

    # 'get' command
    parser_get = subparsers.add_parser("get", help="Print state of a specific checkpoint")
    parser_get.add_argument("checkpoint", type=str, help="The checkpoint ID")
    parser_get.set_defaults(func=get_state)

    # 'run' command
    parser_run = subparsers.add_parser("run", help="Run the master graph")
    parser_run.add_argument("-s", "--stop-before", type=str, help="Node to pause before (e.g., manager, expert_agent)", default=None)
    parser_run.add_argument("-c", "--checkpoint", type=str, help="Checkpoint ID to resume from", default=None)
    parser_run.add_argument("-i", "--isolate-task", type=int, help="Isolate a specific expert task by index (0-based) before running", default=None)
    parser_run.set_defaults(func=run_graph)

    # 'delete' command
    parser_del = subparsers.add_parser("delete", help="Delete a checkpoint and all its descendants")
    parser_del.add_argument("checkpoint", type=str, help="The short ID or full ID to delete")
    parser_del.add_argument("-t", "--thread", type=str, help="Thread ID", default="dev_testing_123")
    parser_del.set_defaults(func=delete_checkpoint)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
