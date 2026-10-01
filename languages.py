import tree_sitter
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_php

LANGUAGE_MAP = {
    ".py": tree_sitter.Language(tree_sitter_python.language()),
    ".js": tree_sitter.Language(tree_sitter_javascript.language()),
    ".jsx": tree_sitter.Language(tree_sitter_javascript.language()),
    ".php": tree_sitter.Language(tree_sitter_php.language_php()),
    ".ts": tree_sitter.Language(tree_sitter_typescript.language_typescript()),
    ".tsx": tree_sitter.Language(tree_sitter_typescript.language_tsx()),
}

# Define the AST mappings for the languages your agents will scan
AST_GRAMMAR_MAP = {
    ".py": {
        "keep_whole": ["import_statement", "import_from_statement", "expression_statement"],
        "prune_bodies": ["function_definition", "class_definition", "decorated_definition"],
        "body_node": "block",
        "comment": "#",
        "container_nodes": ["class_definition"],
        "import_query": """
            (import_statement (dotted_name) @import)
            (import_from_statement module_name: (dotted_name) @import)
        """,
        "import_separator": "."
    },
    ".js": {
        "keep_whole": ["import_statement", "lexical_declaration", "variable_declaration"],
        "prune_bodies": ["function_declaration", "class_declaration", "arrow_function", "method_definition"],
        "body_node": "statement_block",
        "comment": "//",
        "container_nodes": ["class_declaration"],
        "import_query": "(import_statement source: (string) @import)",
        "import_separator": "/"
    },
    ".go": {
        "keep_whole": ["import_declaration"],
        "prune_bodies": ["function_declaration", "method_declaration"],
        "body_node": "block",
        "comment": "//",
        "container_nodes": [], # Go methods attach to structs, but aren't nested inside them in the AST
        "import_query": "(import_spec path: (_) @import)",
        "import_separator": "/"
    },
    ".php": {
        "keep_whole": ["namespace_definition", "namespace_use_declaration", "expression_statement"],
        "prune_bodies": ["function_definition", "method_declaration", "class_declaration", "trait_declaration", "interface_declaration"],
        "body_node": ["compound_statement", "declaration_list", "block"],
        "comment": "//",
        "container_nodes": ["class_declaration", "trait_declaration", "interface_declaration", "namespace_definition"],
        "import_query": "(namespace_use_clause) @import",
        "import_separator": "\\"
    },
    ".ts": {
        "keep_whole": ["import_statement", "lexical_declaration", "variable_declaration", "type_alias_declaration"],
        "prune_bodies": ["function_declaration", "class_declaration", "arrow_function", "method_definition", "interface_declaration", "module"],
        "body_node": ["statement_block", "class_body", "object_type", "module_block"],
        "comment": "//",
        "container_nodes": ["class_declaration", "interface_declaration", "module"],
        "import_query": "(import_statement source: (string) @import)",
        "import_separator": "/"
    },
    ".tsx": {
        "keep_whole": ["import_statement", "lexical_declaration", "variable_declaration", "type_alias_declaration"],
        "prune_bodies": ["function_declaration", "class_declaration", "arrow_function", "method_definition", "interface_declaration", "module"],
        "body_node": ["statement_block", "class_body", "object_type", "module_block"],
        "comment": "//",
        "container_nodes": ["class_declaration", "interface_declaration", "module"],
        "import_query": "(import_statement source: (string) @import)",
        "import_separator": "/"
    },
}

SYMBOL_QUERIES = {
    ".php": """
        (class_declaration
          name: (name) @class_name
          (base_clause (_) @parent_class)
          body: (declaration_list
            (method_declaration name: (name) @method_name) @method_body
          )
        )

        (class_declaration
          name: (name) @class_name
          body: (declaration_list
            (method_declaration name: (name) @method_name) @method_body
          )
        )

        (function_definition name: (name) @function_name) @function_body
    """,
    # TODO: Capture the Parent Class for the following languages
    ".py": """
        (class_definition name: (identifier) @class_name body: (block (function_definition name: (identifier) @method_name) @method_body))
        (function_definition name: (identifier) @function_name) @function_body
    """,
    ".js": """
        (class_declaration name: (identifier) @class_name body: (class_body (method_definition name: (property_identifier) @method_name) @method_body))
        (function_declaration name: (identifier) @function_name) @function_body
        (lexical_declaration (variable_declarator name: (identifier) @function_name value: [(arrow_function) (function_expression)] @function_body))
    """
}

# .jsx and .tsx use the exact same AST structure for methods as their base languages
SYMBOL_QUERIES[".jsx"] = SYMBOL_QUERIES[".js"]
# SYMBOL_QUERIES[".tsx"] = SYMBOL_QUERIES[".ts"]
