import re
from functools import lru_cache

import tree_sitter
import tree_sitter_go
import tree_sitter_javascript
import tree_sitter_php
import tree_sitter_python
import tree_sitter_typescript

LANGUAGE_MAP = {
    ".py": tree_sitter.Language(tree_sitter_python.language()),
    ".js": tree_sitter.Language(tree_sitter_javascript.language()),
    ".jsx": tree_sitter.Language(tree_sitter_javascript.language()),
    ".php": tree_sitter.Language(tree_sitter_php.language_php()),
    ".ts": tree_sitter.Language(tree_sitter_typescript.language_typescript()),
    ".tsx": tree_sitter.Language(tree_sitter_typescript.language_tsx()),
    ".go": tree_sitter.Language(tree_sitter_go.language()),
    # .vue single-file components are parsed with the TypeScript grammar: non-
    # <script> regions are masked out (newlines preserved) before parsing, so the
    # script body — raw JS or TS — parses as a normal program whose line numbers
    # stay SFC-accurate (see utils.masked_source_for_parsing). TS is a superset
    # of JS, so it is the safe default regardless of the declared script lang.
    ".vue": tree_sitter.Language(tree_sitter_typescript.language_typescript()),
}

# Dependency manifest and lockfile names across all supported ecosystems.
# These files are already handled by the SCA layer (osv-scanner), so the
# LLM explorers should never spend budget re-analyzing them.
MANIFEST_NAMES = {
    # Python / Conda
    "requirements.txt",
    "requirements.in",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "pyproject.toml",
    "environment.yml",
    "conda.yaml",
    "conda-lock.yml",
    # Node / JavaScript
    "package.json",
    "packages.json",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lock",
    "bun.lockb",
    "deno.lock",
    "deno.json",
    "deno.jsonc",
    # JVM
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "gradle.lockfile",
    "gradle/libs.versions.toml",
    "ivy.xml",
    # Go
    "go.mod",
    "go.sum",
    # Ruby
    "Gemfile",
    "Gemfile.lock",
    "gems.rb",
    "gems.locked",
    # PHP
    "composer.json",
    "composer.lock",
    # Rust
    "Cargo.toml",
    "Cargo.lock",
    # Dart
    "pubspec.yaml",
    "pubspec.lock",
    # Elixir
    "mix.exs",
    "mix.lock",
    # Swift / Objective-C
    "Package.swift",
    "Package.resolved",
    "Podfile",
    "Podfile.lock",
    # .NET
    "packages.config",
    "packages.lock.json",
    "project.json",
    "project.lock.json",
    "global.json",
    # C / C++
    "vcpkg.json",
    "conanfile.txt",
    "conanfile.py",
    "conan.lock",
    "CMakeLists.txt",
}

# Define the AST mappings for the languages your agents will scan
AST_GRAMMAR_MAP = {
    ".py": {
        "keep_whole": [
            "import_statement",
            "import_from_statement",
            "expression_statement",
        ],
        "prune_bodies": [
            "function_definition",
            "class_definition",
            "decorated_definition",
        ],
        "body_node": "block",
        "comment": "#",
        "container_nodes": ["class_definition"],
        "import_query": """
            (import_statement (dotted_name) @import)
            (import_from_statement module_name: (dotted_name) @import)
        """,
        "import_separator": ".",
    },
    ".js": {
        "keep_whole": [
            "import_statement",
            "lexical_declaration",
            "variable_declaration",
        ],
        "prune_bodies": [
            "function_declaration",
            "class_declaration",
            "arrow_function",
            "method_definition",
        ],
        "body_node": "statement_block",
        "comment": "//",
        "container_nodes": ["class_declaration"],
        "import_query": "(import_statement source: (string) @import)",
        "import_separator": "/",
    },
    ".go": {
        "keep_whole": ["import_declaration"],
        "prune_bodies": ["function_declaration", "method_declaration"],
        "body_node": "block",
        "comment": "//",
        "container_nodes": [],  # Go methods attach to structs, but aren't nested inside them in the AST
        "import_query": "(import_spec path: (_) @import)",
        "import_separator": "/",
    },
    ".php": {
        "keep_whole": [
            "namespace_definition",
            "namespace_use_declaration",
            "expression_statement",
        ],
        "prune_bodies": [
            "function_definition",
            "method_declaration",
            "class_declaration",
            "trait_declaration",
            "interface_declaration",
        ],
        "body_node": ["compound_statement", "declaration_list", "block"],
        "comment": "//",
        "container_nodes": [
            "class_declaration",
            "trait_declaration",
            "interface_declaration",
            "namespace_definition",
        ],
        "import_query": "(namespace_use_clause) @import",
        "import_separator": "\\",
    },
    ".ts": {
        "keep_whole": [
            "import_statement",
            "lexical_declaration",
            "variable_declaration",
            "type_alias_declaration",
        ],
        "prune_bodies": [
            "function_declaration",
            "class_declaration",
            "arrow_function",
            "method_definition",
            "interface_declaration",
            "module",
        ],
        "body_node": ["statement_block", "class_body", "object_type", "module_block"],
        "comment": "//",
        "container_nodes": ["class_declaration", "interface_declaration", "module"],
        "import_query": "(import_statement source: (string) @import)",
        "import_separator": "/",
    },
    ".tsx": {
        "keep_whole": [
            "import_statement",
            "lexical_declaration",
            "variable_declaration",
            "type_alias_declaration",
        ],
        "prune_bodies": [
            "function_declaration",
            "class_declaration",
            "arrow_function",
            "method_definition",
            "interface_declaration",
            "module",
        ],
        "body_node": ["statement_block", "class_body", "object_type", "module_block"],
        "comment": "//",
        "container_nodes": ["class_declaration", "interface_declaration", "module"],
        "import_query": "(import_statement source: (string) @import)",
        "import_separator": "/",
    },
}

# .vue SFCs reuse the TypeScript AST structure end-to-end: the non-<script>
# regions are blanked into spaces before parsing, so the tree-sitter tree (and
# thus every AST_GRAMMAR_MAP/SYMBOL_QUERIES rule below) only ever sees the
# script body as a regular TS program.
AST_GRAMMAR_MAP[".vue"] = AST_GRAMMAR_MAP[".ts"]

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
    ".py": """
        (class_definition name: (identifier) @class_name superclasses: (argument_list (_) @parent_class) body: (block (function_definition name: (identifier) @method_name) @method_body))
        (class_definition name: (identifier) @class_name body: (block (function_definition name: (identifier) @method_name) @method_body))
        (function_definition name: (identifier) @function_name) @function_body
    """,
    ".js": """
        (class_declaration name: (identifier) @class_name body: (class_body (method_definition name: (property_identifier) @method_name) @method_body))
        (function_declaration name: (identifier) @function_name) @function_body
        (lexical_declaration (variable_declarator name: (identifier) @function_name value: [(arrow_function) (function_expression)] @function_body))
    """,
    ".ts": """
        (class_declaration name: (type_identifier) @class_name body: (class_body (method_definition name: (property_identifier) @method_name) @method_body))
        (function_declaration name: (identifier) @function_name) @function_body
        (lexical_declaration (variable_declarator name: (identifier) @function_name value: [(arrow_function) (function_expression)] @function_body))
    """,
    ".tsx": """
        (class_declaration name: (type_identifier) @class_name body: (class_body (method_definition name: (property_identifier) @method_name) @method_body))
        (function_declaration name: (identifier) @function_name) @function_body
        (lexical_declaration (variable_declarator name: (identifier) @function_name value: [(arrow_function) (function_expression)] @function_body))
    """,
    ".go": """
        (method_declaration
          receiver: (parameter_list (parameter_declaration type: (type_identifier) @class_name))
          name: (field_identifier) @method_name
          body: (block) @method_body)
        (method_declaration
          receiver: (parameter_list (parameter_declaration type: (pointer_type (type_identifier) @class_name)))
          name: (field_identifier) @method_name
          body: (block) @method_body)
        (function_declaration name: (identifier) @function_name body: (block) @function_body)
    """,
}

# .jsx and .tsx use the exact same AST structure for methods as their base languages
SYMBOL_QUERIES[".jsx"] = SYMBOL_QUERIES[".js"]
SYMBOL_QUERIES[".vue"] = SYMBOL_QUERIES[".ts"]

# Decision-guard extraction ("how is this function used?"): calls appearing
# inside if/else-if conditions or assigned (as boolean/guard-named expressions)
# to variables are surfaced as usage contexts, e.g. 'isAPI() (participates in:
# $check_mfa)'. GUARD_SPEC holds the per-language AST node types; missing keys
# mean the language has no guard analysis (format_node_context fails open).

GUARD_VAR_REGEX = re.compile(
    r"(?i)(auth|mfa|2fa|check|allow|deny|valid|perm|role|access|guard|admin|is_)"
)

_PHP_GUARD_SPEC = {
    "call_fields": {
        "function_call_expression": "function",
        "scoped_call_expression": "name",
        "member_call_expression": "name",
    },
    "condition_nodes": {"if_statement": "condition", "else_if_clause": "condition"},
    "assignment_nodes": {"assignment_expression": ("left", "right")},
    "boolean_types": (
        "binary_expression",
        "unary_op_expression",
        "parenthesized_expression",
    ),
    "builtins": {
        "isset",
        "empty",
        "count",
        "sizeof",
        "is_array",
        "is_string",
        "is_null",
        "is_numeric",
        "strtolower",
        "strtoupper",
        "trim",
        "explode",
        "implode",
        "in_array",
        "array_key_exists",
        "sprintf",
        "printf",
    },
    "wrap": ("<?php\nclass _SnippetScope {\n", "\n}"),
    "text_node": "text",
}

_PY_GUARD_SPEC = {
    "call_fields": {
        "call": "function",
    },
    "condition_nodes": {"if_statement": "condition"},
    "assignment_nodes": {"assignment": ("left", "right")},
    "boolean_types": (
        "boolean_operator",
        "not_operator",
        "unary_operator",
        "binary_operator",
        "parenthesized_expression",
    ),
    "builtins": {
        "len",
        "isinstance",
        "issubclass",
        "callable",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "print",
        "bool",
        "int",
        "str",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "type",
        "super",
        "vars",
        "dir",
        "id",
        "hash",
        "iter",
        "next",
        "any",
        "all",
        "sum",
        "min",
        "max",
        "abs",
        "sorted",
        "reversed",
        "enumerate",
        "zip",
        "map",
        "filter",
        "range",
        "open",
        "repr",
        "format",
        "classmethod",
        "staticmethod",
        "property",
    },
}

# js/ts/tsx/jsx/vue share one spec: && / || are binary_expression in the
# tree-sitter-javascript grammar, and const/let declarators are the idiomatic
# guard assignments. Builtins are lowercase to match the normalized callees.
_JS_GUARD_SPEC = {
    "call_fields": {
        "call_expression": "function",
    },
    "condition_nodes": {"if_statement": "condition"},
    "assignment_nodes": {
        "assignment_expression": ("left", "right"),
        "variable_declarator": ("name", "value"),
    },
    "boolean_types": (
        "binary_expression",
        "unary_expression",
        "parenthesized_expression",
    ),
    "builtins": {
        "boolean",
        "string",
        "number",
        "bigint",
        "symbol",
        "object",
        "array",
        "json",
        "math",
        "date",
        "regexp",
        "error",
        "typeerror",
        "rangeerror",
        "parseint",
        "parsefloat",
        "isnan",
        "isfinite",
        "encodeuri",
        "encodeuricomponent",
        "decodeuri",
        "decodeuricomponent",
        "require",
    },
    "wrap": ("class _SnippetScope {\n", "\n}"),
}

_GO_GUARD_SPEC = {
    "call_fields": {
        "call_expression": "function",
    },
    "condition_nodes": {"if_statement": "condition"},
    "assignment_nodes": {
        "assignment_statement": ("left", "right"),
        "short_var_declaration": ("left", "right"),
    },
    "boolean_types": (
        "binary_expression",
        "unary_expression",
        "parenthesized_expression",
    ),
    "builtins": {
        "len",
        "cap",
        "make",
        "new",
        "append",
        "copy",
        "delete",
        "panic",
        "recover",
        "print",
        "println",
        "close",
        "complex",
        "real",
        "imag",
        "min",
        "max",
        "clear",
    },
}

GUARD_SPEC = {
    ".php": _PHP_GUARD_SPEC,
    ".py": _PY_GUARD_SPEC,
    ".go": _GO_GUARD_SPEC,
}
for _ext in (".js", ".jsx", ".ts", ".tsx", ".vue"):
    GUARD_SPEC[_ext] = _JS_GUARD_SPEC


@lru_cache(maxsize=len(LANGUAGE_MAP))
def _guard_parser(ext: str) -> tree_sitter.Parser:
    return tree_sitter.Parser(LANGUAGE_MAP[ext])


def extract_function_calls(node, call_fields: dict) -> list[str]:
    """Recursively collect bare, lowercased callee names under an AST subtree.

    Uses the language's ``call_fields`` ({node_type: field_name}); the captured
    field text is already the comparable base name (e.g. 'isAPI', 'is2FAEnabled'),
    so no scope/prefix normalization is needed here.
    """
    calls = []

    field = call_fields.get(node.type)
    if field:
        name = node.child_by_field_name(field)
        if name:
            calls.append(name.text.decode("utf-8", errors="ignore").lower())

    for child in node.children:
        calls.extend(extract_function_calls(child, call_fields))

    return calls


def _extract_decision_guards(root_node, spec: dict) -> list[dict]:
    """Extract decision-guard call usages from an AST: calls used inside
    spec condition nodes or assigned (as boolean/guard-named expressions) via
    spec assignment nodes. Returns [{'callee', 'context'}, ...] where ``callee``
    is a bare lowercased name and ``context`` is '$var' or 'if ...'."""
    guards = []
    call_fields = spec["call_fields"]
    condition_nodes = spec["condition_nodes"]
    assignment_nodes = spec["assignment_nodes"]
    boolean_types = spec["boolean_types"]
    builtins = spec["builtins"]

    def collect(expr_node, context: str) -> None:
        for call in extract_function_calls(expr_node, call_fields):
            if call not in builtins:
                guards.append({"callee": call, "context": context})

    def walk(node):
        assignment = assignment_nodes.get(node.type)
        if assignment:
            var_field, value_field = assignment
            left = node.child_by_field_name(var_field)
            right = node.child_by_field_name(value_field)

            if left and right:
                var_name = left.text.decode("utf-8", errors="ignore").lstrip("$")
                if GUARD_VAR_REGEX.search(var_name) or right.type in boolean_types:
                    collect(right, f"${var_name}")

        elif node.type in condition_nodes:
            cond = node.child_by_field_name(condition_nodes[node.type])
            if cond:
                raw_cond = cond.text.decode("utf-8", errors="ignore").strip()
                cond_clean = " ".join(raw_cond.split())
                if len(cond_clean) > 60:
                    cond_clean = cond_clean[:57] + "..."
                collect(cond, f"if {cond_clean}")

        for child in node.children:
            walk(child)

    walk(root_node)
    return guards


def guard_usages(code: str, ext: str) -> list[dict] | None:
    """Extract decision-guard usages from a source slice, or None if unparsable.

    Whole-file slices parse bare; slices that error bare (e.g. class methods)
    are retried wrapped in the language's class shell. A bare result holding
    nothing but text nodes (PHP slices without a <?php tag are silently parsed
    as inline HTML) is treated as a misparse and retried wrapped too. Returns
    None for languages without a guard spec or when both parses fail, else the
    list from _extract_decision_guards (possibly empty).
    """
    spec = GUARD_SPEC.get(ext)
    if spec is None:
        return None
    parser = _guard_parser(ext)

    def usable(root) -> bool:
        if root.has_error:
            return False
        text_node = spec.get("text_node")
        return not (
            text_node and all(child.type == text_node for child in root.children)
        )

    sources = [code]
    wrap = spec.get("wrap")
    if wrap:
        sources.append(f"{wrap[0]}{code}{wrap[1]}")
    for source in sources:
        tree = parser.parse(source.encode("utf-8"))
        if usable(tree.root_node):
            return _extract_decision_guards(tree.root_node, spec)
    return None


# ==========================================
# Node triage: per-language scan-signals config
# ==========================================
#
# tree-sitter node-type declarations consumed by the node-triage machinery in
# utils.py (is_node_worth_scanning / _node_code_is_worth_scanning / _is_pure_type
# / _is_config_only). Like the maps above, these are pure data; the generic
# triage algorithm itself stays in utils.

# tree-sitter node types indicating executable logic (function calls, imports,
# string interpolation, control flow). Nodes exposing none of these are inert.
SCAN_SIGNAL_TYPES: dict[str, set[str]] = {
    ".py": {
        "call",
        "import_statement",
        "import_from_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "with_statement",
        "match_statement",
        "interpolation",
    },
    ".js": {
        "call_expression",
        "new_expression",
        "import_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "switch_statement",
        "try_statement",
        "template_substitution",
    },
    ".jsx": {
        "call_expression",
        "new_expression",
        "import_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "switch_statement",
        "try_statement",
        "template_substitution",
    },
    ".ts": {
        "call_expression",
        "new_expression",
        "import_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "switch_statement",
        "try_statement",
        "template_substitution",
    },
    ".tsx": {
        "call_expression",
        "new_expression",
        "import_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "switch_statement",
        "try_statement",
        "template_substitution",
    },
    # .vue scripts parse with the TypeScript grammar, so they share its signals.
    ".vue": {
        "call_expression",
        "new_expression",
        "import_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "switch_statement",
        "try_statement",
        "template_substitution",
    },
    ".php": {
        "function_call_expression",
        "member_call_expression",
        "scoped_call_expression",
        "object_creation_expression",
        "namespace_use_declaration",
        "include_expression",
        "include_once_expression",
        "require_expression",
        "require_once_expression",
        "echo_statement",
        "if_statement",
        "for_statement",
        "foreach_statement",
        "while_statement",
        "switch_statement",
        "try_statement",
        "encapsed_string",
    },
}

# Import-like declarations are tolerated inside pure type/interface/config nodes
# (they only bring names into scope and do not execute anything by themselves).
IMPORT_TYPES: dict[str, set[str]] = {
    ".py": {"import_statement", "import_from_statement"},
    ".js": {"import_statement"},
    ".jsx": {"import_statement"},
    ".ts": {"import_statement"},
    ".tsx": {"import_statement"},
    ".vue": {"import_statement"},
    ".php": {"namespace_use_declaration"},
}

# Nodes that introduce callable/structured definitions (bodies, classes, types).
DEFINITION_TYPES: set[str] = {
    "function_definition",
    "class_definition",
    "decorated_definition",
    "method_declaration",
    "function_declaration",
    "class_declaration",
    "arrow_function",
    "method_definition",
    "function_expression",
    "lambda",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    "type_alias_statement",
}

# Node types whose names are security-relevant when used as assignment targets.
NAME_NODE_TYPES: set[str] = {
    "assignment",
    "variable_declarator",
    "assignment_expression",
    "property_declaration",
    "property_element",
    "public_field_definition",
    "property_signature",
    "pair",
    "array_element_initializer",
}

MAGIC_METHODS: dict[str, set[str]] = {
    ".py": {
        "__reduce__",
        "__reduce_ex__",
        "__setstate__",
        "__getstate__",
        "__getattr__",
        "__setattr__",
        "__getattribute__",
        "__del__",
        "__delattr__",
        "__enter__",
        "__exit__",
        "__new__",
        "__init__",
        "__call__",
        "__getitem__",
        "__setitem__",
        "__repr__",
        "__str__",
    },
    ".php": {
        "__construct",
        "__destruct",
        "__wakeup",
        "__sleep",
        "__call",
        "__callstatic",
        "__get",
        "__set",
        "__isset",
        "__unset",
        "__tostring",
        "__invoke",
        "__set_state",
        "__clone",
        "__debuginfo",
        "__serialize",
        "__unserialize",
    },
    ".js": set(),
    ".jsx": set(),
    ".ts": set(),
    ".tsx": set(),
}

# ---- utils._is_pure_type() detection ------------------------------------------

# ext -> AST node types that mark a node as "declares types". Languages absent
# from this map never take the type-declaration branch (notably .vue, which
# falls through to the behavioral/default branches exactly as before).
_JS_TS_TYPE_CONSTRUCTS = {
    "type_alias_declaration",
    "interface_declaration",
    "enum_declaration",
}
_JS_TS_PURE_FORBIDDEN = {
    "function_declaration",
    "class_declaration",
    "arrow_function",
    "method_definition",
    "function_expression",
    "assignment",
    "variable_declarator",
    "public_field_definition",
    "pair",
}
PURE_TYPE_CONSTRUCTS: dict[str, set[str]] = {
    ext: _JS_TS_TYPE_CONSTRUCTS for ext in (".js", ".jsx", ".ts", ".tsx")
}
# Extra forbidden node types for the type-declaration branch, unioned with the
# language's executable SCAN_SIGNAL_TYPES minus its IMPORT_TYPES.
PURE_TYPE_FORBIDDEN_TYPES: dict[str, set[str]] = {
    ext: _JS_TS_PURE_FORBIDDEN for ext in (".js", ".jsx", ".ts", ".tsx")
}

# Python branch: any behavioral node type disqualifies a "pure type" node.
BEHAVIORAL_NODE_TYPES: dict[str, set[str]] = {
    ".py": {
        "call",
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "with_statement",
        "match_statement",
        "interpolation",
        "function_definition",
        "lambda",
    },
}
# ext -> standalone type-alias statement types that prove purity on their own.
TYPE_ALIAS_NODE_TYPES: dict[str, set[str]] = {".py": {"type_alias_statement"}}

# PHP branch: interfaces must be runtime-free, and property-only classes count
# as pure DTO shapes.
PHP_INTERFACE_TYPES: set[str] = {"interface_declaration"}
PHP_PROPERTY_TYPES: set[str] = {"property_declaration"}
PHP_METHOD_TYPES: set[str] = {"method_declaration", "function_definition"}

# Raw AST fragments extracted for some languages need a wrapper to parse: PHP
# method/class fragments (as returned by get_node_code) omit the `<?php` tag,
# which tree-sitter needs to avoid parsing everything as plain text. Per ext:
# (insert prefix, detect prefix) — the insert prefix is prepended only when the
# snippet does not already start with the detect prefix.
FRAGMENT_WRAP: dict[str, tuple[str, str]] = {".php": ("<?php\n", "<?")}
