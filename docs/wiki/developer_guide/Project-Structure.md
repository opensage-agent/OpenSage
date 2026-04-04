# Project Structure

## Directory Overview

```
OpenSage/
├── README.md
├── docs/                    # Docs source (MkDocs)
├── src/
│   └── opensage/              # Core Python package (current layout)
│       ├── agents/          # Base agent + tool loading
│       ├── bash_tools/      # Agent Skills (SKILL.md + scripts/)
│       ├── cli/             # CLI entry points (opensage web / dependency-check)
│       ├── config/          # TOML config system + dataclasses
│       ├── evaluations/     # Benchmarks + evaluation runners
│       ├── features/        # Feature flags / optional behaviors
│       ├── memory/          # Neo4j-backed memory (search/update/tools)
│       ├── plugins/         # ADK plugins
│       ├── sandbox/         # Sandbox backends + initializers
│       ├── sandbox_scripts/ # Scripts invoked inside sandboxes
│       ├── session/         # Session + managers (sandboxes/agents/neo4j/ensemble)
│       ├── templates/       # Default configs + Dockerfiles
│       ├── toolbox/         # Python tool wrappers / MCP toolsets
│       ├── util_agents/     # Utility sub-agents (e.g. memory management)
│       └── utils/           # Shared utilities
├── examples/                # Example agents and configs
├── tests/                   # Unit/integration tests
└── third_party/             # External benchmark/tool dependencies
```
