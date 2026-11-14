"""Tiny drop-in subset of LangGraph's StateGraph API, used only when `langgraph` is not installed.
Supports: add_node, add_edge, add_conditional_edges, set_entry_point, compile().invoke(state, config)."""
END = "__end__"


class _Compiled:
    def __init__(self, g): self.g = g

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        limit = (config or {}).get("recursion_limit", 25)
        node, state, steps = self.g.entry, dict(state), 0
        while node != END:
            steps += 1
            if steps > limit:
                raise RuntimeError(f"recursion limit {limit} reached")
            state.update(self.g.nodes[node](state) or {})
            if node in self.g.cond:
                fn, mapping = self.g.cond[node]
                node = mapping[fn(state)]
            else:
                node = self.g.edges[node]
        return state


class StateGraph:
    def __init__(self, _schema=None):
        self.nodes, self.edges, self.cond, self.entry = {}, {}, {}, None

    def add_node(self, name, fn): self.nodes[name] = fn
    def add_edge(self, a, b): self.edges[a] = b
    def add_conditional_edges(self, a, fn, mapping): self.cond[a] = (fn, mapping)
    def set_entry_point(self, n): self.entry = n
    def compile(self): return _Compiled(self)
