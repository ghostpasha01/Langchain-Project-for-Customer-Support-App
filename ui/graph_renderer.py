import graphviz
from graphviz import Digraph


class GraphRenderer:
    _graph: Digraph

    def _create(self):
        self._graph = graphviz.Digraph()

        self._graph.node("GreetingNode")

        self._graph.node("AuthenticatedUserNode")
        self._graph.node("CallCustomerNode")

        self._graph.edge("GreetingNode", "AuthenticatedUserNode")
        self._graph.edge("AuthenticatedUserNode", "CallCustomerNode")

        self._styling()

    def _styling(self):
        # Set graph layout attributes
        self._graph.attr(
            size="10,10", ratio="fill", ranksep="2", rankdir="TB", margin="0.2"
        )

        # Node attributes
        self._graph.attr(
            "node",
            shape="ellipse",
            style="filled",
            fillcolor="#D1E8E2",
            fontcolor="#005B5B",
            fontname="Arial",
            fontsize="12",
            width="0",
            height="0",
        )

        # Edge attributes
        self._graph.attr(
            "edge",
            color="#7D968D",
            penwidth="1.5",
            fontname="Arial",
            fontsize="10",
        )

    def _update(self, current_node: str):
        self._create()
        self._graph.node(current_node, fillcolor="#009B77", style="filled")

    def get(self, current_node: str) -> Digraph:
        self._update(current_node)
        return self._graph
