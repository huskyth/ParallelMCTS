from mcts.node import Node


class MCTS:
    def __init__(self, predict):
        self.root = Node(1)
        self.predict = predict

    def _simulate(self, state):
        current_node = self.root
        while True:
            if current_node.is_leaf():
                break

            action, current_node = current_node.select()
            state.do_action(action)

        is_end, winner = state.is_end()
        if is_end is True:
            assert winner is not None
            value = 1 if winner == state.current_play() else -1
        else:
            value, probability = self.predict(state.get_torch_state())
            current_node.expand(probability)
        current_node.update(-value)

    def get_action_probability(self):
        pass
