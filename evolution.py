import copy
import random
from player import Player
import numpy as np


class Evolution:
    ALPHA = 0.3  # used for crossover
    SELECTION_Q = 4
    PARENT_Q = 16
    MUTATION_RATE = 0.3
    MUTATION_MULT = 0.5
    selection_type = 'Q'  # top-k or sus or q_tournament or roulette
    parent_selection = 'Q'  # all or sus or q_tournament or roulette

    def __init__(self):
        self.game_mode = "Neuroevolution"

    def mutate(self, player: Player):
        mutated = self.clone_player(player)
        for layer in range(len(mutated.nn.weights)):
            # generate crossover weights
            for i in range(len(mutated.nn.weights[layer])):
                for j in range(len(mutated.nn.weights[layer][i])):
                    if random.random() < self.MUTATION_RATE:
                        mutated.nn.weights[layer][i][j] += self.MUTATION_MULT * np.random.normal(0, 1)
            for i in range(len(mutated.nn.bias[layer])):
                if random.random() < self.MUTATION_RATE:
                    mutated.nn.bias[layer][i] += self.MUTATION_MULT * np.random.normal(0, 1)

        # if random.random() < self.MUTATION_RATE:
        #     # needs mutation, select random weight and change it randomly
        #     random_layer = random.randint(0, len(mutated.nn.weights) - 1)
        #     random_layer_row = random.randint(0, len(mutated.nn.weights[random_layer]) - 1)
        #     random_layer_column = random.randint(0, len(mutated.nn.weights[random_layer][random_layer_row]) - 1)
        #     mutated.nn.weights[random_layer][random_layer_row][
        #         random_layer_column] += self.MUTATION_MULT * np.random.normal(0, 1)
        # if random.random() < self.MUTATION_RATE:
        #     random_bias_layer = random.randint(0, len(mutated.nn.bias) - 1)
        #     random_bias = random.randint(0, len(mutated.nn.bias[random_bias_layer]) - 1)
        #     mutated.nn.bias[random_bias_layer][random_bias] += self.MUTATION_MULT * np.random.normal(0, 1)
        return mutated

    def generate_children(self, parent1: Player, parent2: Player):
        final_parent1 = self.mutate(parent1)
        final_parent2 = self.mutate(parent2)

        child1 = Player(self.game_mode)
        child2 = Player(self.game_mode)
        # crossover
        for layer in range(len(final_parent1.nn.weights)):
            # generate crossover weights
            for i in range(len(final_parent1.nn.weights[layer])):
                for j in range(len(final_parent1.nn.weights[layer][i])):
                    child1.nn.weights[layer][i][j] = self.ALPHA * final_parent1.nn.weights[layer][i][j] + \
                                                     (1 - self.ALPHA) * final_parent2.nn.weights[layer][i][j]
                    child2.nn.weights[layer][i][j] = self.ALPHA * final_parent2.nn.weights[layer][i][j] + \
                                                     (1 - self.ALPHA) * final_parent1.nn.weights[layer][i][j]
            for i in range(len(final_parent1.nn.bias[layer])):
                child1.nn.bias[layer][i] = self.ALPHA * final_parent1.nn.bias[layer][i] + \
                                           (1 - self.ALPHA) * final_parent2.nn.bias[layer][i]
                child2.nn.bias[layer][i] = self.ALPHA * final_parent2.nn.bias[layer][i] + \
                                           (1 - self.ALPHA) * final_parent1.nn.bias[layer][i]
        return child1, child2

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation

        :param num_players: number of players that we return
        """
        sum = 0
        min_fit = players[0].fitness
        max_fit = players[0].fitness
        for player in players:
            sum += player.fitness
            if player.fitness > max_fit:
                max_fit = player.fitness
            if player.fitness < min_fit:
                min_fit = player.fitness
        print(f'MIN :{min_fit}\tAVG fitness:{sum / len(players)}\tMAX :{max_fit}')
        with open('result.txt', 'a') as history:
            history.write(f'{min_fit} {sum / len(players)} {max_fit}\n')
        # TODO (Implement top-k algorithm here)
        if self.selection_type == 'top-k':
            players.sort(key=lambda x: x.fitness, reverse=True)
            return players[: num_players]
        elif self.selection_type == 'Q':
            selected = list()
            for _ in range(num_players):
                random.shuffle(players)
                tournament = players[:self.SELECTION_Q]
                winner = max(tournament, key=lambda item: item.fitness)
                selected.append(self.clone_player(winner))
            return selected
        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)

        # TODO (Additional: Learning curve)


    def generate_new_population(self, num_players, prev_players: list = None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            if self.parent_selection == 'all':
                new_players = []
                random.shuffle(prev_players)

                for i in range(0, len(prev_players), 2):
                    child1, child2 = self.generate_children(prev_players[i], prev_players[i + 1])
                    new_players.append(child1)
                    new_players.append(child2)
                return new_players

            elif self.parent_selection == 'Q':

                new_players = []
                for _ in range(num_players // 2):
                    random.shuffle(prev_players)
                    tournament = prev_players[:self.SELECTION_Q]
                    parent1 = max(tournament, key=lambda item: item.fitness)
                    random.shuffle(prev_players)
                    tournament = prev_players[:self.SELECTION_Q]
                    parent2 = max(tournament, key=lambda item: item.fitness)
                    child1, child2 = self.generate_children(parent1, parent2)
                    new_players.append(child1)
                    new_players.append(child2)
                return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
