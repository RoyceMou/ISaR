from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Product, Discrete
import numpy as np


class ThreeCardPokerEnv(Env):

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        return Product(Discrete(52), # Player card 1
                       Discrete(52), # Player card 2
                       Discrete(52), # Player card 3
                       Discrete(52), # Dealer card 1
                       Discrete(52), # Dealer card 2
                       Discrete(52)) # Dealer card 3

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        return Discrete(2)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        # Six cards are drawn from a deck of 52 cards
        deck = np.arange(52);
        np.random.shuffle(deck);
        self._state = deck[-6:]
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        print(action)
        # action should be close to a one-hot vector encoding the actions
        # the player folds if a is 0 and continues if a is 1
        a = 0 if action[0] > action[1] else 1

        next_observation = np.copy(self._state) # the game is over after one action, so the next state is arbitrary

        if a == 0:
            return Step(observation=next_observation, reward=-1, done=True)

        phand = tuple(self.number_to_card(number) for number in self._state[:3])
        dhand = tuple(self.number_to_card(number) for number in self._state[3:])


        is_player_hand_better = self.is_player_hand_better(phand, dhand)
        reward = 4 if is_player_hand_better == 1 else -2 if is_player_hand_better == -1 else 0
        return Step(observation=next_observation, reward=reward, done=True)

    @staticmethod
    def compare(pval, dval):
        if pval > dval:
            return 1
        elif pval < dval:
            return -1
        else:
            return 0

    @staticmethod
    def compare_high(phand, dhand):
        self = ThreeCardPokerEnv
        pvalues = sorted(value for value, suit in phand)
        dvalues = sorted(value for value, suit in dhand)
        res = self.compare(pvalues[-1], dvalues[-1])
        if res == 0:
            res = self.compare(pvalues[-2], dvalues[-2])
            if res == 0:
                return self.compare(pvalues[-3], dvalues[-3])
            else:
                return res
        else:
            return res

    @staticmethod
    def has_straight_flush(hand):
        self = ThreeCardPokerEnv
        return self.has_straight(hand) and self.has_flush(hand)

    @staticmethod
    def has_three_of_a_kind(hand):
        return len(np.unique(value for value, suit in hand)) == 1

    @staticmethod
    def has_straight(hand):
        values = tuple(value for value, suit in hand)
        low = min(values)
        return low + 1 in values and low + 2 in values

    @staticmethod
    def has_flush(hand):
        return len(np.unique(suit for value, suit in hand)) == 1

    @staticmethod
    def has_pair(hand):
        return len(np.unique(value for value, suit in hand)) == 2

    @staticmethod
    def is_player_hand_better(phand, dhand):
        """
        returns 1 if the player's hand is better than the dealer's
        returns 0 if the player's hand is the same as the dealer's
        returns -1 if the player's hand is worse than the dealer's
        """
        self = ThreeCardPokerEnv
        if self.has_straight_flush(phand):
            if self.has_straight_flush(dhand):
                return self.compare_high(phand, dhand)
            else:
                return 1

        elif self.has_three_of_a_kind(phand):
            if self.has_straight_flush(dhand):
                return 0
            elif self.has_three_of_a_kind(dhand):
                return self.compare_high(phand, dhand)
            else:
                return 1

        elif self.has_straight(phand):
            if self.has_straight_flush(dhand) or \
               self.has_three_of_a_kind(dhand):
                return 0
            elif self.has_straight(dhand):
                return self.compare_high(phand, dhand)
            else:
                return 1

        elif self.has_flush(phand):
            if self.has_straight_flush(dhand) or \
               self.has_three_of_a_kind(dhand) or \
               self.has_straight(dhand):
                return 0
            elif self.has_flush(dhand):
                return self.compare_high(phand, dhand)
            else:
                return 1

        elif self.has_pair(phand):
            if self.has_straight_flush(dhand) or \
               self.has_three_of_a_kind(dhand) or \
               self.has_straight(dhand) or \
               self.has_flush(dhand):
                return 0
            elif self.has_pair(dhand):
                pvalues = sorted(value for value, suit in phand)
                dvalues = sorted(value for value, suit in dhand)
                ppair, pval = (pvalues[0], pvalues[2]) if pvalues[0] == pvalues[1] else (pvalues[1], pvalues[0])
                dpair, dval = (dvalues[0], dvalues[2]) if dvalues[0] == dvalues[1] else (dvalues[1], dvalues[0])
                res = self.compare(ppair, dpair)
                if res == 0:
                    return self.compare(pval, dval)
                else:
                    return res
            else:
                return 1

        else:   # player has a high card
            if self.has_straight_flush(dhand) or \
               self.has_three_of_a_kind(dhand) or \
               self.has_straight(dhand) or \
               self.has_flush(dhand) or \
               self.has_pair(dhand):
                return 0
            else:
                return self.compare_high(phand, dhand)

    @staticmethod
    def number_to_card(number):
        value = number % 13
        suit = number / 13
        return (value, suit)

    @staticmethod
    def card_to_string(card):
        value, suit = card
        svalue = ['Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace'][value]
        ssuit = ['Diamonds', 'Clubs', 'Hearts', 'Spades'][suit]
        return '{0} of {1}'.format(svalue, ssuit)


    def render(self):
        pcards = tuple(self.card_to_string(card) for card in self._state[:3])
        dcards = tuple(self.card_to_string(card) for card in self._state[3:])
        print('current state:\nplayer hand: {0}\ndealer hand: {1}'.format(pcards, dcards))
