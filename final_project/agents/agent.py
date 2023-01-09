from game.players import BasePokerPlayer
from agents.agent_utils.card_utils import gen_cards, estimate_hole_card_win_rate
import random as rand

class HonestPlayer(
    BasePokerPlayer
):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]

        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
                nb_simulation=1000,
                nb_player=2,
                hole_card=gen_cards(hole_card),
                community_card=gen_cards(community_card)
            )
        

        pot_amount = round_state['pot']['main']['amount']
        call_amount = valid_actions[1]['amount']
        amount0 = pot_amount * win_rate / (1 - win_rate)
        print("win_rate:", win_rate)
        print(pot_amount, call_amount)

        raise_amount = [valid_actions[2]['amount']['min'], valid_actions[2]['amount']['max']]

        risk_factor = 0.05
        conserve_factor = 1.
        if len(community_card) == 5:
            conserve_factor = 0.1
        
        if len(community_card) == 4:
            conserve_factor = 0.4
        
        if len(community_card) == 3:
            conserve_factor = 0.8
        
        
        round_count = round_state['round_count']
        # print("rc", round_count)
        next_player = round_state['next_player']
        # print("np", next_player)
        token_self = round_state['seats'][next_player]['stack']
        # print("info:", round_count, token_self)
        
        if (20 - round_count) * 10 < token_self - 1000:
            if call_amount == 0:
                action = valid_actions[1] # call
                return action['action'], action['amount']
            
            action = valid_actions[0] # fold
            return action['action'], action['amount']
        
        
        if call_amount > amount0 * (1 + risk_factor):
            print("fold")
            action = valid_actions[0] # fold
            return action['action'], action['amount']

        if win_rate > 0.5 and raise_amount[0] < amount0 * (1 + risk_factor):
            print("raise")
            action = valid_actions[2] # raise
            amount = win_rate * pot_amount / (1 - win_rate)

            amount = rand.randrange(
                int(amount * (1 - conserve_factor)), int(amount * (1 + risk_factor))
            )
            amount = max(amount, raise_amount[0])
            amount = min(amount, raise_amount[1])
            return action['action'], amount
        
        print("call")
        action = valid_actions[1] # call

        return action['action'], action['amount']

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class HonestPlayer2(
    BasePokerPlayer
):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]

        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
                nb_simulation=1000,
                nb_player=2,
                hole_card=gen_cards(hole_card),
                community_card=gen_cards(community_card)
            )
        # print("win_rate:", win_rate)
        call_amount = valid_actions[1]['amount']
        pot_amount = round_state['pot']['main']['amount']
        raise_amount = [valid_actions[2]['amount']['min'], valid_actions[2]['amount']['max']]
        if call_amount > pot_amount * win_rate / (1 - win_rate):
            action = valid_actions[0]
            return action['action'], action['amount']
        elif raise_amount[0] < pot_amount * win_rate / (1 - win_rate) and win_rate > 0.5:
            action = valid_actions[2]
            amount = int(pot_amount * win_rate / (1 - win_rate))
            amount = max(amount, raise_amount[0])
            amount = min(amount, raise_amount[1])
            return action['action'], amount
        else:
            action = valid_actions[1]
            return action['action'], action['amount']

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class HonestPlayer3(
    BasePokerPlayer
):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]

        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
                nb_simulation=1000,
                nb_player=2,
                hole_card=gen_cards(hole_card),
                community_card=gen_cards(community_card)
            )
        

        pot_amount = round_state['pot']['main']['amount']
        call_amount = valid_actions[1]['amount']
        print("win_rate:", win_rate)
        print(pot_amount, call_amount)

        raise_amount = [valid_actions[2]['amount']['min'], valid_actions[2]['amount']['max']]

        risk_factor = 0.05
        conserve_factor = 1
        if len(community_card) == 5:
            conserve_factor = 0.1
        
        if len(community_card) == 4:
            conserve_factor = 0.4
        
        if len(community_card) == 3:
            conserve_factor = 0.8
        conserve_factor = 0
        

        if win_rate * pot_amount <= call_amount * (1 - win_rate) * (1 + risk_factor):
            print("fold")
            action = valid_actions[0] # fold
            return action['action'], action['amount']

        if win_rate > 0.5 and win_rate * pot_amount * (1 + risk_factor) >= raise_amount[0] * (1 - win_rate):
            print("raise")
            action = valid_actions[2] # raise
            amount = win_rate * pot_amount / (1 - win_rate)
            amount = min(raise_amount[0] * 10, amount)
            amount = rand.randrange(
                int(amount * (1 - conserve_factor)), int(amount * (1 + risk_factor))
            )
            amount = max(amount, raise_amount[0])
            amount = min(amount, raise_amount[1])
            return action['action'], amount
        
        print("call")
        action = valid_actions[1] # call

        return action['action'], action['amount']

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return HonestPlayer()

def setup_ai2():
    return HonestPlayer2()

def setup_ai3():
    return HonestPlayer3()
