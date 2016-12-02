# -*- coding: utf-8 -*-
import sys
import numpy as np
import StringIO
import gym
from gym import spaces
from gym.utils import seeding

class PentagoEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "ansi"]
    }

    NB_ROTATE_DIRECTIONS = 2
    NB_QUADRANTS = 4

    INFO_KEY = "Result"
    LOOSE_REWARD = -1
    LEGAL_MOVE_REWARD = -.1
    TIE_REWARD = 0
    WIN_REWARD = 1        


    def __init__(self, size, opponent_policy, agent_starts = True, to_win=None):
        # State
        self.size = size
        self.halfsize = self.size / 2 
        self.win_seq_len = self.size - 1 if to_win is None else to_win
        self.nb_board_squares = size ** 2
        self.agent_starts = agent_starts
        self.opponent_policy = opponent_policy
        self.player = 1 if self.agent_starts else 2
        self.opponent = 2 if self.player == 1 else 1
        self.episode = 0
        self.np_random = None

        # Constants
        self.board_to_key_multiplier = np.power(np.ones(self.nb_board_squares)*3, np.arange(self.nb_board_squares-1, -1, -1))
               
        # Moves: (size, size) * quadrants *  rotate directions
        self.moves = [((x,y), q, c) for x in range(self.size) for y in range(self.size) for q in range(self.NB_QUADRANTS) for c in range(self.NB_ROTATE_DIRECTIONS)]        
                
        # Spaces
        self.nb_moves = len(self.moves)
        self.action_space = spaces.Discrete(self.nb_moves)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=2, shape=(self.size, self.size)),    # Board, 3 "colors" pr square
            spaces.Box(low=0, high=1, shape=(self.nb_moves,)),          # Legal moves mask
            spaces.Discrete(sys.maxsize)))

        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _reset(self):
        self.episode += 1
        self.board = np.zeros((self.size, self.size), dtype=np.uint8)
        self.nb_actions = 0
                
        if self.agent_starts:
            return self.build_observation()
        else:
            return self.play_opponent_move()        


    #@profile
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Take player action                
        obs = self.play_action(action, self.player)

        player_won = self.has_player_won(self.player)
        opponent_won = self.has_player_won(self.opponent)
        if player_won and not opponent_won:
            return obs, PentagoEnv.WIN_REWARD, True, {PentagoEnv.INFO_KEY: "Player {} won".format(self.player)} 
        if opponent_won and not player_won:
            return obs, PentagoEnv.LOOSE_REWARD, True, {PentagoEnv.INFO_KEY: "Player {} lost".format(self.player)}
        if (player_won and opponent_won) or self.nb_actions == self.nb_board_squares:
            return obs, PentagoEnv.TIE_REWARD, True, {PentagoEnv.INFO_KEY: "Player {} tied".format(self.player)}

        # Take opponent action
        obs = self.play_opponent_move()

        player_won = self.has_player_won(self.player)
        opponent_won = self.has_player_won(self.opponent) 
        if player_won and not opponent_won:
            return obs, PentagoEnv.WIN_REWARD, True, {PentagoEnv.INFO_KEY: "Player {} lost".format(self.opponent)}
        if opponent_won and not player_won:
            return obs, PentagoEnv.LOOSE_REWARD, True, {PentagoEnv.INFO_KEY: "Player {} won".format(self.opponent)}
        if (player_won and opponent_won) or self.nb_actions == self.nb_board_squares:
            return obs, PentagoEnv.TIE_REWARD, True, {PentagoEnv.INFO_KEY: "Player {} tied".format(self.opponent)}

        # Return non-terminal state 
        return obs, PentagoEnv.LEGAL_MOVE_REWARD, False, {PentagoEnv.INFO_KEY: "Player {} legal move".format(self.player)}

        
    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO.StringIO() if mode == "ansi" else sys.stdout
        outfile.write(np.array_str(self.board) + "\n")
        return outfile


    def play_opponent_move(self):
        action = self.opponent_policy.get_action(self.build_observation())
        return self.play_action(action, self.opponent)      

            
    #@profile    
    def play_action(self, action, player):
        # Lookup move details        
        ((x,y), quadrant, clockwise) = self.moves[action]
        return self.play_move(x, y, quadrant, clockwise, player)


    #@profile
    def play_move(self, x, y, quadrant, clockwise, player):
        # Check legal move
        assert self.is_legal_move(x,y), "illegal move" 

        # Make move
        self.board[x, y] = player
        x_, y_= 0, 0
        if quadrant == 1 or quadrant == 3:
            x_ += self.halfsize
        if quadrant == 2 or quadrant == 3:
            y_ += self.halfsize
        oldboard = np.copy(self.board)  
        for i in range(self.halfsize):
            for j in range(self.halfsize):
                self.board[x_ + i, y_ + j] = \
                    oldboard[x_ + self.halfsize - 1 - j, y_ + i] if clockwise else \
                    oldboard[x_ + j][y_ + self.halfsize - 1 - i]

        self.nb_actions += 1
        return self.build_observation()


    def is_legal_move(self, x, y):
        return self.board[x, y] == 0


    #@profile
    def has_player_won(self, player):
        self.win_seq = np.ones(self.win_seq_len) * player
        if self.win_seq_len == self.size - 1:
            return self.has_player_won_size_minus_1()
        else:
            return self.has_player_won_full_size()


    #@profile
    def has_player_won_size_minus_1(self):
        #check lines & columns
        for i in range(self.size):
            if self.is_win_seq(self.board[i,:-1]) or self.is_win_seq(self.board[i,1:]):
                return True
            if self.is_win_seq(self.board[:-1,i]) or self.is_win_seq(self.board[1:,i]):
                return True
                
        #check 2 main diagonals
        diag = self.board.diagonal()
        if self.is_win_seq(diag[:-1]) or self.is_win_seq(diag[1:]):
            return True
        flipped = np.fliplr(self.board)
        diag = flipped.diagonal()
        if self.is_win_seq(diag[:-1]) or self.is_win_seq(diag[1:]):
            return True
                
        #check the 4 small diagonals
        if self.is_win_seq(self.board.diagonal(offset=1)) or self.is_win_seq(self.board.diagonal(offset=-1)):
            return True
        if self.is_win_seq(flipped.diagonal(offset=1)) or self.is_win_seq(flipped.diagonal(offset=-1)):
            return True
            
        return False

    
    #@profile
    def has_player_won_full_size(self):
        #check lines & columns
        for i in range(self.size):
            if self.is_win_seq(self.board[i,:]):
                return True
            if self.is_win_seq(self.board[:,i]):
                return True
                
        #check 2 main diagonals
        diag = self.board.diagonal()
        if self.is_win_seq(diag):
            return True
        flipped = np.fliplr(self.board)
        diag = flipped.diagonal()
        if self.is_win_seq(diag):
            return True
                            
        return False        
        
        
    def is_win_seq(self, seq):
        if self.win_seq_len != len(seq): return False
        return np.all(np.equal(seq, self.win_seq))   


    #@profile
    def build_observation(self):
        mask = np.repeat(self.board == 0, self.NB_QUADRANTS * self.NB_ROTATE_DIRECTIONS)

        key = np.sum(self.board.flatten() * self.board_to_key_multiplier) 

        return (self.board, mask, key)       

    
