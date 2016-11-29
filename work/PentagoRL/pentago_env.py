# -*- coding: utf-8 -*-
import sys
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class PentagoEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    INFO_KEY = "reason"
    LOOSE_REWARD = -1
    LEGAL_MOVE_REWARD = -.1
    TIE_REWARD = 0
    WIN_REWARD = 1        


    def __init__(self, size, opponent_policy, agent_starts = True):
        # Constants

        # State
        self.size = size
        self.halfsize = self.size / 2
        self.agent_starts = agent_starts
        self.opponent_policy = opponent_policy
        self.player = 1 if self.agent_starts else 2
        self.opponent = 2 if self.player == 1 else 1
        self.episode = 0
               
        # Moves: (size, size) * 4 quadrants * 2 clock rotate directions
        self.moves = [((x,y), q, c) for x in range(self.size) for y in range(self.size) for q in range(4) for c in range(2)]
        
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
                
        if self.agent_starts:
            return self.build_observation()
        else:
            return self.play_opponent_move()        


    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Take player action                
        obs = self.play_action(action, self.player)

        player_won = self.has_player_won(self.player)
        if player_won:
            return obs, PentagoEnv.WIN_REWARD, True, {PentagoEnv.INFO_KEY: "Player {}: won".format(self.player)}
        opponent_won = self.has_player_won(self.opponent) 
        if opponent_won:
            return obs, PentagoEnv.LOOSE_REWARD, True, {PentagoEnv.INFO_KEY: "Player {}: lost".format(self.player)}
        if player_won and opponent_won:
            return obs, PentagoEnv.TIE_REWARD, True, {PentagoEnv.INFO_KEY: "Player {}: tied".format(self.player)}

        # Take opponent action
        obs = self.play_opponent_move()

        player_won = self.has_player_won(self.player)
        if player_won:
            return obs, PentagoEnv.WIN_REWARD, True, {PentagoEnv.INFO_KEY: "Player {}: lost".format(self.opponent)}
        opponent_won = self.has_player_won(self.opponent) 
        if opponent_won:
            return obs, PentagoEnv.LOOSE_REWARD, True, {PentagoEnv.INFO_KEY: "Player {}: won".format(self.opponent)}
        if player_won and opponent_won:
            return obs, PentagoEnv.TIE_REWARD, True, {PentagoEnv.INFO_KEY: "Player {}: tied".format(self.opponent)}

        # Return non-terminal state 
        return obs, PentagoEnv.LEGAL_MOVE_REWARD, False, {PentagoEnv.INFO_KEY: "Player {}: legal move".format(self.player)}

        
    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write("\n")
        np.savetxt(outfile, self.board, fmt="%i")

        if mode != 'human':
            return outfile


    def play_opponent_move(self):
        action = self.opponent_policy.get_action(self.build_observation(), False)
        return self.play_action(action, self.opponent)      
            
        
    def play_action(self, action, player):
        # Lookup move details        
        ((x,y), quadrant, clockwise) = self.moves[action]
        return self.play_move(x, y, quadrant, clockwise, player)


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

        return self.build_observation()


    def is_legal_move(self, x, y):
        return self.board[x, y] == 0


    def has_player_won(self, player):
        self.winseq = np.ones(self.size - 1) * player

        #check lines & columns
        for i in range(self.size):
            if self.is_winseq(self.board[i,:-1]) or self.is_winseq(self.board[i,1:]):
                return True
            if self.is_winseq(self.board[:-1,i]) or self.is_winseq(self.board[1:,i]):
                return True
                
        #check 2 main diagonals
        diag = self.board.diagonal()
        if self.is_winseq(diag[:-1]) or self.is_winseq(diag[1:]):
            return True
        flipped = np.fliplr(self.board)
        diag = flipped.diagonal()
        if self.is_winseq(diag[:-1]) or self.is_winseq(diag[1:]):
            return True
                
        #check the 4 small diagonals
        if self.is_winseq(self.board.diagonal(offset=1)) or self.is_winseq(self.board.diagonal(offset=-1)):
            return True
        if self.is_winseq(flipped.diagonal(offset=1)) or self.is_winseq(flipped.diagonal(offset=-1)):
            return True
            
        return False
        
        
    def is_winseq(self, seq):
        return np.all(np.equal(seq, self.winseq))


    def build_observation(self):
        mask = np.array([self.is_legal_move(x,y) for ((x,y), _, _) in self.moves], dtype=np.uint8)

        key = 0
        for x in range(self.size):
            for y in range(self.size):
                key = 3*key + self.board[x, y] 

        return (self.board, mask, key)       

    
