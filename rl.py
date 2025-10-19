'''
!pip install sb3-contrib==2.6.0 stable-baselines3==2.6.0 gymnasium>=0.29.1 numpy torch
!apt-get -y install fonts-noto-cjk
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']
plt.rcParams['axes.unicode_minus'] = False
'''
# ===== 標準函式庫 =====
import os
import json
import copy
import random
import unicodedata
import subprocess
from math import comb
from collections import Counter, defaultdict, deque, OrderedDict
# ===== 科學運算與數據處理 =====
import numpy as np
# ===== 強化學習環境 =====
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
# ===== 繪圖與視覺化 =====
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
def pad_str_fixed_width(s, width):
        length = 0
        for ch in s:
            length += 2 if unicodedata.east_asian_width(ch) in ('F', 'W') else 1
        return s + ' ' * max(0, width - length)
# ---- 全域動作 ID 常數 ----
ID_DISCARD_START = 0
ID_DISCARD_END = 33
ID_PASS = 34
ID_WIN = 35
ID_CHI_L_START = 36
ID_CHI_L_END = 69
ID_CHI_M_START = 70
ID_CHI_M_END = 103
ID_CHI_R_START = 104
ID_CHI_R_END = 137
ID_PONG_START = 138
ID_PONG_END = 171
ID_KONG_START = 172
ID_KONG_END = 205
ID_ADD_KONG_START = 206
ID_ADD_KONG_END = 239
ID_CONC_KONG_START = 240
ID_CONC_KONG_END = 273
class MahjongEnv:
    def __init__(self):
        self.tile_names = [
            '1萬', '2萬', '3萬', '4萬', '5萬', '6萬', '7萬', '8萬', '9萬',
            '1筒', '2筒', '3筒', '4筒', '5筒', '6筒', '7筒', '8筒', '9筒',
            '1索', '2索', '3索', '4索', '5索', '6索', '7索', '8索', '9索',
            '東', '南', '西', '北', '中', '發', '白'
        ]
        self.tile_to_index = {name: idx for idx, name in enumerate(self.tile_names)}
        self.deck = []
        self.players = [{'hand': [], 'discards': [], 'melds': [], 'hidden_melds': []} for _ in range(4)]
        self.current_player = 0
        self.banker = random.randint(0, 3)
        self.banker_wins = 0
        self.game_over = False
        self.last_tile_drawn = None
        self.last_tile_discarded = None
        self.last_discard_player = None
        self.action_performed = False
        self.tile_claimed = False
        self.last_claimed_tile = None
        self.last_discard_efficiency = None
        self.last_action_reward = None
        self.last_action_eff_delta = None
        self.total_reward = 0.0
        self.passed_players_for_current_discard = set()
        self.reward_weights = {
            "efficiency": 1.0,
            "ron": 16.0,
            "tsumo": 18.0,
            "lose_ron": -12.0,
            "lose_tsumo": -6.0,
            "chi": 0.15,
            "pong": 0.3,
            "ming_kong": 0.6,
            "add_kong": 0.6,
            "concealed_kong": 1.0
        }
        self.terminal_focus = 0.0
    def reset(self):
        self.deck = [i for i in range(34) for _ in range(4)]
        random.shuffle(self.deck)
        self.players = [{'hand': [], 'discards': [], 'melds': [], 'hidden_melds': []} for _ in range(4)]
        self.banker = random.randint(0, 3)
        self.current_player = self.banker
        self.game_over = False
        self.last_tile_drawn = None
        self.last_tile_discarded = None
        self.last_discard_player = None
        self.action_performed = False
        self.tile_claimed = False
        self.last_action_reward = None
        self.last_action_eff_delta = None
        self.total_reward = 0.0
        self.passed_players_for_current_discard = set()
        for i in range(4):
            self.players[i]['hand'] = sorted([self.deck.pop() for _ in range(16)])
        return self.get_state()
    def get_state(self):
        return {
            'current_player': self.current_player,
            'banker': self.banker,
            'banker_wins': self.banker_wins,
            'deck_remaining': len(self.deck),
            'players': [
                {
                    'hand': p['hand'],
                    'discards': p['discards'],
                    'melds': p['melds'],
                    'hidden_melds': p['hidden_melds']
                } for p in self.players
            ]
        }
    # =============== 基本動作 ===============
    def draw_tile(self):
        if len(self.deck) <= 16:
            self.game_over = True
            return None, "流局：牌庫剩餘16張"
        tile = self.deck.pop()
        self.players[self.current_player]['hand'].append(tile)
        self.players[self.current_player]['hand'].sort()
        self.last_tile_drawn = tile
        self.action_performed = False
        self.tile_claimed = False
        return tile, None
    def discard_tile(self, tile_name):
        if tile_name not in self.tile_to_index:
            return False, f"無效的牌: {tile_name}"
        tile = self.tile_to_index[tile_name]
        player = self.players[self.current_player]
        if tile not in player['hand']:
            return False, f"無效的出牌: 手牌中無 {tile_name}"
        if self.last_claimed_tile is not None and tile == self.last_claimed_tile:
            return False, f"無效的出牌: 不能打出當回合剛吃/碰/槓進的牌 {tile_name}"
        if self.current_player == 0:
            hand = self.players[0]['hand'][:]
            inst_probs = self._compute_hand_instance_probabilities(hand)
            hand_sorted = sorted(hand)
            mins = [inst_probs[i] for i, t in enumerate(hand_sorted) if t == tile]
            self.last_discard_efficiency = (min(mins) if mins else 0.0)
        else:
            self.last_discard_efficiency = None
        player['hand'].remove(tile)
        self.last_tile_discarded = tile
        self.last_discard_player = self.current_player
        self.tile_claimed = False
        self.action_performed = False
        self.last_claimed_tile = None
        self.passed_players_for_current_discard = set()
        self.last_tile_drawn = None
        return True, None
    def can_pong(self, player_idx, discarded_tile):
        return self.players[player_idx]['hand'].count(discarded_tile) >= 2
    def pong(self, player_idx, tile):
        hand = self.players[player_idx]['hand']
        if hand.count(tile) < 2:
            return False, "手牌中缺少碰牌所需的牌"
        for _ in range(2):
            hand.remove(tile)
        self.players[player_idx]['melds'].append([tile] * 3)
        self.action_performed = True
        self.tile_claimed = True
        self.last_claimed_tile = self.last_tile_discarded
        self.passed_players_for_current_discard = set()
        return True, None
    def can_kong(self, player_idx, discarded_tile):
        return self.players[player_idx]['hand'].count(discarded_tile) >= 3
    def can_concealed_kong(self, player_idx):
        hand = self.players[player_idx]['hand']
        return sorted([tile for tile, count in Counter(hand).items() if count >= 4])
    def can_add_kong(self, player_idx, drawn_tile):
        hand = self.players[player_idx]['hand']
        possible_kongs = []
        for meld in self.players[player_idx]['melds']:
            if len(meld) == 3 and meld[0] == meld[1] == meld[2]:
                tile = meld[0]
                if tile in hand:
                    possible_kongs.append(tile)
        return sorted(list(set(possible_kongs)))
    def kong(self, player_idx, tile):
        hand = self.players[player_idx]['hand']
        if hand.count(tile) < 3:
            return False, "手牌中缺少槓牌所需的牌"
        for _ in range(3):
            hand.remove(tile)
        self.players[player_idx]['melds'].append([tile] * 4)
        self.action_performed = True
        self.tile_claimed = True
        self.passed_players_for_current_discard = set()
        return True, None
    def concealed_kong(self, player_idx, tile):
        hand = self.players[player_idx]['hand']
        if hand.count(tile) < 4:
            return False, "手牌中缺少暗槓所需的牌"
        for _ in range(4):
            hand.remove(tile)
        self.players[player_idx]['hidden_melds'].append([tile] * 4)
        self.action_performed = True
        return True, None
    def add_kong(self, player_idx, tile):
        hand = self.players[player_idx]['hand']
        for meld in self.players[player_idx]['melds']:
            if len(meld) == 3 and meld.count(tile) == 3 and tile in hand:
                hand.remove(tile)
                meld.append(tile)
                self.action_performed = True
                return True, None
        return False, "無可加槓的碰"
    def get_valid_chi_combinations(self, player_idx, discarded_tile):
        hand = self.players[player_idx]['hand']
        tile = discarded_tile
        combinations = {2: None, 3: None, 4: None}
        if tile is not None and tile < 27:
            suit = tile // 9
            rank = tile % 9
            start_idx = suit * 9
            if rank >= 2:
                seq = [start_idx + rank - 2, start_idx + rank - 1, tile]
                if all(t in hand for t in seq[:2]):
                    combinations[2] = [self.tile_names[t] for t in seq]
            if 1 <= rank <= 7:
                seq = [start_idx + rank - 1, tile, start_idx + rank + 1]
                if all(t in hand for t in [seq[0], seq[2]]):
                    combinations[3] = [self.tile_names[t] for t in seq]
            if rank <= 6:
                seq = [tile, start_idx + rank + 1, start_idx + rank + 2]
                if all(t in hand for t in seq[1:]):
                    combinations[4] = [self.tile_names[t] for t in seq]
        return combinations
    def chi(self, player_idx, combination):
        if not combination or len(combination) != 3:
            return False, "無效的吃牌組合"
        try:
            tile_indices = [self.tile_to_index[tile] for tile in combination]
        except KeyError:
            return False, "無效的牌名"
        if self.last_tile_discarded not in tile_indices:
            return False, "吃牌組合必須包含棄牌"
        hand = self.players[player_idx]['hand']
        tiles_to_remove = [t for t in tile_indices if t != self.last_tile_discarded]
        if not all(t in hand for t in tiles_to_remove):
            return False, "手牌中缺少吃牌所需的牌"
        for t in tiles_to_remove:
            hand.remove(t)
        self.players[player_idx]['melds'].append(tile_indices)
        self.action_performed = True
        self.tile_claimed = True
        self.last_claimed_tile = self.last_tile_discarded
        self.passed_players_for_current_discard = set()
        return True, None
    def can_win(self, player_idx, tile, is_self_draw=False):
        player = self.players[player_idx]
        hand = player['hand'].copy()
        if is_self_draw:
            if tile not in hand:
                return False
        else:
            hand.append(tile)
        hand.sort()
        open_melds = player['melds'] + player['hidden_melds']
        meld_count = len(open_melds)
        remaining_tiles = len(hand)
        needed_melds = 5 - meld_count
        if remaining_tiles != (needed_melds * 3 + 2):
            return False
        def is_sequence(tiles):
            if len(tiles) != 3 or tiles[0] >= 27:
                return False
            suit = tiles[0] // 9
            rank = tiles[0] % 9
            return tiles == [suit * 9 + rank, suit * 9 + rank + 1, suit * 9 + rank + 2]
        def is_triplet(tiles):
            return len(tiles) == 3 and tiles[0] == tiles[1] == tiles[2]
        def check_win(hand, melds=None, pairs=None):
            if melds is None:
                melds = []
            if pairs is None:
                pairs = []
            if len(melds) == needed_melds and len(pairs) == 1:
                return True
            if not hand:
                return False
            hand = sorted(hand)
            if len(pairs) == 0 and len(hand) >= 2 and hand[0] == hand[1]:
                new_hand = hand[2:]
                if check_win(new_hand, melds, pairs + [[hand[0], hand[1]]]):
                    return True
            if len(hand) >= 3 and hand[0] == hand[1] == hand[2]:
                new_hand = hand[3:]
                if check_win(new_hand, melds + [[hand[0]] * 3], pairs):
                    return True
            if hand[0] < 27:
                suit = hand[0] // 9
                rank = hand[0] % 9
                seq = [suit * 9 + rank, suit * 9 + rank + 1, suit * 9 + rank + 2]
                if rank <= 6 and all(t in hand for t in seq):
                    new_hand = hand.copy()
                    for t in seq:
                        new_hand.remove(t)
                    if check_win(new_hand, melds + [seq], pairs):
                        return True
            return False
        return check_win(hand)
    # =============== 入口：step（保留舊呼叫介面） ===============
    def step(self, action):
        if self.game_over:
            return self.get_state(), "遊戲結束"
        act_type, param = action
        if act_type == 'draw':
            tile, error = self.draw_tile()
            if error:
                return self.get_state(), error
            return self.get_state(), ""
        elif act_type == 'discard':
            success, error = self.discard_tile(param)
            if not success:
                return self.get_state(), error
            return self.get_state(), ""
        elif act_type == 'chi':
            success, error = self.chi(self.current_player, param)
            if not success:
                return self.get_state(), error
            return self.get_state(), ""
        elif act_type == 'pong':
            success, error = self.pong(self.current_player, self.last_tile_discarded)
            if not success:
                return self.get_state(), error
            return self.get_state(), ""
        elif act_type == 'kong':
            success, error = self.kong(self.current_player, self.last_tile_discarded)
            if not success:
                return self.get_state(), error
            return self.get_state(), ""
        elif act_type == 'concealed_kong':
            success, error = self.concealed_kong(self.current_player, param)
            if not success:
                return self.get_state(), error
            return self.get_state(), ""
        elif act_type == 'add_kong':
            kong_tile = param
            success, error = self.add_kong(self.current_player, kong_tile)
            if not success:
                return self.get_state(), error
            return self.get_state(), ""
        elif act_type == 'win':
            tile = self.last_tile_drawn if param == 'tsumo' else self.last_tile_discarded
            is_self_draw = (param == 'tsumo')
            if self.can_win(self.current_player, tile, is_self_draw):
                self.game_over = True
                if not is_self_draw:
                    self.tile_claimed = True
                return self.get_state(), f"玩家{self.current_player} {'自摸' if is_self_draw else '榮和'}胡牌！"
            return self.get_state(), "無效的胡牌動作"
        elif act_type == 'pass':
            if (self.last_tile_discarded is not None and
                self.last_discard_player is not None and
                not self.tile_claimed):
                self.passed_players_for_current_discard.add(self.current_player)
            return self.get_state(), ""
        return self.get_state(), "無效動作"
    # =============== 產出「動作種類」清單（不含 ID） ===============
    def get_action_menu(self, player_idx, tsumo=False):
        actions = []
        tile = self.last_tile_drawn if tsumo else self.last_tile_discarded
        actions.append((0, "過"))
        if tile is None:
            return actions
        if self.can_win(player_idx, tile, is_self_draw=tsumo):
            actions.append((1, "胡"))
        if not tsumo:
            if (self.last_discard_player is not None) and player_idx == (self.last_discard_player + 1) % 4:
                chi_combinations = self.get_valid_chi_combinations(player_idx, self.last_tile_discarded)
                for action_id, combo in chi_combinations.items():
                    if combo:
                         label = f"吃({'左' if action_id == 2 else '中' if action_id == 3 else '右'}) [{' '.join(combo)}]"
                         actions.append((action_id, label))

            if self.last_tile_discarded is not None and self.can_pong(player_idx, self.last_tile_discarded):
                 actions.append((5, "碰"))

            if self.last_tile_discarded is not None and self.can_kong(player_idx, self.last_tile_discarded):
                 actions.append((6, "明槓"))
        else:
            add_kong_tiles = self.can_add_kong(player_idx, self.last_tile_drawn)
            for i, tile in enumerate(add_kong_tiles[:4], 7):
                actions.append((i, f"加槓 {self.tile_names[tile]}"))

            concealed_kong_tiles = self.can_concealed_kong(player_idx)
            for i, tile in enumerate(concealed_kong_tiles[:4], 11):
                actions.append((i, f"暗槓 {self.tile_names[tile]}"))
        return actions
    # =============== 回傳「動作 ID + 文字」選單（不包含出牌） ===============
    def get_action_id_menu(self, player_idx, tsumo=False):
        id_menu = []
        id_menu.append((ID_PASS, "過"))
        target_tile = self.last_tile_drawn if tsumo else self.last_tile_discarded
        if (not tsumo and
            self.last_tile_discarded is not None and
            self.last_discard_player is not None and
            not self.tile_claimed and
            player_idx in getattr(self, "passed_players_for_current_discard", set())):
            return id_menu
        if target_tile is not None and self.can_win(player_idx, target_tile, is_self_draw=tsumo):
            id_menu.append((ID_WIN, "胡"))
        if not tsumo:
            if self.last_tile_discarded is not None and self.last_discard_player is not None and not self.tile_claimed:
                d = self.last_tile_discarded
                if player_idx == (self.last_discard_player + 1) % 4:
                    chi_combinations = self.get_valid_chi_combinations(player_idx, d)
                    if chi_combinations[2]:
                        id_menu.append((ID_CHI_L_START + d, f"吃(左) [{' '.join(chi_combinations[2])}]"))
                    if chi_combinations[3]:
                        id_menu.append((ID_CHI_M_START + d, f"吃(中) [{' '.join(chi_combinations[3])}]"))
                    if chi_combinations[4]:
                        id_menu.append((ID_CHI_R_START + d, f"吃(右) [{' '.join(chi_combinations[4])}]"))
                if self.can_pong(player_idx, d):
                    id_menu.append((ID_PONG_START + d, "碰"))
                if self.can_kong(player_idx, d):
                    id_menu.append((ID_KONG_START + d, "明槓"))
        else:
            add_kong_tiles = self.can_add_kong(player_idx, self.last_tile_drawn)
            for t in add_kong_tiles:
                id_menu.append((ID_ADD_KONG_START + t, f"加槓 {self.tile_names[t]}"))
            concealed_kong_tiles = self.can_concealed_kong(player_idx)
            for t in concealed_kong_tiles:
                id_menu.append((ID_CONC_KONG_START + t, f"暗槓 {self.tile_names[t]}"))
        return id_menu
    def get_action_id_menu_with_eff(self, player_idx: int, tsumo: bool = False):
        base_menu = self.get_action_id_menu(player_idx, tsumo=tsumo)
        base_eff = self._eff_pass_current_context(player_idx, tsumo=tsumo)
        id_menu = []
        rewards = {}
        d = self.last_tile_discarded
        def parse_chi_combo(label: str):
            if '[' in label and ']' in label:
                inside = label[label.index('[')+1:label.index(']')].strip()
                try:
                    return [self.tile_to_index[n] for n in inside.split()]
                except KeyError:
                    return None
            return None
        for action_id, label in base_menu:
            eff = None
            if action_id == ID_PASS:
                eff = base_eff
            elif action_id == ID_WIN:
                eff = 600.00
            elif ID_CHI_L_START <= action_id <= ID_CHI_L_END and d is not None and (action_id - ID_CHI_L_START) == d:
                tiles = parse_chi_combo(label);  eff = self._eff_after_chi(player_idx, tiles) if tiles else None
            elif ID_CHI_M_START <= action_id <= ID_CHI_M_END and d is not None and (action_id - ID_CHI_M_START) == d:
                tiles = parse_chi_combo(label);  eff = self._eff_after_chi(player_idx, tiles) if tiles else None
            elif ID_CHI_R_START <= action_id <= ID_CHI_R_END and d is not None and (action_id - ID_CHI_R_START) == d:
                tiles = parse_chi_combo(label);  eff = self._eff_after_chi(player_idx, tiles) if tiles else None
            elif ID_PONG_START <= action_id <= ID_PONG_END and d is not None and (action_id - ID_PONG_START) == d:
                eff = self._eff_after_pong(player_idx, d)
            elif ID_KONG_START <= action_id <= ID_KONG_END and d is not None and (action_id - ID_KONG_START) == d:
                eff = self._eff_after_ming_kong(player_idx, d)
            elif tsumo and ID_ADD_KONG_START <= action_id <= ID_ADD_KONG_END:
                eff = self._eff_after_add_kong(player_idx, action_id - ID_ADD_KONG_START)
            elif tsumo and ID_CONC_KONG_START <= action_id <= ID_CONC_KONG_END:
                eff = self._eff_after_concealed_kong(player_idx, action_id - ID_CONC_KONG_START)
            if eff is None:
                id_menu.append((action_id, label))
            else:
                rewards[action_id] = round(eff - base_eff, 4)
                id_menu.append((action_id, f"{label} ({eff:.2f}%)"))
        self.last_action_rewards = rewards
        self.last_action_base_eff = base_eff
        return id_menu, base_eff, rewards
    # =============== 用 ID 解析成 step 的 action tuple ===============
    def decode_action_id(self, action_id, tsumo=False):
        if ID_DISCARD_START <= action_id <= ID_DISCARD_END:
            tile_idx = action_id - ID_DISCARD_START
            if 0 <= tile_idx < 34:
                return ('discard', self.tile_names[tile_idx])
            return (None, "無效的出牌ID")
        if action_id == ID_PASS:
            return ('pass', None)
        if action_id == ID_WIN:
            return ('win', 'tsumo' if tsumo else 'ron')
        if not tsumo:
            d = self.last_tile_discarded
            if d is None:
                return (None, "目前沒有可反應的棄牌")
            if ID_CHI_L_START <= action_id <= ID_CHI_L_END:
                tile_idx = action_id - ID_CHI_L_START
                if tile_idx != d:
                    return (None, "吃左ID與當前棄牌不符")
                combos = self.get_valid_chi_combinations(self.current_player, d)
                if combos[2]:
                    return ('chi', combos[2])
                return (None, "沒有可用的吃左組合")
            if ID_CHI_M_START <= action_id <= ID_CHI_M_END:
                tile_idx = action_id - ID_CHI_M_START
                if tile_idx != d:
                    return (None, "吃中ID與當前棄牌不符")
                combos = self.get_valid_chi_combinations(self.current_player, d)
                if combos[3]:
                    return ('chi', combos[3])
                return (None, "沒有可用的吃中組合")
            if ID_CHI_R_START <= action_id <= ID_CHI_R_END:
                tile_idx = action_id - ID_CHI_R_START
                if tile_idx != d:
                    return (None, "吃右ID與當前棄牌不符")
                combos = self.get_valid_chi_combinations(self.current_player, d)
                if combos[4]:
                    return ('chi', combos[4])
                return (None, "沒有可用的吃右組合")
            if ID_PONG_START <= action_id <= ID_PONG_END:
                tile_idx = action_id - ID_PONG_START
                if tile_idx != d:
                    return (None, "碰ID與當前棄牌不符")
                return ('pong', None)
            if ID_KONG_START <= action_id <= ID_KONG_END:
                tile_idx = action_id - ID_KONG_START
                if tile_idx != d:
                    return (None, "明槓ID與當前棄牌不符")
                return ('kong', None)
        if tsumo:
            if ID_ADD_KONG_START <= action_id <= ID_ADD_KONG_END:
                tile_idx = action_id - ID_ADD_KONG_START
                return ('add_kong', tile_idx)
            if ID_CONC_KONG_START <= action_id <= ID_CONC_KONG_END:
                tile_idx = action_id - ID_CONC_KONG_START
                return ('concealed_kong', tile_idx)
        return (None, "無效的動作ID")
    # =============== 輸出 0~273 的動作ID遮罩（1=可執行，0=不可） ===============
    def get_action_id_mask(self, player_idx, tsumo=False):
        idmask = [0] * 274
        effective_tsumo = bool(tsumo)
        if player_idx == self.current_player and self.last_tile_drawn is not None:
            effective_tsumo = True

        if player_idx == self.current_player:
            action_menu = self.get_action_menu(player_idx, tsumo=effective_tsumo)
        else:
            action_menu = self.get_action_menu(player_idx, tsumo=False)

        self_phase_actions_present = any(aid == 1 or 7 <= aid <= 14 for aid, _ in action_menu)
        has_live_discard = (self.last_tile_discarded is not None and
                   self.last_discard_player is not None and
                   not self.tile_claimed)
        reactive_actions_present   = any(aid == 1 or 2 <= aid <= 6  for aid, _ in action_menu)
        is_reaction_context = (
            (not effective_tsumo) and
            has_live_discard and
            reactive_actions_present and
            (self.last_tile_drawn is None)
        )
        is_action_phase = (self_phase_actions_present if effective_tsumo else is_reaction_context)
        if is_action_phase:
            idmask[ID_PASS] = 1
            target_tile = self.last_tile_drawn if effective_tsumo else self.last_tile_discarded
            if target_tile is not None and self.can_win(player_idx, target_tile, is_self_draw=effective_tsumo):
                idmask[ID_WIN] = 1
            if not effective_tsumo:
                d = self.last_tile_discarded
                if d is not None:
                    if player_idx == (self.last_discard_player + 1) % 4:
                        chi_combos = self.get_valid_chi_combinations(player_idx, d)
                        if chi_combos[2]: idmask[ID_CHI_L_START + d]  = 1
                        if chi_combos[3]: idmask[ID_CHI_M_START + d]  = 1
                        if chi_combos[4]: idmask[ID_CHI_R_START + d]  = 1
                    if self.can_pong(player_idx, d):
                        idmask[ID_PONG_START + d] = 1
                    if self.can_kong(player_idx, d):
                        idmask[ID_KONG_START + d] = 1
            else:
                for t in self.can_add_kong(player_idx, self.last_tile_drawn):
                    idmask[ID_ADD_KONG_START + t] = 1
                for t in self.can_concealed_kong(player_idx):
                    idmask[ID_CONC_KONG_START + t] = 1
        else:
            if player_idx == self.current_player:
                from collections import Counter
                counts = Counter(self.players[player_idx]['hand'])
                forbid = self.last_claimed_tile
                for t in range(34):
                    if counts[t] > 0 and (forbid is None or t != forbid):
                        idmask[ID_DISCARD_START + t] = 1
        def slice_range(a, b):
            return idmask[a:b+1]
        table = OrderedDict()
        table['出牌'] = slice_range(0, 33)
        table['過']   = [idmask[34]]
        table['胡']   = [idmask[35]]
        table['吃(左)'] = slice_range(36, 69)
        table['吃(中)'] = slice_range(70, 103)
        table['吃(右)'] = slice_range(104, 137)
        table['碰']     = slice_range(138, 171)
        table['明槓']   = slice_range(172, 205)
        table['加槓']   = slice_range(206, 239)
        table['暗槓']   = slice_range(240, 273)
        return table
    # ======== 手牌效率：工具 ========
    def _hand_to_count(self, hand):
        cnt = [0]*34
        for t in hand:
            cnt[t] += 1
        return cnt
    def _get_used_tiles(self, count, open_tiles):
        used = count[:]
        for t in open_tiles:
            used[t] += 1
        return used
    def _find_max_meld_combinations(self, count):
        results = []
        max_melds = [0]
        def dfs(cur, path, meld_count=0, pos=0):
            while pos < 34 and cur[pos] == 0:
                pos += 1
            if pos >= 34:
                if meld_count > max_melds[0]:
                    max_melds[0] = meld_count
                    results.clear()
                    results.append(copy.deepcopy(path))
                elif meld_count == max_melds[0]:
                    results.append(copy.deepcopy(path))
                return
            if pos <= 26:
                suit = pos // 9
                r = pos % 9
                if 1 <= r <= 3:
                    a = suit * 9 + r
                    b = a + 1
                    c = a + 2
                    d = a + 3
                    e = a + 4
                    if cur[a] >= 1 and cur[b] >= 1 and cur[c] >= 1 and cur[d] >= 1 and cur[e] >= 1:
                        cur[a] -= 1; cur[b] -= 1; cur[c] -= 1; cur[d] -= 1; cur[e] -= 1
                        path.append([a, b, c, d, e])
                        dfs(cur, path, meld_count + 1, pos)
                        path.pop()
                        cur[a] += 1; cur[b] += 1; cur[c] += 1; cur[d] += 1; cur[e] += 1
            if cur[pos] >= 3:
                cur[pos] -= 3
                path.append([pos, pos, pos])
                dfs(cur, path, meld_count + 1, pos)
                path.pop()
                cur[pos] += 3
            if pos <= 26 and pos % 9 <= 6:
                if cur[pos] >= 1 and cur[pos + 1] >= 1 and cur[pos + 2] >= 1:
                    cur[pos] -= 1; cur[pos + 1] -= 1; cur[pos + 2] -= 1
                    path.append([pos, pos + 1, pos + 2])
                    dfs(cur, path, meld_count + 1, pos)
                    path.pop()
                    cur[pos] += 1; cur[pos + 1] += 1; cur[pos + 2] += 1
            dfs(cur, path, meld_count, pos + 1)
        dfs(count[:], [])
        return max_melds[0], results
    def _find_max_efficiency_combinations(self, hand_count, used_tiles):
        results = []
        max_eff_count = [0]
        def dfs(count, path, marked, pos=0, eff_count=0):
            while pos < 34 and count[pos] == 0:
                pos += 1
            if pos >= 34:
                if eff_count > max_eff_count[0]:
                    max_eff_count[0] = eff_count
                    results.clear()
                    results.append((copy.deepcopy(path), marked[:]))
                elif eff_count == max_eff_count[0]:
                    results.append((copy.deepcopy(path), marked[:]))
                return
            if count[pos] >= 2:
                count[pos] -= 2
                marked[pos] += 2
                path.append(("pair", [pos, pos], [pos], max(0, 4 - used_tiles[pos])))
                dfs(count, path, marked, pos, eff_count + 1)
                path.pop()
                marked[pos] -= 2
                count[pos] += 2
            if pos <= 26 and pos % 9 <= 7 and count[pos] >= 1 and count[pos+1] >= 1:
                if pos % 9 == 0:
                    wait = [pos + 2]
                elif pos % 9 == 7:
                    wait = [pos - 1]
                else:
                    wait = [pos - 1, pos + 2]
                eff = sum(max(0, 4 - used_tiles[w]) / 2 for w in wait if 0 <= w < 34)
                count[pos] -= 1
                count[pos+1] -= 1
                marked[pos] += 1
                marked[pos+1] += 1
                path.append(("sequence_candidate", [pos, pos+1], wait, eff))
                dfs(count, path, marked, pos, eff_count + 1)
                path.pop()
                marked[pos] -= 1
                marked[pos+1] -= 1
                count[pos] += 1
                count[pos+1] += 1
            if pos <= 26 and pos % 9 <= 6 and count[pos] >= 1 and count[pos+2] >= 1:
                mid = pos + 1
                eff = max(0, 4 - used_tiles[mid]) / 2
                count[pos] -= 1
                count[pos+2] -= 1
                marked[pos] += 1
                marked[pos+2] += 1
                path.append(("middle_wait", [pos, pos+2], [mid], eff))
                dfs(count, path, marked, pos, eff_count + 1)
                path.pop()
                marked[pos] -= 1
                marked[pos+2] -= 1
                count[pos] += 1
                count[pos+2] += 1
            dfs(count, path, marked, pos + 1, eff_count)
        dfs(hand_count[:], [], [0] * 34)
        return results, max_eff_count[0]
    def _comb(self, n, k):
        if k < 0 or k > n: return 0
        if k == 0 or k == n: return 1
        k = min(k, n-k)
        num = 1
        den = 1
        for i in range(1, k+1):
            num *= n - (k - i)
            den *= i
        return num // den
    def _get_efficiency_for_tile(self, tile: int, counts: list[int], total_remaining: int) -> float:
        total = 0
        all_combos = comb(total_remaining, 2)
        if all_combos == 0:
            return 0.0
        if tile <= 26:
            if tile % 9 >= 2:
                total += min(counts[tile - 2], counts[tile - 1])
            if 1 <= tile % 9 <= 7:
                total += min(counts[tile - 1], counts[tile + 1])
            if tile % 9 <= 6:
                total += min(counts[tile + 1], counts[tile + 2])
        if counts[tile] >= 2:
            total += comb(counts[tile], 2)
        return (total / all_combos) * 100.0
    def _efficiency_to_prob(self, eff, base):
        if base == 0:
            return 0.0
        return round(eff / base * 100.0, 2)
    def _evaluate_combination(self, melds, count, used_tiles):
        temp_count = count[:]
        for meld in melds:
            for t in meld:
                temp_count[t] -= 1
        candidates, _ = self._find_max_efficiency_combinations(temp_count, used_tiles)
        def five_run_bonus(m):
            if len(m) != 5:
                return 0
            a, b, c, d, e = m
            if not (0 <= a <= 26 and a//9 == e//9 and a+1==b and b+1==c and c+1==d and d+1==e):
                return 0
            waits = []
            if a % 9 >= 1:
                waits.append(a - 1)
            waits.append(c)
            if e % 9 <= 7:
                waits.append(e + 1)
            return sum(max(0, 4 - used_tiles[w]) for w in waits if 0 <= w < 34) / 2
        five_eff = sum(five_run_bonus(m) for m in melds)
        if not candidates:
            return (len(melds), five_eff, 0), None
        total_remaining = sum(max(0, 4 - used_tiles[i]) for i in range(34))
        best_eff, best_lonely_eff = -1, -1
        best_combo_data = None
        for combo, _ in candidates:
            total_eff = five_eff
            temp_c = temp_count[:]
            for _, tiles, _, eff in combo:
                for t in tiles:
                    temp_c[t] -= 1
                total_eff += eff
            remaining_tiles = [max(0, 4 - used_tiles[i]) for i in range(34)]
            lonely_eff = 0.0
            for t in range(34):
                for _ in range(temp_c[t]):
                    lonely_eff += self._get_efficiency_for_tile(t, remaining_tiles, total_remaining)
            if (total_eff > best_eff) or (total_eff == best_eff and lonely_eff > best_lonely_eff):
                best_eff = total_eff
                best_lonely_eff = lonely_eff
                best_combo_data = combo
        return (len(melds), best_eff, best_lonely_eff), best_combo_data
    def _find_best_combination(self, meld_combinations, count, used_tiles):
        scores_and_combinations = []
        for melds in meld_combinations:
            (meld_count, eff, lonely_eff), combo_data = self._evaluate_combination(melds, count, used_tiles)
            five_count = sum(1 for m in melds if len(m) == 5)
            scores_and_combinations.append(((meld_count, five_count, eff, lonely_eff), melds, combo_data))
        scores_and_combinations.sort(key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3]), reverse=True)
        best_score = scores_and_combinations[0][0]
        best_candidates = [(melds, combo_data) for score, melds, combo_data in scores_and_combinations if score == best_score]
        return best_score, random.choice(best_candidates)
    def _gather_open_tiles_public(self):
        opens = []
        for p in self.players:
            opens.extend(p['discards'])
            for meld in p['melds']:
                opens.extend(meld)
        return opens
    def _compute_hand_tile_probabilities(self, hand):
        count = self._hand_to_count(hand)
        _, max_meld_combos = self._find_max_meld_combinations(count)
        open_tiles = self._gather_open_tiles_public()
        used_tiles = self._get_used_tiles(count, open_tiles)
        used_tile_num = sum(count) + len(open_tiles)
        remaining_tiles_total = max(0, 136 - used_tile_num)
        probs = [0.0] * 34
        _best_score, (best_melds, best_combo_data) = self._find_best_combination(max_meld_combos, count, used_tiles)
        def five_run_prob(meld):
            if len(meld) != 5:
                return None
            a, b, c, d, e = meld
            if not (0 <= a <= 26 and a//9 == e//9 and a+1==b and b+1==c and c+1==d and d+1==e):
                return None
            waits = []
            if a % 9 >= 1: waits.append(a - 1)
            waits.append(c)
            if e % 9 <= 7: waits.append(e + 1)
            cand = sum(max(0, 4 - used_tiles[w]) for w in waits if 0 <= w < 34) / 2
            return round(100.0 + self._efficiency_to_prob(cand, base=remaining_tiles_total), 2)
        for meld in best_melds:
            p5 = five_run_prob(meld)
            if p5 is not None:
                for t in meld:
                    probs[t] = max(probs[t], p5)
            else:
                for t in meld:
                    probs[t] = max(probs[t], 100.0)
        promoted_pair_tile = None
        if best_combo_data:
            pair_items = [(tag, tiles, waits, eff) for (tag, tiles, waits, eff) in best_combo_data if tag == "pair"]
            if len(pair_items) == 1:
                promoted_pair_tile = pair_items[0][1][0]
                probs[promoted_pair_tile] = 100.0
            for tag, tiles, _waits, candidate_eff in best_combo_data:
                if tag == "pair" and promoted_pair_tile is not None and tiles[0] == promoted_pair_tile:
                    continue
                p = self._efficiency_to_prob(candidate_eff, base=remaining_tiles_total)
                for t in tiles:
                    probs[t] = max(probs[t], p)
        temp_count = count[:]
        for meld in best_melds:
            for t in meld:
                temp_count[t] -= 1
        if best_combo_data:
            for tag, tiles, _waits, _eff in best_combo_data:
                if tag == "pair" and promoted_pair_tile is not None and tiles[0] == promoted_pair_tile:
                    temp_count[promoted_pair_tile] -= 2
                else:
                    for t in tiles:
                        temp_count[t] -= 1
        remaining_tiles_list = [max(0, 4 - used_tiles[i]) for i in range(34)]
        total_remaining_for_single = sum(remaining_tiles_list)
        for t in range(34):
            for _ in range(temp_count[t]):
                single_prob = self._get_efficiency_for_tile(t, remaining_tiles_list, total_remaining_for_single)
                probs[t] = max(probs[t], round(single_prob, 2))
        return probs
    def _compute_hand_instance_probabilities(self, hand):
        hand_sorted = sorted(hand)
        eff_map = self.compute_tile_efficiencies(hand_sorted)
        instance_probs = [eff_map[t] for t in hand_sorted]
        return instance_probs
    def compute_tile_efficiencies(self, hand):
        count = self._hand_to_count(hand)
        _, meld_combos = self._find_max_meld_combinations(count)
        open_tiles = self._gather_open_tiles_public()
        used_tiles = self._get_used_tiles(count, open_tiles)
        used_tile_num = sum(count) + len(open_tiles)
        total_remaining = max(0, 136 - used_tile_num)
        tile_eff_list = {t: [] for t in hand}
        def assign_eff(tile_id, eff):
            if tile_id in tile_eff_list:
                tile_eff_list[tile_id].append(eff)
        for melds in meld_combos:
            temp_count = count[:]
            for meld in melds:
                if len(meld) == 5:
                    waits = []
                    if meld[0] % 9 >= 1: waits.append(meld[0] - 1)
                    waits.append(meld[2])
                    if meld[4] % 9 <= 7: waits.append(meld[4] + 1)
                    eff = sum(max(0, 4 - used_tiles[w]) for w in waits) / 2
                    bonus = (eff / total_remaining * 100 / 2) if total_remaining > 0 else 0
                    for t in meld:
                        temp_count[t] -= 1
                        assign_eff(t, 100 + bonus)
                elif len(meld) == 3:
                    for t in meld:
                        temp_count[t] -= 1
                        assign_eff(t, 100.0)
            candidate_combos, _ = self._find_max_efficiency_combinations(temp_count, used_tiles)
            if candidate_combos:
                for combo, _ in candidate_combos:
                    for _, tiles, _, eff in combo:
                        prob = self._efficiency_to_prob(eff, total_remaining)
                        for t in tiles:
                            assign_eff(t, prob)
                temp_cand = temp_count[:]
                for combo, _ in candidate_combos:
                    for _, tiles, _, _ in combo:
                        for t in tiles:
                            temp_cand[t] -= 1
                remaining_list = [max(0, 4 - used_tiles[i]) for i in range(34)]
                for t in range(34):
                    for _ in range(temp_cand[t]):
                        prob = self._get_efficiency_for_tile(t, remaining_list, total_remaining)
                        assign_eff(t, prob)
        result = {t: (min(effs) if effs else 100.0) for t, effs in tile_eff_list.items()}
        return result
    # ======== 整體手牌效率（面子*100 + 候選%） ========
    def _total_efficiency_percent(self, hand_count: list[int], used_tiles: list[int], extra_melds: int = 0) -> float:
        _, meld_combos = self._find_max_meld_combinations(hand_count)
        remaining_total = max(0, 136 - (sum(hand_count) + len(self._gather_open_tiles_public())))
        best_score, _pack = self._find_best_combination(meld_combos, hand_count, used_tiles)
        meld_cnt, _five_cnt, eff_units, _lonely = best_score
        meld_cnt += max(0, int(extra_melds))
        return meld_cnt * 100.0 + self._efficiency_to_prob(eff_units, base=remaining_total)
    def _hand_count_from_list(self, tiles: list[int]) -> list[int]:
        cnt = [0]*34
        for t in tiles:
            cnt[t] += 1
        return cnt
    def _public_used_tiles_with_extra(self, base_count: list[int], extra_used: list[int] | None = None) -> list[int]:
        used = base_count[:]
        opens = self._gather_open_tiles_public()
        for t in opens:
            used[t] += 1
        if extra_used:
            for t in extra_used:
                used[t] += 1
        for i in range(34):
            if used[i] > 4:
                used[i] = 4
        return used
    # ======== PASS (過) 的基準效率 ========
    def _existing_meld_cnt(self, player_idx: int) -> int:
        return len(self.players[player_idx]['melds']) + len(self.players[player_idx]['hidden_melds'])
    def _eff_pass_current_context(self, player_idx: int, tsumo: bool) -> float:
        hand = self.players[player_idx]['hand'][:]
        count = self._hand_count_from_list(hand)
        extra_melds = self._existing_meld_cnt(player_idx)
        if tsumo:
            used_tiles = self._public_used_tiles_with_extra(count, extra_used=None)
        else:
            extra = [self.last_tile_discarded] if self.last_tile_discarded is not None else None
            used_tiles = self._public_used_tiles_with_extra(count, extra_used=extra)
        return round(self._total_efficiency_percent(count, used_tiles, extra_melds=extra_melds), 2)
    # ======== 模擬吃／碰／明槓（反應棄牌）後的效率 ========
    def _eff_after_chi(self, player_idx: int, chi_tiles: list[int]) -> float:
        assert self.last_tile_discarded in chi_tiles
        hand = self.players[player_idx]['hand'][:]
        for t in chi_tiles:
            if t != self.last_tile_discarded:
                hand.remove(t)
        count = self._hand_count_from_list(hand)
        extra_used = chi_tiles[:]
        used_tiles = self._public_used_tiles_with_extra(count, extra_used=extra_used)
        extra_melds = self._existing_meld_cnt(player_idx) + 1
        return round(self._total_efficiency_percent(count, used_tiles, extra_melds=extra_melds), 2)
    def _eff_after_pong(self, player_idx: int, tile: int) -> float:
        hand = self.players[player_idx]['hand'][:]
        hand.remove(tile); hand.remove(tile)
        count = self._hand_count_from_list(hand)
        extra_used = [tile, tile, tile]
        used_tiles = self._public_used_tiles_with_extra(count, extra_used=extra_used)
        extra_melds = self._existing_meld_cnt(player_idx) + 1
        return round(self._total_efficiency_percent(count, used_tiles, extra_melds=extra_melds), 2)
    def _eff_after_ming_kong(self, player_idx: int, tile: int) -> float:
        hand = self.players[player_idx]['hand'][:]
        for _ in range(3):
            hand.remove(tile)
        count = self._hand_count_from_list(hand)
        extra_used = [tile, tile, tile, tile]
        used_tiles = self._public_used_tiles_with_extra(count, extra_used=extra_used)
        extra_melds = self._existing_meld_cnt(player_idx) + 1
        base_eff = self._total_efficiency_percent(count, used_tiles, extra_melds=extra_melds)
        remaining = [max(0, 4 - used_tiles[i]) for i in range(34)]
        sea = sum(remaining)
        if sea > 0:
            weighted_delta = 0.0
            original_eff = self._eff_pass_current_context(player_idx, tsumo=False)
            for i in range(34):
                if remaining[i] == 0:
                    continue
                new_count = count[:]
                new_count[i] += 1
                new_eff = self._total_efficiency_percent(new_count, used_tiles, extra_melds=extra_melds)
                weighted_delta += (new_eff - original_eff) * (remaining[i] / sea)
            return round(base_eff + weighted_delta, 2)
        return round(base_eff, 2)
    def _eff_after_add_kong(self, player_idx: int, tile: int) -> float:
        hand = self.players[player_idx]['hand'][:]
        hand.remove(tile)
        count = self._hand_count_from_list(hand)
        used_tiles = self._public_used_tiles_with_extra(count, extra_used=[tile])
        extra_melds = self._existing_meld_cnt(player_idx)
        base_eff = self._total_efficiency_percent(count, used_tiles, extra_melds=extra_melds)
        original_eff = self._eff_pass_current_context(player_idx, tsumo=True)
        remaining = [max(0, 4 - used_tiles[i]) for i in range(34)]
        sea = sum(remaining)
        if sea > 0:
            weighted_delta = 0.0
            for i in range(34):
                if remaining[i] == 0:
                    continue
                new_count = count[:]
                new_count[i] += 1
                new_eff = self._total_efficiency_percent(new_count, used_tiles, extra_melds=extra_melds)
                weighted_delta += (new_eff - original_eff) * (remaining[i] / sea)
            return round(base_eff + weighted_delta, 2)
        return round(base_eff, 2)
    def _eff_after_concealed_kong(self, player_idx: int, tile: int) -> float:
        hand = self.players[player_idx]['hand'][:]
        for _ in range(4):
            hand.remove(tile)
        count = self._hand_count_from_list(hand)
        used_tiles = self._public_used_tiles_with_extra(count, extra_used=[tile, tile, tile, tile])
        extra_melds = self._existing_meld_cnt(player_idx) + 1
        base_eff = self._total_efficiency_percent(count, used_tiles, extra_melds=extra_melds)
        original_eff = self._eff_pass_current_context(player_idx, tsumo=True)
        remaining = [max(0, 4 - used_tiles[i]) for i in range(34)]
        sea = sum(remaining)
        if sea > 0:
            weighted_delta = 0.0
            for i in range(34):
                if remaining[i] == 0:
                    continue
                new_count = count[:]
                new_count[i] += 1
                new_eff = self._total_efficiency_percent(new_count, used_tiles, extra_melds=extra_melds)
                weighted_delta += (new_eff - original_eff) * (remaining[i] / sea)
            return round(base_eff + weighted_delta, 2)
        return round(base_eff, 2)
    # =============== UI ===============
    def render(self, show_input_prompt=False, after_discard=False, input_tile=None, action_menu=None, tsumo=False, in_tile=None):
        display_deck = max(0, len(self.deck) - 16)
        player_str = "您" if self.current_player == 0 else f"玩家{self.current_player}"
        banker_str = f" (莊×{self.banker_wins})" if self.current_player == self.banker else ""
        print(f"牌庫剩餘: {display_deck} 張")
        print(f"當前玩家: {player_str}{banker_str}")
        print(f"摸牌: {self.tile_names[self.last_tile_drawn] if self.last_tile_drawn is not None else ''}")
        if self.current_player == 0:
            hand_sorted = sorted(self.players[0]['hand'])
            inst_probs = self._compute_hand_instance_probabilities(hand_sorted)
            parts = [f"{self.tile_names[t]}[{p:.2f}%]" for t, p in zip(hand_sorted, inst_probs)]
            print("  手牌: " + ' '.join(parts))
        else:
            hand_str = ' '.join(self.tile_names[t] for t in self.players[self.current_player]['hand'])
            print(f"  手牌: {hand_str}")
        melds = self.players[self.current_player]['melds']
        meld_str = '[]' if not melds else ' | '.join(' '.join(self.tile_names[t] for t in meld) for meld in melds)
        print(f"  鳴牌區: {meld_str}")
        hidden_melds = self.players[self.current_player]['hidden_melds']
        if self.current_player == 0:
            hidden_meld_str = '[]' if not hidden_melds else ' | '.join(' '.join(self.tile_names[t] for t in meld) for meld in hidden_melds)
        else:
            hidden_meld_str = '[]' if not hidden_melds else ' | '.join('暗槓' for _ in hidden_melds)
        print(f"  暗槓區: {hidden_meld_str}")
        discards = self.players[self.current_player]['discards']
        discard_str = '[]' if not discards else f"[{', '.join(self.tile_names[t] for t in discards)}]"
        print(f"  棄牌堆: {discard_str}")
        if self.current_player == 0 and show_input_prompt:
            def count_tiles(tile_list):
                counter = [0] * 34
                for t in tile_list:
                    counter[t] += 1
                return counter
            def fmt_row(label, counts):
                return f"{label}: " + ''.join(pad_str_fixed_width(str(n), 4) for n in counts)
            print("種0: " + ''.join(pad_str_fixed_width(name, 4) for name in self.tile_names))
            hand_counts = count_tiles(self.players[0]['hand'])
            print(fmt_row("手0", hand_counts))
            in_counts = [0] * 34
            if in_tile is not None:
                in_counts[in_tile] = 1
            elif action_menu and not tsumo:
                if any(aid != ID_PASS for aid, _ in action_menu):
                    tile = self.last_tile_discarded
                    if tile is not None:
                        in_counts[tile] = 1
            print(fmt_row("進0", in_counts))
            for i in range(4):
                discard_counts = count_tiles(self.players[i]['discards'])
                print(fmt_row(f"棄{i}", discard_counts))
            for i in range(4):
                meld_counts = [0] * 34
                for meld in self.players[i]['melds']:
                    for t in meld:
                        meld_counts[t] += 1
                print(fmt_row(f"鳴{i}", meld_counts))
            hidden_counts = [0] * 34
            for meld in self.players[0]['hidden_melds']:
                for t in meld:
                    hidden_counts[t] += 1
            print(fmt_row("暗0", hidden_counts))
        if show_input_prompt:
            if action_menu:
                tile = self.last_tile_drawn if tsumo else self.last_tile_discarded
                print(f"  {'摸牌' if tsumo else '棄牌'}: {self.tile_names[tile] if tile is not None else ''}")
                print("可執行的動作（以動作ID輸入）：")
                for action_id, action_desc in action_menu:
                    print(f"{action_id}. {action_desc}")
                print("請輸入動作ID：")
            elif input_tile:
                print(f"  請輸入要打出的牌（輸入ID 0~33）：{input_tile}")
            else:
                print(f"  請輸入要打出的牌（輸入ID 0~33）：")
        if self.current_player == 0 and show_input_prompt:
            table = self.get_action_id_mask(player_idx=0, tsumo=tsumo)
            header = "種類  " + ''.join(pad_str_fixed_width(name, 4) for name in self.tile_names)
            print(header)
            row = "出牌  " + ''.join(pad_str_fixed_width(str(n), 4) for n in table['出牌'])
            print(row)
            print("過    " + pad_str_fixed_width(str(table['過'][0]), 4))
            print("胡    " + pad_str_fixed_width(str(table['胡'][0]), 4))
            for label in ['吃(左)', '吃(中)', '吃(右)', '碰', '明槓', '加槓', '暗槓']:
                row = pad_str_fixed_width(label, 6) + ''.join(pad_str_fixed_width(str(n), 4) for n in table[label])
                print(row)
        if after_discard:
            print(f"  出牌: {self.tile_names[self.last_tile_discarded] if self.last_tile_discarded is not None else ''}")
        if self.current_player == 0 and self.last_discard_efficiency is not None:
            eff = self.last_discard_efficiency
            preview = self.compute_reward_for_action("discard", eff)
            print(f"(preview) reward: {preview:.4f} (eff={eff:.2f}%)")
        print("---------------------------------")
    def compute_reward_for_action(self, act_type: str, delta: float) -> float:
        raw_reward = 0.0
        if act_type == "chi":
            raw_reward = self.reward_weights["chi"] * (delta / 100.0)
        elif act_type == "pong":
            raw_reward = self.reward_weights["pong"] * (delta / 100.0)
        elif act_type == "kong":
            raw_reward = self.reward_weights["ming_kong"] * (delta / 100.0)
        elif act_type == "add_kong":
            raw_reward = self.reward_weights["add_kong"] * (delta / 100.0)
        elif act_type == "concealed_kong":
            raw_reward = self.reward_weights["concealed_kong"] * (delta / 100.0)
        elif act_type in ("discard", "出牌"):
            eff = max(0.0, min(100.0, delta))
            raw_reward = self.reward_weights["efficiency"] * (0.5 - eff / 100.0)
        raw_reward *= (1.0 - self.terminal_focus)
        try:
            with open("贏局分析.txt", "a", encoding="utf-8") as f_log:
                f_log.write(f"{self.terminal_focus:.4f}, {raw_reward:.6f}\n")
        except Exception:
            pass
        return self._clip_and_normalize(raw_reward)
    def potential(self, player_idx: int = 0) -> float:
        hand = self.players[player_idx]['hand'][:]
        count = self._hand_count_from_list(hand)
        used = self._public_used_tiles_with_extra(count, extra_used=None)
        extra = self._existing_meld_cnt(player_idx)
        return self._total_efficiency_percent(count, used, extra_melds=extra) / 100.0
    def _clip_and_normalize(self, reward: float, min_val=-1.0, max_val=1.0) -> float:
        reward = max(min(reward, max_val), min_val)
        return reward
    def set_terminal_focus(self, f: float):
        self.terminal_focus = max(0.0, min(1.0, float(f)))
def main():
    env = MahjongEnv()
    env.reset()
    while not env.game_over:
        env.action_performed = False
        env.tile_claimed = False
        if env.last_tile_discarded is not None and env.last_discard_player is not None:
            claimed = False
            for offset in [1, 2, 3]:
                if claimed:
                    break
                player_idx = (env.last_discard_player + offset) % 4
                env.current_player = player_idx
                id_menu, base_eff, rewards = env.get_action_id_menu_with_eff(player_idx, tsumo=False)
                if len(id_menu) == 1 and id_menu[0][0] == ID_PASS:
                    continue
                if player_idx == 0:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    env.render(show_input_prompt=True, action_menu=id_menu, tsumo=False)
                    choice = input().strip()
                    if not choice or not choice.isdigit():
                        env.step(('pass', None))
                        continue
                    action_id = int(choice)
                    act_type, param = env.decode_action_id(action_id, tsumo=False)
                    if act_type is None or act_type == 'pass':
                        env.step(('pass', None))
                        continue
                    if act_type == 'win':
                        state, msg = env.step(('win', 'ron'))
                        print(msg)
                        return
                    else:
                        delta = rewards.get(action_id, 0.0)
                        reward_val = env.compute_reward_for_action(act_type, delta)
                        print(f"reward: {reward_val:.4f} (eff+={delta:.2f}%)")
                        env.total_reward += reward_val
                        state, msg = env.step((act_type, param))
                        if "無效" not in msg:
                            env.action_performed = True
                            claimed = True
                            break
                        else:
                            print(msg)
                            env.step(('pass', None))
                            continue
                else:
                    choose = None
                    d = env.last_tile_discarded
                    if any(aid == ID_WIN for aid, _ in id_menu):
                        choose = ('win', 'ron')
                    elif any(aid == ID_KONG_START + d for aid, _ in id_menu):
                        choose = ('kong', None)
                    elif any(aid == ID_PONG_START + d for aid, _ in id_menu):
                        choose = ('pong', None)
                    else:
                        chi = env.get_valid_chi_combinations(player_idx, d)
                        if any(aid == ID_CHI_L_START + d for aid, _ in id_menu) and chi[2]:
                            choose = ('chi', chi[2])
                        elif any(aid == ID_CHI_M_START + d for aid, _ in id_menu) and chi[3]:
                            choose = ('chi', chi[3])
                        elif any(aid == ID_CHI_R_START + d for aid, _ in id_menu) and chi[4]:
                            choose = ('chi', chi[4])
                    if choose:
                        state, msg = env.step(choose)
                        if choose[0] == 'win':
                            env.game_over = True
                            print(f"玩家{player_idx} 胡牌！(榮和)")
                            return
                        if "無效" not in msg:
                            env.action_performed = True
                            claimed = True
                            break
                    env.step(('pass', None))
        if env.action_performed:
            is_kong = any(meld[-1] == env.last_tile_discarded and len(meld) == 4 for meld in env.players[env.current_player]['melds']) or \
                      any(meld[-1] == env.last_tile_drawn and len(meld) == 4 for meld in env.players[env.current_player]['melds']) or \
                      any(meld[-1] == env.last_tile_drawn for meld in env.players[env.current_player]['hidden_melds'])
            if is_kong:
                state, msg = env.draw_tile()
                if env.game_over:
                    print(msg)
                    return
                env.last_tile_drawn = None
            env.last_tile_discarded = None
            env.last_discard_player = None
            if env.current_player == 0:
                while True:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    env.render(show_input_prompt=True)
                    choice = input().strip()
                    if not choice or not choice.isdigit():
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print("請輸入 0~33 的出牌ID")
                        env.render(show_input_prompt=True)
                        continue
                    action_id = int(choice)
                    act_type, param = env.decode_action_id(action_id, tsumo=False)
                    if act_type != 'discard':
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print("此階段只能出牌（ID 0~33）")
                        env.render(show_input_prompt=True)
                        continue
                    temp_hand = env.players[env.current_player]['hand'].copy()
                    state, msg = env.step((act_type, param))
                    if "無效" not in msg:
                        os.system('cls' if os.name == 'nt' else 'clear')
                        env.render(show_input_prompt=False, after_discard=True)
                        break
                    else:
                        env.players[env.current_player]['hand'] = temp_hand
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print(msg)
                        env.render(show_input_prompt=True)
            else:
                player_hand = env.players[env.current_player]['hand']
                candidates = [t for t in player_hand if t != env.last_claimed_tile]
                if not candidates:
                    candidates = player_hand[:]
                tile = random.choice(candidates)
                tile_name = env.tile_names[tile]
                state, msg = env.step(('discard', tile_name))
                os.system('cls' if os.name == 'nt' else 'clear')
                env.render(show_input_prompt=False, after_discard=True)
            continue
        if env.last_tile_discarded is not None and env.last_discard_player is not None and not env.tile_claimed:
            env.players[env.last_discard_player]['discards'].append(env.last_tile_discarded)
            env.current_player = (env.last_discard_player + 1) % 4
            env.last_tile_discarded = None
            env.last_discard_player = None
        state, msg = env.step(('draw', None))
        if env.game_over:
            print(msg)
            break
        action_menu = env.get_action_menu(env.current_player, tsumo=True)
        deck_empty = len(env.deck) <= 16
        if any(action_id != 0 for action_id, _ in action_menu):
            if env.current_player == 0:
                while True:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    id_menu, base_eff, rewards = env.get_action_id_menu_with_eff(env.current_player, tsumo=True)
                    env.render(show_input_prompt=True, action_menu=id_menu, tsumo=True)
                    choice = input().strip()
                    if not choice:
                        state, msg = env.step(('pass', None))
                        env.last_tile_drawn = None
                        break
                    if not choice.isdigit():
                        print("無效的動作ID")
                        continue
                    action_id = int(choice)
                    act_type, param = env.decode_action_id(action_id, tsumo=True)
                    if act_type is None:
                        print(param)
                        continue
                    if act_type == 'pass':
                        state, msg = env.step(('pass', None))
                        env.last_tile_drawn = None
                        break
                    if act_type == 'win':
                        state, msg = env.step(('win', 'tsumo'))
                        print(msg)
                        return
                    elif act_type == 'add_kong':
                        delta = rewards.get(action_id, 0.0)
                        reward_val = env.compute_reward_for_action(act_type, delta)
                        print(f"reward: {reward_val:.4f} (eff+={delta:.2f}%)")
                        env.total_reward += reward_val
                        state, msg = env.step(('add_kong', param))
                        if "無效" not in msg:
                            env.action_performed = True
                            if not deck_empty:
                                env.draw_tile()
                            else:
                                print("剩餘可摸牌為 0，加槓後不得補牌")
                            break
                        else:
                            print(msg)
                    elif act_type == 'concealed_kong':
                        delta = rewards.get(action_id, 0.0)
                        reward_val = env.compute_reward_for_action(act_type, delta)
                        print(f"reward: {reward_val:.4f} (eff+={delta:.2f}%)")
                        env.total_reward += reward_val
                        state, msg = env.step(('concealed_kong', param))
                        if "無效" not in msg:
                            env.action_performed = True
                            if not deck_empty:
                                env.draw_tile()
                            else:
                                print("剩餘可摸牌為 0，加槓後不得補牌")
                            break
                        else:
                            print(msg)
                    else:
                        print("此階段僅能：過/胡/加槓/暗槓（或直接過）")
            else:
                valid_actions = [(action_id, desc) for action_id, desc in action_menu if action_id != 0]
                if valid_actions:
                    for action_id, _ in sorted(valid_actions, key=lambda x: x[0], reverse=True):
                        if action_id == 1:
                            state, msg = env.step(('win', 'tsumo'))
                            if "無效" not in msg:
                                env.game_over = True
                                print(f"玩家{env.current_player} 自摸胡牌！")
                                return
                        elif action_id in [11, 12, 13, 14]:
                            concealed_kong_tiles = env.can_concealed_kong(env.current_player)
                            if action_id - 11 < len(concealed_kong_tiles):
                                state, msg = env.step(('concealed_kong', concealed_kong_tiles[action_id - 11]))
                                if "無效" not in msg:
                                    env.action_performed = True
                                    if not deck_empty:
                                        env.draw_tile()
                                    else:
                                        print("剩餘可摸牌為 0，加槓後不得補牌")
                                    break
                        elif action_id in [7, 8, 9, 10]:
                            add_kong_tiles = env.can_add_kong(env.current_player, env.last_tile_drawn)
                            if action_id - 7 < len(add_kong_tiles):
                                state, msg = env.step(('add_kong', add_kong_tiles[action_id - 7]))
                                if "無效" not in msg:
                                    env.action_performed = True
                                    if not deck_empty:
                                        env.draw_tile()
                                    else:
                                        print("剩餘可摸牌為 0，加槓後不得補牌")
                                    break
                        if env.action_performed:
                            break
        env.last_tile_drawn = None
        if env.current_player == 0:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                env.render(show_input_prompt=True)
                choice = input().strip()
                if not choice or not choice.isdigit():
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("請輸入 0~33 的出牌ID")
                    env.render(show_input_prompt=True)
                    continue
                action_id = int(choice)
                act_type, param = env.decode_action_id(action_id, tsumo=False)
                if act_type != 'discard':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("此階段只能出牌（ID 0~33）")
                    env.render(show_input_prompt=True)
                    continue
                temp_hand = env.players[env.current_player]['hand'].copy()
                state, msg = env.step((act_type, param))
                if "無效" not in msg:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    env.render(show_input_prompt=False, after_discard=True)
                    break
                else:
                    env.players[env.current_player]['hand'] = temp_hand
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(msg)
                    env.render(show_input_prompt=True)
        else:
            player_hand = env.players[env.current_player]['hand']
            candidates = [t for t in player_hand if t != env.last_claimed_tile]
            if not candidates:
                candidates = player_hand[:]
            tile = random.choice(candidates)
            tile_name = env.tile_names[tile]
            state, msg = env.step(('discard', tile_name))
            os.system('cls' if os.name == 'nt' else 'clear')
            env.render(show_input_prompt=False, after_discard=True)
class RewardScheduler:
    def __init__(self, window=100):
        self.phase = "early"
        self.history = deque(maxlen=window)
        self.smooth_phase_factor = 0.0
    def update_phase(self):
        if not self.history:
            return
        win_rate = np.mean([1 if r > 0 else 0 for r in self.history])
        target_phase = (
            "early" if win_rate < 0.08 else
            "mid" if win_rate < 0.18 else
            "late"
        )
        self.smooth_phase_factor = 0.9 * self.smooth_phase_factor + 0.1 * (["early", "mid", "late"].index(target_phase) / 2)
        self.phase = ["early", "mid", "late"][int(round(self.smooth_phase_factor * 2))]
    def get_weights(self):
        f = self.smooth_phase_factor
        ron = 8 + f * (15 - 8)
        tsumo = 10 + f * (17 - 10)
        base = 0.5 + f * 0.5
        return {
            "efficiency": 1.2 * base,
            "chi": 0.2 * base,
            "pong": 0.3 * base,
            "ming_kong": 0.6 * base,
            "add_kong": 0.6 * base,
            "concealed_kong": 1.0 * base,
            "ron": ron,
            "tsumo": tsumo,
            "lose_ron": -0.8 * ron,
            "lose_tsumo": -0.6 * tsumo,
        }
    def record(self, reward_sum):
        self.history.append(reward_sum)
        self.update_phase()
# ===== 最近N局胡牌率長期加分 =====
class LongTermWinRateBonus:
    def __init__(self, window=3000, alpha=2.0):
        self.window = int(window)
        self.alpha = float(alpha)
        self.hist = deque(maxlen=self.window)
        self.current_rate = 0.0
    def add(self, is_win: int):
        self.hist.append(1 if is_win else 0)
        if len(self.hist) > 0:
            self.current_rate = sum(self.hist) / len(self.hist)
        else:
            self.current_rate = 0.0
    def bonus(self, terminal_focus: float = 0.0) -> float:
        scale = (0.5 + 0.5 * terminal_focus)
        bonus_val = scale * self.alpha * (self.current_rate ** 1.5) * 3.0
        return bonus_val * 8.0
# ===== RL 訓練包裝：MahjongRLTrainEnvV2 =====
class MahjongRLTrainEnvV2(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, log_path="贏局分析.txt", log_enabled=True, winrate_window=3000, winrate_alpha=2.0):
        super().__init__()
        self.env = MahjongEnv()
        self.action_space = spaces.Discrete(274)
        self.observation_space = spaces.Box(low=0, high=4, shape=(374,), dtype=np.int32)
        self.max_turn_guard = 10000
        self._last_info_for_rewards = None
        self.ep_reward_sum = 0.0
        self.log_enabled = log_enabled
        self.log_f = open(log_path, "w", encoding="utf-8") if log_enabled else None
        self.scheduler = RewardScheduler(window=50)
        self.ltwr = LongTermWinRateBonus(window=winrate_window, alpha=winrate_alpha)
        self.log_buffer = []
    # ========= 工具：把 render() 的輸出行，寫成 txt =========
    def _log(self, s=""):
        if not self.log_enabled:
            return
        line = str(s) + ("\n" if not str(s).endswith("\n") else "")
        self.log_buffer.append(line)
    def _log_like_render(self, show_masks=True):
        e = self.env
        display_deck = max(0, len(e.deck) - 16)
        player_str = "您" if e.current_player == 0 else f"玩家{e.current_player}"
        banker_str = f" (莊×{e.banker_wins})" if e.current_player == e.banker else ""
        self._log(f"牌庫剩餘: {display_deck} 張")
        self._log(f"當前玩家: {player_str}{banker_str}")
        self._log(f"摸牌: {e.tile_names[e.last_tile_drawn] if e.last_tile_drawn is not None else ''}")
        hand_str = ' '.join(e.tile_names[t] for t in e.players[e.current_player]['hand'])
        self._log(f"  手牌: {hand_str}")
        melds = e.players[e.current_player]['melds']
        meld_str = '[]' if not melds else ' | '.join(' '.join(e.tile_names[t] for t in meld) for meld in melds)
        self._log(f"  鳴牌區: {meld_str}")
        hidden_melds = e.players[e.current_player]['hidden_melds']
        if e.current_player == 0:
            hidden_meld_str = '[]' if not hidden_melds else ' | '.join(' '.join(e.tile_names[t] for t in meld) for meld in hidden_melds)
        else:
            hidden_meld_str = '[]' if not hidden_melds else ' | '.join('暗槓' for _ in hidden_melds)
        self._log(f"  暗槓區: {hidden_meld_str}")
        discards = e.players[e.current_player]['discards']
        discard_str = '[]' if not discards else f"[{', '.join(e.tile_names[t] for t in discards)}]"
        self._log(f"  棄牌堆: {discard_str}")
        def count_tiles(tile_list):
            counter = [0] * 34
            for t in tile_list:
                counter[t] += 1
            return counter
        def fmt_row(label, counts):
            pad = lambda x: f"{x}{' ' * (4 - len(str(x)))}"
            return f"{label}: " + ''.join(pad(n) for n in counts)
        self._log("種0: " + ''.join(f"{name}{' ' * (4 - (2 if len(name)==1 else 0))}" if len(name)==2 else f"{name}{' ' * (4-len(name))}" for name in e.tile_names))
        hand_counts = count_tiles(e.players[0]['hand'])
        self._log(fmt_row("手0", hand_counts))
        in_counts = [0] * 34
        if (e.last_tile_discarded is not None and
            e.last_discard_player is not None and
            not e.tile_claimed and
            e.current_player == 0):
            id_menu = e.get_action_id_menu(player_idx=0, tsumo=False)
            if any(aid != ID_PASS for aid, _ in id_menu):
                in_counts[e.last_tile_discarded] = 1
        self._log(fmt_row("進0", in_counts))
        for i in range(4):
            discard_counts = count_tiles(e.players[i]['discards'])
            self._log(fmt_row(f"棄{i}", discard_counts))
        for i in range(4):
            meld_counts = [0] * 34
            for meld in e.players[i]['melds']:
                for t in meld:
                    meld_counts[t] += 1
            self._log(fmt_row(f"鳴{i}", meld_counts))
        hidden_counts = [0] * 34
        for meld in e.players[0]['hidden_melds']:
            for t in meld:
                hidden_counts[t] += 1
        self._log(fmt_row("暗0", hidden_counts))
        if show_masks:
            has_live_discard = (e.last_tile_discarded is not None and e.last_discard_player is not None and not e.tile_claimed)
            tsumo_ctx = (e.current_player == 0 and e.last_tile_drawn is not None and not has_live_discard)
            table = e.get_action_id_mask(player_idx=0, tsumo=tsumo_ctx)
            pad = lambda s, w: f"{s}{' ' * (w - len(s))}"
            self._log("種類  " + ''.join(pad(name, 4) for name in e.tile_names))
            self._log("出牌  " + ''.join(pad(str(n), 4) for n in table['出牌']))
            self._log("過    " + pad(str(table['過'][0]), 4))
            self._log("胡    " + pad(str(table['胡'][0]), 4))
            for label in ['吃(左)', '吃(中)', '吃(右)', '碰', '明槓', '加槓', '暗槓']:
                self._log(pad(label, 6) + ''.join(pad(str(n), 4) for n in table[label]))
        self._log("---------------------------------")
    def _log_player_action(self, pid, text, action_id=None, suffix=None):
        who = "AI" if pid == 0 else f"玩家{pid}"
        if action_id is not None:
            line = f"[{who}] {text}"
        else:
            line = f"[{who}] {text}"
        if suffix:
            line = f"{line} {suffix}"
        self._log(line)
    def _names(self, tiles):
        return ' '.join(self.env.tile_names[t] for t in tiles)
    def _desc_action(self, act_type, param, tsumo_ctx):
        e = self.env
        if act_type == 'discard':
            return f"打出 {param}"
        if act_type == 'pass':
            return "過"
        if act_type == 'win':
            return "自摸胡" if tsumo_ctx else "榮和"
        if act_type == 'chi':
            if param and isinstance(param[0], str):
                return f"吃 [{' '.join(param)}]"
            else:
                return f"吃 [{self._names(param)}]"
        if act_type == 'pong':
            t = e.last_tile_discarded
            return f"碰 {e.tile_names[t]}" if t is not None else "碰"
        if act_type == 'kong':
            t = e.last_tile_discarded
            return f"明槓 {e.tile_names[t]}" if t is not None else "明槓"
        if act_type == 'add_kong':
            return f"加槓 {e.tile_names[param]}"
        if act_type == 'concealed_kong':
            return f"暗槓 {e.tile_names[param]}"
        return act_type
    # ========= Gym API =========
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.log_buffer.clear()
        self.env.reset()
        self.ep_reward_sum = 0.0
        self.ep_length = 0
        obs = self._build_obs()
        self._log("\n===== 新局開始 =====")
        if self.env.game_over:
            return obs, {"action_mask": self._build_action_mask()}
        _state, _msg = self.env.step(('draw', None))
        obs = self._build_obs()
        if self.env.current_player == 0:
            self._log_like_render(show_masks=True)
        else:
            self._log_player_action(self.env.current_player, "摸牌")
            e = self.env
            player_hand = e.players[e.current_player]['hand']
            candidates = [t for t in player_hand if t != e.last_claimed_tile] or player_hand[:]
            tile = random.choice(candidates)
            tile_name = e.tile_names[tile]
            self._log_player_action(e.current_player, f"打出 {tile_name}", action_id=ID_DISCARD_START + tile)
            _state, _msg = e.step(('discard', tile_name))
            e.last_tile_drawn = None
        return obs, {"action_mask": self._build_action_mask()}
    def step(self, action):
        e = self.env
        reward = 0.0
        self.ep_length += 1
        phi_s = e.potential(player_idx=0) if e.current_player == 0 else 0.0
        aid = int(action)
        terminal_focus = getattr(e, "terminal_focus", 0.0)
        END_GAIN_BOOST = 0.5 + 0.5 * terminal_focus
        LAMBDA_SHAPING = 0.7 * (1.0 - terminal_focus)
        def finish(terminated=False, end_line=None, end_delta=0.0, info=None):
            total_this_step = reward + (end_delta if terminated else 0.0)
            self.ep_reward_sum += total_this_step
            if info is None:
                info = {"action_mask": self._build_action_mask()}
            if end_line is not None:
                info["end_line"] = end_line
                text = str(end_line)
                evt = None
                if "AI胡牌" in text and "自摸" in text:
                    evt = "AI_TSUMO"
                elif "AI胡牌" in text:
                    evt = "AI_RON"
                elif ("AI放槍" in text) or ("玩家0 放槍" in text):
                    evt = "AI_DEAL_IN"
                elif ("電腦玩家" in text and "自摸" in text) or ("被自摸" in text):
                    evt = "AI_LOST_TSUMO"
                elif "流局" in text:
                    evt = "DRAW"
                if evt:
                    info["terminal_event"] = evt
                    info["ai_ron"] = 1 if evt == "AI_RON" else 0
                    info["ai_tsumo"] = 1 if evt == "AI_TSUMO" else 0
                    info["ai_deal_in"] = 1 if evt == "AI_DEAL_IN" else 0
                    info["ai_lost_tsumo"] = 1 if evt == "AI_LOST_TSUMO" else 0
            obs = self._build_obs()
            if terminated:
                if end_line is None:
                    end_line = "終局"
                self._log(end_line)
                evt = info.get("terminal_event", None)
                is_win = 1 if evt in ("AI_TSUMO", "AI_RON") else 0
                self.ltwr.add(is_win)
                lt_bonus = self.ltwr.bonus(self.env.terminal_focus)
                total_this_step += lt_bonus
                self.ep_reward_sum += lt_bonus
                self._log(f"[LongTerm] window={self.ltwr.window}, win_rate={self.ltwr.current_rate:.3f}, bonus={lt_bonus:+.4f}")
                self._log(f"總reward: {self.ep_reward_sum:.4f}")
                self.scheduler.record(self.ep_reward_sum)
                self.env.reward_weights = self.scheduler.get_weights()
                self._log(f"[Scheduler] Phase={self.scheduler.phase}, 新權重={self.env.reward_weights}")
                # === 只在 AI 胡牌時才輸出整場紀錄 ===
                if is_win and self.log_enabled and self.log_f is not None:
                    self.log_f.write("\n" + "="*40 + "\n")
                    self.log_f.write(f"🀄 結果事件: {evt}\n")
                    self.log_f.writelines(self.log_buffer)
                    self.log_f.write("="*40 + "\n\n")
                    self.log_f.flush()
                self.log_buffer.clear()
                info["lt_winrate_bonus"] = {
                    "bonus": lt_bonus,
                    "win_rate": self.ltwr.current_rate,
                    "window": self.ltwr.window,
                    "alpha": self.ltwr.alpha,
                    "is_win": is_win
                }
                info["episode"] = {
                    "r": self.ep_reward_sum,
                    "l": self.ep_length
                }
            return obs, total_this_step, terminated, False, info
        tsumo_ctx = (e.current_player == 0 and e.last_tile_drawn is not None)
        action_rewards = {}
        if e.current_player == 0:
            _menu, _base_eff, rewards = e.get_action_id_menu_with_eff(0, tsumo=tsumo_ctx)
            action_rewards = rewards
        act_type, param_or_err = e.decode_action_id(aid, tsumo=tsumo_ctx)
        if e.current_player == 0 and act_type is None:
            self._log_like_render(show_masks=True)
            reward = -1.0
            return finish(terminated=False, info={"action_mask": self._build_action_mask(), "invalid_action": True})
        if e.current_player == 0 and act_type is not None:
            desc_for_log = self._desc_action(act_type, param_or_err, tsumo_ctx)
            if act_type == 'win':
                is_tsumo = tsumo_ctx
                _state, _msg = e.step(('win', 'tsumo' if is_tsumo else 'ron'))
                if is_tsumo:
                    reward_val = self.env.reward_weights["tsumo"] * END_GAIN_BOOST
                    return finish(
                        terminated=True,
                        end_line=f"AI胡牌(自摸)->reward+{reward_val}",
                        end_delta=reward_val
                    )
                else:
                    from_pid = e.last_discard_player if e.last_discard_player is not None else "?"
                    reward_val = self.env.reward_weights["ron"] * END_GAIN_BOOST
                    return finish(
                        terminated=True,
                        end_line=f"AI胡牌(玩家{from_pid}放槍)->reward+{reward_val}",
                        end_delta=reward_val
                    )
            if act_type == 'pass':
                _state, _msg = e.step(('pass', None))
                self._log_player_action(0, desc_for_log, action_id=aid, suffix="| reward=+0.0000")
            elif act_type in ('chi', 'pong', 'kong'):
                delta_eff = action_rewards.get(aid, 0.0)
                _state, _msg = e.step((act_type, param_or_err))
                reward_val = e.compute_reward_for_action(act_type, delta_eff)
                reward += reward_val
                self._log_player_action(0, desc_for_log, action_id=aid, suffix=f"| reward={reward_val:+.4f} (eff={delta_eff:.2f}%)")
                e.action_performed = True
                e.tile_claimed = True
                e.last_discard_player = None
                e.last_tile_discarded = None
                if act_type == 'kong':
                    if len(e.deck) > 16:
                        e.draw_tile()
                    e.last_tile_drawn = None
                self._log_like_render(show_masks=True)
                return finish(terminated=False)
            elif act_type in ('add_kong', 'concealed_kong'):
                delta_eff = action_rewards.get(aid, 0.0)
                _state, _msg = e.step((act_type, param_or_err))
                reward_val = e.compute_reward_for_action(act_type, delta_eff)
                reward += reward_val
                self._log_player_action(0, desc_for_log, action_id=aid, suffix=f"| reward={reward_val:+.4f} (eff={delta_eff:.2f}%)")
                if len(e.deck) > 16:
                    e.draw_tile()
                e.last_tile_drawn = None
                self._log_like_render(show_masks=True)
                return finish(terminated=False)
            elif act_type == 'discard':
                _state, _msg = e.step(('discard', param_or_err))
                ai_disc_reward = 0.0
                if e.last_discard_efficiency is not None:
                    eff = e.last_discard_efficiency
                    ai_disc_reward = e.compute_reward_for_action("discard", eff)
                    reward += ai_disc_reward
                    self._log_player_action(
                        0, desc_for_log, action_id=aid,
                        suffix=f"| reward={ai_disc_reward:+.4f} (eff={eff:.2f}%)"
                    )
        if e.last_tile_discarded is not None and e.last_discard_player is not None and not e.tile_claimed:
            claimed = False
            for offset in [1, 2, 3]:
                if claimed:
                    break
                pid = (e.last_discard_player + offset) % 4
                e.current_player = pid
                id_menu, _base_eff, _rewards = e.get_action_id_menu_with_eff(pid, tsumo=False)
                d = e.last_tile_discarded
                if pid == 0 and any(aid_ != ID_PASS for aid_, _ in id_menu):
                    self._log_like_render(show_masks=True)
                    return finish(terminated=False)
                if pid != 0 and all(aid_ == ID_PASS for aid_, _ in id_menu):
                    _state, _msg = e.step(('pass', None))
                    continue
                if any(aid_ == ID_WIN for aid_, _ in id_menu):
                    e.game_over = True
                    if e.last_discard_player == 0:
                        reward_val = self.env.reward_weights["lose_ron"] * END_GAIN_BOOST
                        return finish(
                            terminated=True,
                            end_line=f"AI放槍(玩家{pid}胡牌)->reward{reward_val}",
                            end_delta=reward_val
                        )
                    else:
                        return finish(
                            terminated=True,
                            end_line=f"玩家{pid} 榮和！",
                            end_delta=0.0
                        )
                choose = None
                choose_id = None
                if d is not None:
                    if any(aid_ == ID_KONG_START + d for aid_, _ in id_menu):
                        choose = ('kong', None); choose_id = ID_KONG_START + d
                    elif any(aid_ == ID_PONG_START + d for aid_, _ in id_menu):
                        choose = ('pong', None); choose_id = ID_PONG_START + d
                    else:
                        chi = e.get_valid_chi_combinations(pid, d)
                        if any(aid_ == ID_CHI_L_START + d for aid_, _ in id_menu) and chi[2]:
                            choose = ('chi', chi[2]); choose_id = ID_CHI_L_START + d
                        elif any(aid_ == ID_CHI_M_START + d for aid_, _ in id_menu) and chi[3]:
                            choose = ('chi', chi[3]); choose_id = ID_CHI_M_START + d
                        elif any(aid_ == ID_CHI_R_START + d for aid_, _ in id_menu) and chi[4]:
                            choose = ('chi', chi[4]); choose_id = ID_CHI_R_START + d
                if choose:
                    self._log_player_action(pid, self._desc_action(choose[0], choose[1], False), action_id=choose_id)
                    _state, _msg = e.step(choose)
                    if "無效" not in _msg:
                        claimed = True
                        e.action_performed = True
                        if choose[0] == 'kong':
                            if len(e.deck) > 16:
                                e.draw_tile()
                                self._log_player_action(e.current_player, "槓補摸牌")
                            e.last_tile_drawn = None
                        player_hand = e.players[e.current_player]['hand']
                        candidates = [t for t in player_hand if t != e.last_claimed_tile] or player_hand[:]
                        tile = random.choice(candidates)
                        tile_name = e.tile_names[tile]
                        discard_id = ID_DISCARD_START + tile
                        self._log_player_action(e.current_player, f"打出 {tile_name}", action_id=discard_id)
                        _state, _msg = e.step(('discard', tile_name))
                        break
        if e.last_tile_discarded is not None and e.last_discard_player is not None and not e.tile_claimed:
            e.players[e.last_discard_player]['discards'].append(e.last_tile_discarded)
            next_player = (e.last_discard_player + 1) % 4
            e.last_tile_discarded = None
            e.last_discard_player = None
            e.current_player = next_player
            e.passed_players_for_current_discard = set()
        else:
            if e.last_discard_player is not None:
                e.current_player = (e.last_discard_player + 1) % 4
                e.last_discard_player = None
            else:
                e.current_player = (e.current_player + 1) % 4
        _state, msg = e.step(('draw', None))
        if e.game_over:
            return finish(terminated=True, end_line="流局->reward+0", end_delta=0.0)
        if e.current_player != 0:
            self._log_player_action(e.current_player, "摸牌")
        turn_guard = 0
        while e.current_player != 0 and not e.game_over and turn_guard < self.max_turn_guard:
            turn_guard += 1
            action_menu = e.get_action_menu(e.current_player, tsumo=True)
            deck_empty = len(e.deck) <= 16
            if any(aid_ == 1 for aid_, _ in action_menu):
                _state, _msg = e.step(('win', 'tsumo'))
                reward_val = self.env.reward_weights["lose_tsumo"] * END_GAIN_BOOST
                return finish(
                    terminated=True,
                    end_line=f"電腦玩家{e.current_player}自摸->reward{reward_val}",
                    end_delta=reward_val
                )
            acted = False
            for action_id_, _ in action_menu:
                if action_id_ in [11, 12, 13, 14]:
                    conc = e.can_concealed_kong(e.current_player)
                    if conc:
                        tile = conc[0]
                        _state, _msg = e.step(('concealed_kong', tile))
                        if "無效" not in _msg:
                            acted = True
                            self._log_player_action(e.current_player, f"暗槓 {e.tile_names[tile]}", action_id=ID_CONC_KONG_START + tile)
                            if not deck_empty:
                                e.draw_tile()
                                self._log_player_action(e.current_player, "槓補摸牌")
                            break
                if action_id_ in [7, 8, 9, 10]:
                    addk = e.can_add_kong(e.current_player, e.last_tile_drawn)
                    if addk:
                        tile = addk[0]
                        _state, _msg = e.step(('add_kong', tile))
                        if "無效" not in _msg:
                            acted = True
                            self._log_player_action(e.current_player, f"加槓 {e.tile_names[tile]}", action_id=ID_ADD_KONG_START + tile)
                            if not deck_empty:
                                e.draw_tile()
                                self._log_player_action(e.current_player, "槓補摸牌")
                            break
            e.last_tile_drawn = None
            player_hand = e.players[e.current_player]['hand']
            candidates = [t for t in player_hand if t != e.last_claimed_tile] or player_hand[:]
            tile = random.choice(candidates)
            tile_name = e.tile_names[tile]
            discard_id = ID_DISCARD_START + tile
            self._log_player_action(e.current_player, f"打出 {tile_name}", action_id=discard_id)
            _state, _msg = e.step(('discard', tile_name))
            e.last_tile_drawn = None
            if e.last_tile_discarded is not None and e.last_discard_player is not None and not e.tile_claimed:
                claimed = False
                for offset in [1, 2, 3]:
                    if claimed:
                        break
                    pid = (e.last_discard_player + offset) % 4
                    e.current_player = pid
                    id_menu, _base_eff, _rewards = e.get_action_id_menu_with_eff(pid, tsumo=False)
                    d = e.last_tile_discarded
                    if pid == 0 and any(aid_ != ID_PASS for aid_, _ in id_menu):
                        self._log_like_render(show_masks=True)
                        return finish(terminated=False)
                    if any(aid_ == ID_WIN for aid_, _ in id_menu):
                        e.game_over = True
                        if e.last_discard_player == 0:
                            lose_ron = self.env.reward_weights["lose_ron"]
                            return finish(
                                terminated=True,
                                end_line=f"玩家{pid} 榮和！玩家0 放槍！ {lose_ron:+.1f}",
                                end_delta=lose_ron
                            )
                        else:
                            return finish(
                                terminated=True,
                                end_line=f"玩家{pid} 榮和！",
                                end_delta=0.0
                            )
                    choose = None
                    choose_id = None
                    if d is not None:
                        if any(aid_ == ID_KONG_START + d for aid_, _ in id_menu):
                            choose = ('kong', None); choose_id = ID_KONG_START + d
                        elif any(aid_ == ID_PONG_START + d for aid_, _ in id_menu):
                            choose = ('pong', None); choose_id = ID_PONG_START + d
                        else:
                            chi = e.get_valid_chi_combinations(pid, d)
                            if any(aid_ == ID_CHI_L_START + d for aid_, _ in id_menu) and chi[2]:
                                choose = ('chi', chi[2]); choose_id = ID_CHI_L_START + d
                            elif any(aid_ == ID_CHI_M_START + d for aid_, _ in id_menu) and chi[3]:
                                choose = ('chi', chi[3]); choose_id = ID_CHI_M_START + d
                            elif any(aid_ == ID_CHI_R_START + d for aid_, _ in id_menu) and chi[4]:
                                choose = ('chi', chi[4]); choose_id = ID_CHI_R_START + d
                    if choose:
                        self._log_player_action(pid, self._desc_action(choose[0], choose[1], False), action_id=choose_id)
                        _state, _msg = e.step(choose)
                        if "無效" not in _msg:
                            claimed = True
                            e.action_performed = True
                            if choose[0] == 'kong':
                                if len(e.deck) > 16:
                                    e.draw_tile()
                                    self._log_player_action(e.current_player, "槓補摸牌")
                                e.last_tile_drawn = None
                            player_hand2 = e.players[e.current_player]['hand']
                            candidates2 = [t for t in player_hand2 if t != e.last_claimed_tile] or player_hand2[:]
                            tile2 = random.choice(candidates2)
                            tile_name2 = e.tile_names[tile2]
                            discard_id2 = ID_DISCARD_START + tile2
                            self._log_player_action(e.current_player, f"打出 {tile_name2}", action_id=discard_id2)
                            _state, _msg = e.step(('discard', tile_name2))
            if e.last_tile_discarded is not None and e.last_discard_player is not None and not e.tile_claimed:
                e.players[e.last_discard_player]['discards'].append(e.last_tile_discarded)
                next_player = (e.last_discard_player + 1) % 4
                e.last_tile_discarded = None
                e.last_discard_player = None
                e.current_player = next_player
                e.passed_players_for_current_discard = set()
            else:
                if e.last_discard_player is not None:
                    e.current_player = (e.last_discard_player + 1) % 4
                    e.last_discard_player = None
                else:
                    e.current_player = (e.current_player + 1) % 4
            _state, _msg = e.step(('draw', None))
            if e.game_over:
                return finish(terminated=True, end_line="流局->reward+0", end_delta=0.0)
            if e.current_player != 0:
                self._log_player_action(e.current_player, "摸牌")
        if not e.game_over and e.current_player == 0:
            self._log_like_render(show_masks=True)
        lambda_shaping = 0.7 * (1.0 - self.env.terminal_focus)
        gamma = 0.995
        if e.current_player == 0 or (e.last_discard_player == 0):
            phi_sp = e.potential(player_idx=0)
            shaping_reward = lambda_shaping * (gamma * phi_sp - phi_s)
            reward += shaping_reward
            self._log(
                f"[Shaping] φ_s={phi_s:.3f}, φ_sp={phi_sp:.3f}, "
                f"shaping={shaping_reward:+.3f}, reward(before clip)={reward:+.3f}"
            )
        reward = max(-1.0, min(1.0, reward))
        return finish(terminated=e.game_over)
    # ========= 觀測、遮罩建構 =========
    def _build_obs(self):
        e = self.env
        hand_counts = [0] * 34
        for t in e.players[0]['hand']:
            hand_counts[t] += 1
        in_counts = [0] * 34
        if (e.last_tile_discarded is not None and
            e.last_discard_player is not None and
            not e.tile_claimed and
            e.current_player == 0):
            id_menu = e.get_action_id_menu(player_idx=0, tsumo=False)
            if any(aid != ID_PASS for aid, _ in id_menu):
                in_counts[e.last_tile_discarded] = 1
        discard_blocks = []
        for i in range(4):
            cnt = [0] * 34
            for t in e.players[i]['discards']:
                cnt[t] += 1
            discard_blocks.extend(cnt)
        meld_blocks = []
        for i in range(4):
            cnt = [0] * 34
            for meld in e.players[i]['melds']:
                for t in meld:
                    cnt[t] += 1
            meld_blocks.extend(cnt)
        dark_counts = [0] * 34
        for meld in e.players[0]['hidden_melds']:
            for t in meld:
                dark_counts[t] += 1
        vec = hand_counts + in_counts + discard_blocks + meld_blocks + dark_counts
        assert len(vec) == 374
        return np.array(vec, dtype=np.int32)
    def _build_action_mask(self):
        e = self.env
        has_live_discard = (e.last_tile_discarded is not None and e.last_discard_player is not None and not e.tile_claimed)
        tsumo_ctx = (e.current_player == 0 and e.last_tile_drawn is not None and not has_live_discard)
        table = e.get_action_id_mask(player_idx=0, tsumo=tsumo_ctx)
        mask = np.zeros(274, dtype=bool)
        def fill_range(start, end, arr):
            mask[start:end+1] = np.array(arr, dtype=bool)
        fill_range(0, 33, table['出牌'])
        mask[34] = False
        if any(sum(table[label]) > 0 for label in ['胡', '吃(左)', '吃(中)', '吃(右)', '碰', '明槓', '加槓', '暗槓']):
            mask[34] = True
        mask[35] = bool(table['胡'][0])
        fill_range(36, 69, table['吃(左)'])
        fill_range(70, 103, table['吃(中)'])
        fill_range(104, 137, table['吃(右)'])
        fill_range(138, 171, table['碰'])
        fill_range(172, 205, table['明槓'])
        fill_range(206, 239, table['加槓'])
        fill_range(240, 273, table['暗槓'])
        return mask
    def action_masks(self):
        return self._build_action_mask()
    def close(self):
        if hasattr(self, "log_f") and self.log_f is not None:
            try:
                self.log_f.close()
            finally:
                self.log_f = None
class TerminalFocusAnnealCallback(BaseCallback):
    def __init__(self, env, total_timesteps, start=0.1, peak_at=0.6, verbose=0, force_full_focus=False):
        super().__init__(verbose)
        self.env_ref = env
        self.T = float(total_timesteps)
        self.start = float(start)
        self.peak_at = float(peak_at)
        self.force_full_focus = force_full_focus
    def _on_step(self) -> bool:
        if self.force_full_focus:
            f = 1.0
        else:
            p = self.model.num_timesteps / self.T
            if p <= self.start:
                f = 0.0
            elif p >= self.peak_at:
                f = 1.0
            else:
                x = (p - self.start) / (self.peak_at - self.start)
                f = 0.5 - 0.5 * np.cos(np.pi * x)
        try:
            for e in self.model.get_env().envs:
                e.env.set_terminal_focus(f)
        except Exception:
            pass
        return True
# ===== 便捷包裝，和舊訓練腳本相同操作介面 =====
def mask_fn(env):
    return env.action_masks()
def get_next_version(path="models"):
    os.makedirs(path, exist_ok=True)
    existing = [d for d in os.listdir(path) if d.startswith("v") and d[1:].isdigit()]
    versions = sorted([int(d[1:]) for d in existing]) if existing else []
    return f"v{(versions[-1] + 1) if versions else 1}"
def get_latest_version(path="models"):
    existing = [d for d in os.listdir(path) if d.startswith("v") and d[1:].isdigit()]
    if not existing:
        return None
    versions = sorted([int(d[1:]) for d in existing])
    return f"v{versions[-1]}"
# === 中文字體設定 ===
def set_chinese_font():
    plt.rcParams["axes.unicode_minus"] = False
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if not os.path.exists(font_path):
        print("⚙️ 安裝中文字體中 (fonts-noto-cjk)...")
        subprocess.run(
            ["apt-get", "-y", "install", "fonts-noto-cjk"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        zh_font = font_manager.FontProperties(fname=font_path)
        matplotlib.rcParams["font.family"] = zh_font.get_name()
        matplotlib.rcParams["font.sans-serif"] = [zh_font.get_name()]
        print(f"✅ 成功載入中文字體: {zh_font.get_name()}")
    else:
        print("⚠️ 找不到字體檔案，仍可能出現亂碼。")
set_chinese_font()
# ===== TrainingPlotCallback =====
class TrainingPlotCallback(BaseCallback):
    def __init__(self, plot_path="training_plot.png", verbose=1, smooth_window=300, rolling_window=3000, start_ep=1001):
        super().__init__(verbose)
        self.plot_path = plot_path
        self.smooth_window = smooth_window
        self.rolling_window = rolling_window
        self.start_ep = int(start_ep)
        self.ep_rewards = []
        self.timesteps = []
        self.policy_loss = []
        self.value_loss = []
        self.entropy = []
        self.explained_variance = []
        self.updates = []
        self.update_count = 0
        self.rons, self.tsumos, self.deals, self.lost_tsumos = [], [], [], []
    def _on_step(self) -> bool:
        infos = self.locals["infos"][0]
        if "episode" in infos:
            r = infos["episode"]["r"]
            t = self.num_timesteps
            self.ep_rewards.append(r)
            self.timesteps.append(t)
            evt = infos.get("terminal_event")
            if evt is None:
                txt = str(infos.get("end_line", ""))
                if "AI胡牌" in txt and "自摸" in txt:
                    evt = "AI_TSUMO"
                elif "AI胡牌" in txt:
                    evt = "AI_RON"
                elif ("AI放槍" in txt) or ("玩家0 放槍" in txt):
                    evt = "AI_DEAL_IN"
                elif ("電腦玩家" in txt and "自摸" in txt) or ("被自摸" in txt):
                    evt = "AI_LOST_TSUMO"
                else:
                    evt = "OTHER"
            self.tsumos.append(1 if evt == "AI_TSUMO" else 0)
            self.rons.append(1 if evt == "AI_RON" else 0)
            self.deals.append(1 if evt == "AI_DEAL_IN" else 0)
            self.lost_tsumos.append(1 if evt == "AI_LOST_TSUMO" else 0)
        return True
    def _on_rollout_end(self) -> None:
        log_dict = self.model.logger.name_to_value
        if "train/policy_gradient_loss" in log_dict:
            self.update_count += 1
            self.updates.append(self.update_count)
            self.policy_loss.append(abs(log_dict["train/policy_gradient_loss"]))
            self.value_loss.append(log_dict["train/value_loss"])
            self.entropy.append(log_dict["train/entropy_loss"])
            self.explained_variance.append(log_dict["train/explained_variance"])
    # ===== 平滑處理 =====
    def _moving_average(self, data, window_size):
        if len(data) < window_size:
            return np.array(data, dtype=float)
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    # ===== 真實累計率（避免初期暴衝） =====
    def _true_cumulative_rate(self, data, warmup=300, alpha=0.04):
        if len(data) == 0:
            return np.array([])
        data = np.array(data, dtype=float)
        result = np.zeros_like(data)
        cumsum = 0
        for i, x in enumerate(data):
            cumsum += x
            if i < warmup:
                if i == 0:
                    result[i] = x
                else:
                    result[i] = (1 - alpha) * result[i - 1] + alpha * x
            else:
                result[i] = cumsum / (i + 1)
        return result
    # ===== 畫 >= start_ep 之資料 =====
    def _trim_xy(self, x, *ys):
        x = np.asarray(x)
        mask = x >= self.start_ep
        if mask.sum() == 0:
            return None
        trimmed = [x[mask]]
        for y in ys:
            y = np.asarray(y)
            trimmed.append(y[mask])
        return trimmed
    # ===== 動態 margin =====
    def _apply_dynamic_margin(self, ax, data, min_margin=0.05):
        if len(data) == 0:
            return
        ymin, ymax = np.min(data), np.max(data)
        if ymin == ymax:
            margin = abs(ymin) * min_margin if ymin != 0 else 1.0
        else:
            margin = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - margin, ymax + margin)
    # ===== 畫圖 =====
    def _on_training_end(self) -> None:
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        # --- Reward ---
        if self.ep_rewards:
            smoothed = self._moving_average(self.ep_rewards, self.smooth_window)
            x1 = self.timesteps[-len(smoothed):]
            axes[0, 0].plot(x1, smoothed, label=f"近{self.smooth_window}局平均reward")
            axes[0, 0].set_title("Reward")
            axes[0, 0].set_xlabel("Timesteps")
            axes[0, 0].legend()
            self._apply_dynamic_margin(axes[0, 0], smoothed)
        # --- Policy Loss ---
        if self.policy_loss:
            axes[0, 1].plot(self.updates, self.policy_loss, label="策略損失 (abs)")
            axes[0, 1].set_title("策略損失 (Policy Loss)")
            axes[0, 1].set_xlabel("更新次數")
            axes[0, 1].legend()
            self._apply_dynamic_margin(axes[0, 1], self.policy_loss)
        # --- Value Loss ---
        if self.value_loss:
            axes[1, 0].plot(self.updates, self.value_loss, label="價值損失", color="orange")
            axes[1, 0].set_title("價值損失 (Value Loss)")
            axes[1, 0].set_xlabel("更新次數")
            axes[1, 0].legend()
            self._apply_dynamic_margin(axes[1, 0], self.value_loss)
        # --- Explained Variance ---
        if self.explained_variance:
            axes[1, 1].plot(self.updates, self.explained_variance, label="解釋變異 (Explained Variance)", color="green")
            axes[1, 1].set_title("解釋變異 (Explained Variance)")
            axes[1, 1].set_xlabel("更新次數")
            axes[1, 1].legend()
            self._apply_dynamic_margin(axes[1, 1], self.explained_variance)
        # --- Win Source ---
        if self.rons and self.tsumos:
            n = len(self.rons)
            x_ep = np.arange(1, n + 1)
            cum_ron = self._true_cumulative_rate(self.rons)
            cum_tsumo = self._true_cumulative_rate(self.tsumos)
            cum_winrate = cum_ron + cum_tsumo
            trimmed = self._trim_xy(x_ep, cum_ron, cum_tsumo, cum_winrate)
            if trimmed is not None:
                x_c, cum_ron_c, cum_tsumo_c, cum_winrate_c = trimmed
                axes[2, 0].plot(x_c, cum_ron_c, label="累積榮和率", color="blue", alpha=0.7)
                axes[2, 0].plot(x_c, cum_tsumo_c, label="累積自摸率", color="green", alpha=0.7)
                axes[2, 0].plot(x_c, cum_winrate_c, label="累積胡牌率", color="purple", linewidth=2)
            roll_ron = self._moving_average(self.rons, self.rolling_window)
            roll_tsumo = self._moving_average(self.tsumos, self.rolling_window)
            roll_winrate = roll_ron + roll_tsumo
            if len(roll_ron) > 0:
                x_roll_full = np.arange(n - len(roll_ron) + 1, n + 1)
                trimmed_roll = self._trim_xy(x_roll_full, roll_ron, roll_tsumo, roll_winrate)
                if trimmed_roll is not None:
                    xr, rr, rt, rw = trimmed_roll
                    axes[2, 0].plot(xr, rr, "--", label=f"近{self.rolling_window}局榮和率", color="blue")
                    axes[2, 0].plot(xr, rt, "--", label=f"近{self.rolling_window}局自摸率", color="green")
                    axes[2, 0].plot(xr, rw, "--", label=f"近{self.rolling_window}局胡牌率", color="purple", linewidth=2)
            axes[2, 0].set_title("胡牌來源")
            axes[2, 0].set_xlabel("局數")
            axes[2, 0].legend()
            dyn_data = []
            if trimmed is not None:
                dyn_data += list(cum_ron_c) + list(cum_tsumo_c) + list(cum_winrate_c)
            if len(roll_ron) > 0 and trimmed_roll is not None:
                dyn_data += list(rr) + list(rt) + list(rw)
            if dyn_data:
                self._apply_dynamic_margin(axes[2, 0], np.array(dyn_data))
            axes[2, 0].grid(alpha=0.3)
        # --- Loss Source ---
        if self.deals and self.lost_tsumos:
            n = len(self.deals)
            x_ep = np.arange(1, n + 1)
            cum_deal = self._true_cumulative_rate(self.deals)
            cum_lost = self._true_cumulative_rate(self.lost_tsumos)
            cum_lossrate = cum_deal + cum_lost
            trimmed = self._trim_xy(x_ep, cum_deal, cum_lost, cum_lossrate)
            if trimmed is not None:
                x_c, cum_deal_c, cum_lost_c, cum_lossrate_c = trimmed
                axes[2, 1].plot(x_c, cum_deal_c, label="累積放槍率", color="red", alpha=0.7)
                axes[2, 1].plot(x_c, cum_lost_c, label="累積被自摸率", color="orange", alpha=0.7)
                axes[2, 1].plot(x_c, cum_lossrate_c, label="累積失分率", color="black", linewidth=2)
            roll_deal = self._moving_average(self.deals, self.rolling_window)
            roll_lost = self._moving_average(self.lost_tsumos, self.rolling_window)
            roll_lossrate = roll_deal + roll_lost
            if len(roll_deal) > 0:
                x_roll_full = np.arange(n - len(roll_deal) + 1, n + 1)
                trimmed_roll = self._trim_xy(x_roll_full, roll_deal, roll_lost, roll_lossrate)
                if trimmed_roll is not None:
                    xr, rd, rl, rloss = trimmed_roll
                    axes[2, 1].plot(xr, rd, "--", label=f"近{self.rolling_window}局放槍率", color="red")
                    axes[2, 1].plot(xr, rl, "--", label=f"近{self.rolling_window}局被自摸率", color="orange")
                    axes[2, 1].plot(xr, rloss, "--", label=f"近{self.rolling_window}局失分率", color="black", linewidth=2)
            axes[2, 1].set_title("失分來源")
            axes[2, 1].set_xlabel("局數")
            axes[2, 1].legend()
            dyn_data = []
            if trimmed is not None:
                dyn_data += list(cum_deal_c) + list(cum_lost_c) + list(cum_lossrate_c)
            if len(roll_deal) > 0 and trimmed_roll is not None:
                dyn_data += list(rd) + list(rl) + list(rloss)
            if dyn_data:
                self._apply_dynamic_margin(axes[2, 1], np.array(dyn_data))
            axes[2, 1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close(fig)
        print(f"📊 訓練圖已儲存至 {self.plot_path}")
if __name__ == "__main__":
    env = DummyVecEnv([lambda: ActionMasker(MahjongRLTrainEnvV2(log_path="贏局分析.txt"), mask_fn)])
    print("請選擇：")
    print("[0] 測試模擬一局")
    print("[1] 訓練新版本")
    print("[2] 接續訓練最新版本")
    print("[3] 接續訓練指定版本")
    choice = input("你的選擇：").strip()
    if choice == "0":
        print("開始測試模擬一局")
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            action_mask = env.envs[0].action_masks()
            valid_actions = np.where(action_mask)[0]
            action = int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 34
            obs, rewards, dones, infos = env.step([action])
            done = bool(dones[0])
            steps += 1
            print(f"步數 {steps}, 動作 {action}, 獎勵: {float(rewards[0]):.4f}")
        env.close()
        print("測試結束")
    elif choice == "1":
        version = get_next_version()
        def lr_schedule(progress_remaining: float) -> float:
            return 3e-5 + (2e-4 - 3e-5) * progress_remaining
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr_schedule,
            n_steps=8192,
            batch_size=2048,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.015,
            vf_coef=1.0,
            max_grad_norm=0.5,
            tensorboard_log="./tensorboard/",
        )
        os.makedirs(f"models/{version}", exist_ok=True)
        total_steps = 1000000
        plot_cb = TrainingPlotCallback(plot_path=f"models/{version}/training_plot.png")
        anneal_cb = TerminalFocusAnnealCallback(env, total_steps, start=0.2, peak_at=0.7)
        model.learn(total_timesteps=total_steps, callback=[plot_cb, anneal_cb])
        model.save(f"models/{version}/model")
        print(f"✅ 已儲存為新版本：{version}")
        env.close()
    elif choice == "2":
        latest_version = get_latest_version()
        if latest_version is None:
            print("❌ 尚未有任何模型可以接續訓練。")
        else:
            model_path = f"models/{latest_version}/model"
            print(f"🔁 載入模型：{latest_version}")
            old_model = MaskablePPO.load(model_path, env=env)
            def lr_schedule(progress_remaining: float) -> float:
                return 3e-5 + (2e-4 - 3e-5) * progress_remaining
            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=lr_schedule,
                n_steps=8192,
                batch_size=2048,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.015,
                vf_coef=1.0,
                max_grad_norm=0.5,
                tensorboard_log="./tensorboard/",
            )
            model.policy.load_state_dict(old_model.policy.state_dict())
            new_version = get_next_version()
            os.makedirs(f"models/{new_version}", exist_ok=True)
            total_steps = 1000000
            plot_cb = TrainingPlotCallback(plot_path=f"models/{new_version}/training_plot.png")
            anneal_cb = TerminalFocusAnnealCallback(env, total_steps, force_full_focus=True)
            model.learn(total_timesteps=total_steps, callback=[plot_cb, anneal_cb])
            model.save(f"models/{new_version}/model")
            print(f"✅ 接續訓練完成並儲存為新版本：{new_version}")
            env.close()
    elif choice == "3":
        version = input("請輸入版本（例如 v1）：").strip()
        model_path = f"models/{version}/model"
        if not os.path.exists(model_path):
            print(f"❌ 找不到指定版本模型：{version}")
        else:
            print(f"🔁 載入模型：{version}")
            old_model = MaskablePPO.load(model_path, env=env)
            def lr_schedule(progress_remaining: float) -> float:
                return 3e-5 + (2e-4 - 3e-5) * progress_remaining
            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=lr_schedule,
                n_steps=8192,
                batch_size=2048,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.015,
                vf_coef=1.0,
                max_grad_norm=0.5,
                tensorboard_log="./tensorboard/",
            )
            model.policy.load_state_dict(old_model.policy.state_dict())
            total_steps = 1000000
            plot_cb = TrainingPlotCallback(plot_path=f"models/{version}/training_plot.png")
            anneal_cb = TerminalFocusAnnealCallback(env, total_steps, force_full_focus=True)
            model.learn(total_timesteps=total_steps, callback=[plot_cb, anneal_cb])
            new_version = get_next_version()
            os.makedirs(f"models/{new_version}", exist_ok=True)
            model.save(f"models/{new_version}/model")
            print(f"✅ 接續訓練完成並儲存為新版本：{new_version}")
            env.close()
    else:
        print("⚠️ 無效選項，請重新執行")