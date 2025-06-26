# ============================================================
# self_play_bitboard.py   4×4×4 立体四目  ― ビットボード版
# ============================================================
from pathlib import Path
from datetime import datetime
import numpy as np
import pickle, os, time, psutil, sys
import random
from game import State                       # ビットボード State
from pv_mcts import pv_mcts_scores           # モンテカルロ木探索 + NN
from dual_network import DN_OUTPUT_SIZE      # (=16)
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# -------------------- ハイパーパラメータ --------------------
SP_GAME_COUNT   = 500        # セルフプレイゲーム数（本家 25000）
SP_TEMPERATURE  = 0.8        # ボルツマン温度

# -------------------- 先手視点の価値 ------------------------
def first_player_value(ended_state: State) -> int:
    # 先手勝利:+1 / 敗北:-1 / 引分:0
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

# -------------------- 学習データ保存 ------------------------
def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True)
    path = f"./data/{now:%Y%m%d%H%M%S}.history"
    with open(path, "wb") as f:
        pickle.dump(history, f)

# -------------------- 1 ゲーム実行 -------------------------
def play(model):
    history, state = [], State()

    # self_play_bitboard.py  (while not state.is_done() の前)
    FORCED_PARALLEL = [(0,0), (3,3), (0,3), (3,0)]

    if random.random() < 0.2:                   # *20 %* の確率で平行スタート
        for (x,y) in FORCED_PARALLEL:           # 両者最善で平行を進める
            if state.is_done(): break
            state = state.place(x, y) 

    # 以降は通常の self-play

    while not state.is_done():
        # NN + MCTS で列ごとの確率 scores を取得
        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)     # shape (n_legal,)

        # --- ポリシー配列を「列 ID = 0-15」基準で作る -------------
        policies = [0.0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            col = action % 16          # idx = x + 4*y + 16*z  →  x+4y
            policies[col] = policy

        # 盤面・ポリシー保存（価値は後で付ける）
        history.append([[state.pieces, state.enemy_pieces], policies, None])

        # --- 行動選択：合法セル index から確率的に 1 つ ------------
        action = np.random.choice(state.legal_actions(), p=scores)
        state  = state.next_from_index(action)     # ← ビットボード版

    # -------- 終局後：価値ラベル付け -----------------------------
    value = first_player_value(state)
    for record in history:
        record[2] = value
        value = -value

    return history

# -------------------- セルフプレイ全体 ------------------------
def self_play():
    # best.h5 が無ければ新規作成
    if not Path("./model/best.h5").exists():
        from dual_network import dual_network
        dual_network()

    model = load_model("./model/best.h5", compile=False)


    # -------- ベンチマーク（1 ゲーム） ------------------------
    start = time.time()
    _ = play(model)
    print(f"Benchmark: {time.time()-start:.2f}s / game  •  RAM {psutil.virtual_memory().percent:.1f}%")

    # -------- 本番セルフプレイ -------------------------------
    history = []
    for i in range(SP_GAME_COUNT):
        history.extend(play(model))
        print(f"\rSelfPlay {i+1}/{SP_GAME_COUNT}", end="")
        sys.stdout.flush()
    print()

    write_data(history)          # 学習データ保存

    # -------- 後始末 ----------------------------------------
    K.clear_session()
    del model

# -------------------- 動作確認 -------------------------------
if __name__ == "__main__":
    self_play()
