# =============================================================
# evaluate_network_bitboard.py
#   - latest.h5 と best.h5 を対戦させ平均ポイントを算出
#   - 平均 > 0.5 なら best.h5 を更新
# =============================================================
from pathlib import Path
from shutil import copy
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

import hashlib

from game       import State                 # ビットボード State
from pv_mcts    import pv_mcts_action        # ビットボード対応版

def make_parallel_state():
    """0 0 → 3 3 → 0 3 まで進めた局面を返す"""
    s = State()
    for (x, y) in [(0,0), (3,3), (0,3)]:   # 3 手まで進める
        s = s.place(x, y)                  # place が無ければ idx→next_from_index で
    return s

# -------------------- MD5 チェック用関数 --------------------
def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:8]

# -------------------- パラメータ --------------------
EN_GAME_COUNT  = 200     # 評価ゲーム数 (本家400)
EN_TEMPERATURE = 0.05    # ボルツマン温度

# -------------------- ポイント計算 -------------------
def first_player_point(ended_state: State) -> float:
    if ended_state.is_lose():
        return 0.0 if ended_state.is_first_player() else 1.0
    return 0.5   # 引き分け

# -------------------- 1 ゲーム実行 -------------------
def play(next_actions):
    """next_actions = (先手AI, 後手AI)"""
    state = State()
    while not state.is_done():
        action_fn = next_actions[0] if state.is_first_player() else next_actions[1]
        action    = action_fn(state)                 # セル index
        state     = state.next_from_index(action)    # ★ ビットボード用
    return first_player_point(state)

# -------------------- best プレイヤー更新 -------------
def update_best_player():
    copy("./model/latest.h5", "./model/best.h5")
    print("🏆  best.h5 を更新しました")

# -------------------- ネットワーク評価 ----------------
def evaluate_network():

    # --- モデルパス ---
    latest_path = "./model/latest.h5"
    best_path   = "./model/best.h5"

    # ★★ MD5 を出力して同一かチェック ★★
    print(f"DEBUG md5 latest: {md5(latest_path)}")
    print(f"DEBUG md5   best: {md5(best_path)}")

    # --- モデル読み込み (compile=False で警告抑止) ---
    model_latest = load_model("./model/latest.h5", compile=False)
    model_best   = load_model("./model/best.h5",   compile=False)

    # --- 行動関数 ---
    next_latest = pv_mcts_action(model_latest, EN_TEMPERATURE)
    next_best   = pv_mcts_action(model_best,   EN_TEMPERATURE)
    pair = (next_latest, next_best)

    # --- 対戦 ---
    total = 0.0
    for i in range(EN_GAME_COUNT):
        if i % 2 == 0:                  # 偶数局：latest 先手
            total += play(pair)
        else:                           # 奇数局：best 先手
            total += 1.0 - play(pair[::-1])
        print(f"\rEvaluate {i+1}/{EN_GAME_COUNT}", end="")
    print()

    avg = total / EN_GAME_COUNT
    
    # --- 対戦が終わり avg を求めた直後に追加 -----------------
    print("Average Point (先手視点):", avg)
    print("後手側勝率 (1-avg)      :", 1 - avg)

    # 追加: 平行 4 手後の value をモデル2つで確認
    test_state = make_parallel_state()
    test_tensor = encode_state(test_state)          # pv_mcts で使っているエンコード関数
    v_latest = float(model_latest.predict(test_tensor, verbose=0)[1][0])
    v_best   = float(model_best  .predict(test_tensor, verbose=0)[1][0])
    print(f"value after 4-move parallel  latest={v_latest:+.3f}  best={v_best:+.3f}")

    # --- 後始末 ---
    K.clear_session(); del model_latest, model_best

    # --- best 更新判定 ---
    if avg < 0.48:                        # 後手必勝のため閾値を0.48未満に変更
        update_best_player()
        return True
    return False

# -------------------- 動作確認 ------------------------
if __name__ == "__main__":
    evaluate_network()
