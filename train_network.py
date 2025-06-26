# =====================================================
# train_network_bitboard.py
#   - ビットボード履歴で Dual Net を再学習
# =====================================================
from pathlib import Path
import numpy as np
import pickle, os, sys
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras import backend as K

from dual_network import DN_INPUT_SHAPE         # (4,4,8)

RN_EPOCHS = 40          # エポック数
SIZE      = DN_INPUT_SHAPE[0]                   # 4

# -----------------------------------------------------
# 盤面 64bit → (4,4,8) テンソル  *ベクトル化版*
# -----------------------------------------------------
def bitboards_to_tensor_batch(pieces_arr, enemy_arr):
    """
    pieces_arr, enemy_arr : (N,) uint64 配列
    return                : (N,4,4,8) float32
    """
    N = len(pieces_arr)
    # 64bit → (N,64) bool
    mine_bits  = np.unpackbits(pieces_arr.view(np.uint8)
                               .reshape(N, 8),  # little-endian
                               bitorder='little')
    yours_bits = np.unpackbits(enemy_arr.view(np.uint8)
                               .reshape(N, 8),
                               bitorder='little')

    # reshape → (N,4,4,4)  (z,y,x)
    mine  = mine_bits.reshape(N, SIZE, SIZE, SIZE)
    yours = yours_bits.reshape(N, SIZE, SIZE, SIZE)

    # 軸並べ替え → (N,y,x,z) = (N,4,4,4)
    mine  = mine.transpose(0, 2, 3, 1)          # ch0-3
    yours = yours.transpose(0, 2, 3, 1)         # ch4-7

    planes = np.zeros((N, SIZE, SIZE, SIZE * 2), dtype=np.float32)
    planes[..., :SIZE]  = mine
    planes[..., SIZE:]  = yours
    return planes

# -----------------------------------------------------
# リプレイ読み込み
# -----------------------------------------------------
"""
def load_history():
    hist_files = sorted(Path("./data").glob("*.history"))
    if not hist_files:
        print("❌  data/*.history がありません。self_play を回してください。")
        sys.exit()
    with hist_files[-1].open("rb") as f:
        return pickle.load(f)
"""
"""
def load_history(buffer_size: int = 80000):
    files = sorted(Path("./data").glob("*.history"))
    merged = []
    for p in files[::-1]:              # 新→旧 に走査
        with p.open("rb") as f:
            merged.extend(pickle.load(f))
        if len(merged) >= buffer_size:
            break
    # ランダムにシャッフル & バッファ上限でトリム
    rng = np.random.default_rng(42)
    rng.shuffle(merged)
    return merged[:buffer_size]
"""
def load_history(buffer_size: int = 80_000):
    # --- ① 自己対戦履歴を最新→過去へ読み込み ---
    files = sorted(Path("./data").glob("*.history"), reverse=True)
    merged = []
    for p in files:
        merged.extend(pickle.load(p.open("rb")))
        if len(merged) >= buffer_size:
            break

    # --- ② 追加教師（forced_history）があれば挿入 ---
    forced_path = Path("./data/forced_parallel.history")
    if forced_path.exists():
        forced_data = pickle.load(forced_path.open("rb"))
        merged.extend(forced_data)

    # --- ③ ランダムシャッフル ---
    rng = np.random.default_rng()      # 乱数シード固定しない方が多様化
    rng.shuffle(merged)

    # --- ④ 上限でトリムして返す ---
    return merged[:buffer_size]



# -----------------------------------------------------
# 学習ループ
# -----------------------------------------------------
def train_network():
    history = load_history()

    pieces_int, enemy_int = zip(*[h[0] for h in history])
    y_policies, y_values  = zip(*[(h[1], h[2]) for h in history])

    pieces_arr = np.frombuffer(np.array(pieces_int, dtype=np.uint64), dtype=np.uint64)
    enemy_arr  = np.frombuffer(np.array(enemy_int,  dtype=np.uint64), dtype=np.uint64)

    # --- 入力テンソル (N,4,4,8)
    xs = bitboards_to_tensor_batch(pieces_arr, enemy_arr)

    # --- チェック①：policy 正規化と value 分布 ---
    import numpy as _np
    # policy がすべて 1 に正規化されているか
    print("DEBUG policy_sum≈1 ?", _np.allclose(_np.sum(y_policies, axis=1), 1.0))
    # value ラベルに +1, -1, 0 がちゃんと混ざっているか
    print("DEBUG value distribution", _np.unique(y_values, return_counts=True))

    nonzero_ratio = (xs.sum(axis=(1,2,3)) > 0).mean()
    avg_bits      = xs.sum() / len(xs)

    print(f"DEBUG  non-zero サンプル率 = {nonzero_ratio:.3f}")
    print(f"DEBUG  1サンプル当たり立っているビット数 = {avg_bits:.1f}")


    # --- 出力ラベル
    y_policies = np.asarray(y_policies, dtype=np.float32)
    y_values   = np.asarray(y_values,   dtype=np.float32)

    # --- モデル取得
    if not Path("./model/best.h5").exists():
        from dual_network import dual_network
        dual_network()
    model = load_model("./model/best.h5", compile=False)

    model.compile(
        loss      = ["categorical_crossentropy", "mse"],
        loss_weights=[1.0, 0.3],     # ← ここを追加
        optimizer = "adam"
    )

    # --- 学習率スケジューラ
    #def step_decay(epoch):
        #return 0.00025 if epoch >= 80 else 0.0005 if epoch >= 50 else 0.0003 #エポック数を100→20に変更

    #def step_decay(epoch):
     #   return 0.0001 if epoch>=15 else 0.0002 if epoch>=8 else 0.0003
    #lr_sched = LearningRateScheduler(step_decay, verbose=0)

    #def step_decay(epoch):
     # if epoch < 10:  return 5e-4   # ← 最初を少し下げる
     # if epoch < 20:  return 2.5e-4
     # return 1e-4

    def step_decay(epoch):
        """epoch に応じて学習率を 4 段階で減衰させる"""
        if   epoch < 10: return 5e-4        # 0– 9 epoch
        elif epoch < 20: return 2.5e-4      # 10–19 epoch
        elif epoch < 30: return 1e-4        # 20–29 epoch
        else:            return 5e-5        # 30–39 epoch

    lr_sched = LearningRateScheduler(step_decay, verbose=0)


    # --- 進捗表示
    print_cb = LambdaCallback(
        on_epoch_begin=lambda epoch, logs:
            print(f"\rTrain {epoch+1}/{RN_EPOCHS}", end="")
    )
    # --- 学習実行（履歴を受け取る） ---
    history_obj = model.fit(
        xs, [y_policies, y_values],
        batch_size = 128,
        epochs     = RN_EPOCHS,
        shuffle    = True,
        verbose    = 0,
        callbacks  = [lr_sched, print_cb]
    )
    print()  # 改行

    # --- チェック②：学習損失の推移 ---
    losses = history_obj.history["loss"]
    print(f"DEBUG loss: start={losses[0]:.3f}, end={losses[-1]:.3f}")

    # --- 保存
    os.makedirs("./model", exist_ok=True)
    model.save("./model/latest.h5")
    K.clear_session()

# -----------------------------------------------------
# 動作確認
# -----------------------------------------------------
if __name__ == "__main__":
    train_network()
