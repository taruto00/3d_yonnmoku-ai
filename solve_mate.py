# ==================== solve_mate.py ==========================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import argparse, random, math, numpy as np

# ---------- 盤面ユーティリティ --------------------------------
SIZE = 4
def idx(x:int, y:int, z:int)->int: return x + y*SIZE + z*SIZE*SIZE
def idx_to_xyz(i:int)->Tuple[int,int,int]:
    return i % SIZE, (i//4) % SIZE, i//16
def BIT(x:int,y:int,z:int)->int:   return 1 << idx(x,y,z)

# ---------- 76 本のラインを 1 回生成 ---------------------------
def _make_lines76()->List[int]:
    r,L = range(SIZE),[]
    for y in r:
        for z in r: L.append(sum(BIT(x,y,z) for x in r))
    for x in r:
        for z in r: L.append(sum(BIT(x,y,z) for y in r))
    for x in r:
        for y in r: L.append(sum(BIT(x,y,z) for z in r))
    for z in r:
        L += [sum(BIT(i,i,z) for i in r), sum(BIT(SIZE-1-i,i,z) for i in r)]
    for y in r:
        L += [sum(BIT(i,y,i) for i in r), sum(BIT(SIZE-1-i,y,i) for i in r)]
    for x in r:
        L += [sum(BIT(x,i,i) for i in r), sum(BIT(x,SIZE-1-i,i) for i in r)]
    L += [sum(BIT(i,i,i) for i in r),
          sum(BIT(SIZE-1-i,i,i) for i in r),
          sum(BIT(i,SIZE-1-i,i) for i in r),
          sum(BIT(SIZE-1-i,SIZE-1-i,i) for i in r)]
    return L
LINES76 = _make_lines76()


# --------------------------------------------------------------------
@dataclass
class State:
    pieces:        int = 0  # 現手番の石
    enemy_pieces:  int = 0  # 相手の石

    # --- 追加ここから -------------------------------------------------
    @property
    def my_bits(self) -> int:      # 自分石（現手番）
        return self.pieces

    @property
    def opp_bits(self) -> int:     # 相手石
        return self.enemy_pieces
    # --- 追加ここまで -------------------------------------------------

    # ---- 基本 ----
    @staticmethod
    def pop(bits:int)->int: return bits.bit_count()
    def turn(self)->int:    return self.pop(self.pieces|self.enemy_pieces)+1
    def is_first_player(self)->bool: return self.pop(self.pieces)==self.pop(self.enemy_pieces)

    # ---- 勝敗 ----
    def _is_win(self, bits:int)->bool:
        return any((bits & line)==line for line in LINES76)
    def is_win (self)->bool: return self._is_win(self.pieces)
    def is_lose(self)->bool: return self._is_win(self.enemy_pieces)
    def is_draw(self)->bool: return (self.pieces|self.enemy_pieces)==ALL_MASK
    def is_done (self)->bool: return self.is_win() or self.is_lose() or self.is_draw()

    # ---- 合法手（重力）----
    def legal_actions(self)->List[int]:
        acts, occ = [], self.pieces|self.enemy_pieces
        for x in range(SIZE):
            for y in range(SIZE):
                for z in range(SIZE):          # 下から探索
                    i = idx(x,y,z)
                    if not (occ>>i & 1):
                        acts.append(i); break
        return acts

    # ---- 手を打つ ----

    def next_from_index(self, i) -> "State":
        i = int(i)                     # ★ここを追加（numpy.int64 → int）
        assert not ((self.pieces | self.enemy_pieces) >> i & 1), "occupied"
        return State(self.enemy_pieces, self.pieces | (1 << i))


    def place(self, x:int, y:int)->"State":
        for z in range(SIZE):
            i = idx(x,y,z)
            if not ((self.pieces|self.enemy_pieces)>>i & 1):
                return self.next_from_index(i)
        raise ValueError("column full")

    # ---- 表示 ----
    def __str__(self)->str:
        ox = ('o','x') if self.is_first_player() else ('x','o')
        head = "   " + "   ".join(f"z={z}" for z in range(SIZE))
        rows=[head]
        for y in range(SIZE-1,-1,-1):
            line=[]
            for z in range(SIZE):
                s=[]
                for x in range(SIZE):
                    b=BIT(x,y,z)
                    s.append(ox[0] if self.pieces & b else
                             ox[1] if self.enemy_pieces & b else '-')
                line.append("".join(s))
            rows.append("   ".join(line))
        return "\n".join(rows)

    # ---- ライン評価 (+自分, -相手) ----
    def line_counts(self)->List[int]:
        out=[]
        for line in LINES76:
            if (self.pieces & line) and (self.enemy_pieces & line): out.append(0)
            elif self.pieces & line:  out.append(self.pop(self.pieces & line))
            else:                     out.append(-self.pop(self.enemy_pieces & line))
        return out


# ---------- 脅威列挙（3 連＋空 1 / 2 連＋空 2 など） -----------
def collect_threats(my:int, opp:int)->List[int]:
    occ,out = my|opp,[]
    for line in LINES76:
        if line & opp or (line & my).bit_count()!=3: continue
        empty = line & ~occ
        if empty:
            i = empty.bit_length()-1
            z = i//16
            if z==0 or (occ>>(i-16) & 1):
                out.append(i)
    return list(set(out))

def collect_pre_threats(my:int, opp:int)->List[int]:
    occ,out = my|opp,[]
    for line in LINES76:
        if line & opp or (line & my).bit_count()!=2: continue
        empties = line & ~occ
        while empties:
            b = empties & -empties
            i = b.bit_length()-1
            z = i//16
            if z==0 or (occ>>(i-16) & 1): out.append(i)
            empties ^= b
    return list(set(dict.fromkeys(out)))         # 重複除去

def collect_float_pre_threats(my:int, opp:int)->List[int]:
    occ,out = my|opp,[]
    k = 0
    for line in LINES76:
        if line & opp or (line & my).bit_count()!=3: continue
        k |= line & ~occ
    while k:
        b = k & -k
        top = b.bit_length()-1
        z   = top//16
        k  ^= b
        if z==0: continue
        below = top-16
        if (occ>>below)&1: continue
        if z==1 or (occ>>(below-16)&1):
            out.append(below)
    return list(dict.fromkeys(out))

# ---------- 2-ply 簡易詰み判定 --------------------------------
def my_threat_exists(my:int, opp:int)->bool:
    return bool(collect_threats(my,opp))

def find_forced_block(state:"State")->int|None:
    opp_thr = collect_threats(state.enemy_pieces,state.pieces)
    return opp_thr[0] if len(opp_thr)==1 else None

def tsumi(state:"State", depth:int)->int:
    my,opp = state.pieces, state.enemy_pieces
    if my_threat_exists(my,opp): return +1
    block = find_forced_block(state)
    if block is not None: return -tsumi(state.next_from_index(block),2)
    if collect_threats(opp,my):   return -1
    if depth==1: return 0
    cand = collect_pre_threats(my,opp)+collect_float_pre_threats(my,opp)
    for mv in cand:
        if tsumi(state.next_from_index(mv),1)==-1: return +1
    return 0

def mate_in_two_sequence(state:"State")->List[int]|None:
    if tsumi(state,2)!=1: return None
    cand = collect_pre_threats(state.pieces,state.enemy_pieces)+\
           collect_float_pre_threats(state.pieces,state.enemy_pieces)
    for mv1 in cand:
        s1 = state.next_from_index(mv1)
        blk = find_forced_block(s1)
        if blk is None: continue
        s2 = s1.next_from_index(blk)
        wins = collect_threats(s2.pieces,s2.enemy_pieces)
        if wins: return [mv1, blk, wins[0]]
    return None

# -------------------------------------------------------------
# 強制ライン探索（深さ ply_limit まで）
#   戻り値:  (verdict, path)
#       verdict : +1 先手必勝 / -1 先手必敗 / 0 未確定
#       path    : セル index のリスト（先手→後手→… と交互に続く）
#                 verdict が ±1 のときのみ「詰みまでの 1 変化」を返す
# -------------------------------------------------------------
def proof_search(state: State, ply_limit: int = 32) -> tuple[int, list[int]]:
    cache: dict[Tuple[int,int,int], tuple[int,list[int]]] = {}
    # key = (my_bits, opp_bits, ply_left)

    def dfs(s: State, depth: int, turn: int) -> tuple[int, list[int]]:
        k = (s.pieces, s.enemy_pieces, depth)
        if k in cache: return cache[k]

        # ---- 終端判定 ----
        if s.is_win():   return (+1*turn, [])
        if s.is_lose():  return (-1*turn, [])
        if depth == 0:   return (0, [])         # 読み切り打ち切り

        # ---- 1 手詰みチェック ----
        my_th  = collect_threats(s.pieces, s.enemy_pieces)
        opp_th = collect_threats(s.enemy_pieces, s.pieces)

        # ◆ 先手番 -------------------------------------------------
        if turn == +1:
            # ① 先手の即勝ち
            if my_th:
                moves = my_th
                #return (+1, [my_th[0]])
            # ② 先手が作れる脅威手を全て試す
            else:
                moves = (
                    collect_pre_threats(s.pieces, s.enemy_pieces) +
                    collect_float_pre_threats(s.pieces, s.enemy_pieces)
                )
            ans = (0, [])
            for mv in moves:
                verdict, line = dfs(s.next_from_index(mv), depth-1, -turn)
                if verdict == +1:
                    if ans[0] == 0 or len(ans[1]) > len(line) + 1:
                        ans = (+1, [mv] + line)
            cache[k] = ans;  return cache[k]   # 未確定

        # ◆ 後手番 -------------------------------------------------
        else:
            # ① 後手の唯一受けを強制
            #if len(opp_th) >= 2:      # ダブルリーチ → 先手勝ち
            #    return (+1, [])
            if len(opp_th) >= 1:
                mv = opp_th[0]
                verdict, line = dfs(s.next_from_index(mv), depth-1, -turn)
                cache[k] = (verdict, [mv] + line);  return cache[k]
            cache[k] = (0, []);  return cache[k]

            # ② 後手が自由着手できるなら，全手で“先手勝ち”を証明する必要
            all_fail = True
            best_line = []
            for mv in s.legal_actions():
                verdict, line = dfs(s.next_from_index(mv), depth-1, -turn)
                if verdict != +1:
                    all_fail = False; break
                best_line = [mv] + line        # Keep 1 variation

            if all_fail:
                cache[k] = (+1, best_line); return cache[k]
            cache[k] = (0, []);  return cache[k]

    return dfs(state, ply_limit, +1)


# ---------- 例題局面 ------------------------------------------
"""
o_list=[(0,0,0),(0,2,0),(0,3,0),(1,1,0),(2,2,0),(3,2,0),
        (1,1,1),(1,2,1),(2,2,1),(2,1,3)]
x_list=[(0,1,0),(1,2,0),(2,1,0),(3,0,0),(3,1,0),(3,3,0),
        (2,1,1),(1,1,2),(2,1,2),(2,2,2)]
"""
o_list = [(0,0,0), (0,3,0), (1,0,0), (1,2,0), (2,1,0), (3,1,0), (3,3,0), (3,3,1), (1,0,1), (1,2,1), (2,2,1), (3,1,1), (1,1,2), (3,2,2)]   # 先手＝o
x_list = [(1,1,0), (1,3,0), (2,0,0), (2,2,0), (2,3,0), (3,0,0), (3,2,0), (1,1,1), (2,1,1), (3,2,1), (2,1,2), (2,2,2), (2,1,3), (2,2,3)]   # 後手＝x

def bits(lst): return sum(1<<idx(x,y,z) for x,y,z in lst)
example_state = State(bits(o_list), bits(x_list))
"""
if __name__ == "__main__":
    s = example_state
    print(s, "\n")

    verdict = tsumi(s, 2)
    print("評価 (tsumi depth=2):", verdict)   # +1 / -1 / 0

    # 先手必勝なら必ず手順を表示
    if verdict == 1:
        seq = mate_in_two_sequence(s)
        if seq:
            seq_xyz = [idx_to_xyz(i) for i in seq]
            print("\n必勝手順 (o 先手):")
            print(f"  o: {seq_xyz[0]}")
            print(f"  x: {seq_xyz[1]}")
            print(f"  o: {seq_xyz[2]}  ← 勝ち")
        else:
            print("必勝だが手順抽出に失敗しました。")
    elif verdict == -1:
        print("現手番は Mate-in-2 で詰まされています。")
    else:
        print("未確定（深読みが必要）。")
"""
# ---------- メイン ---------------------------------------------
if __name__ == "__main__":
    s = example_state                     # ← 自由に差し替え可
    print(s, "\n")

    verdict, pv = proof_search(s, ply_limit=64) #ここで読む長さを調整(ply_limit=64なら32手詰めまで探索する)
    print("評価 (deep):", verdict)        # +1 / -1 / 0

    if verdict in (+1, -1):
        pv_xyz = [idx_to_xyz(i) for i in pv]
        side   = ('o','x')* (len(pv)//2+2)
        print("\n詰み手順:")
        for turn,(mv_xyz,pl) in enumerate(zip(pv_xyz, side)):
            print(f"{turn + 1} ({pl}): {mv_xyz}")
        print("  … 詰み")
    else:
        print("未確定（探索深さを伸ばすか，評価関数に頼る）。")

