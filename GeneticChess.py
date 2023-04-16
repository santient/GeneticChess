import argparse
import io
import math
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from stockfish import Stockfish, StockfishException
import time


PIECES = ["", "P", "N", "B", "R", "Q", "p", "n", "b", "r", "q", "K", "k"]
WEIGHTS = np.array([32, 8, 2, 2, 2, 1, 8, 2, 2, 2, 1])
INDICES = np.concatenate([[i] * w for i, w in enumerate(WEIGHTS)])
PROBS = np.array([16/31, 8/31, 2/31, 2/31, 2/31, 1/31])
assert math.isclose(PROBS.sum(), 1)
args = None
seed = None
dna = r"""
                                 |\_
O       o O       o O       o   /  .\_
| O   o | | O   o | | O   o |  |   ___)
| | O | | | | O | | | | O | |  |    \
| o   O | | o   O | | o   O |  |  =  |
o       O o       O o       O  /_____\
                              [_______]
""" # Composed from ASCII art archive. Credit to Brent James Benton for the knight.


class ChessProblem(ElementwiseProblem):

    def __init__(self):
        xl = np.zeros(68)
        xu = np.full(68, 61)
        xu[64:] = 7
        super().__init__(n_var=68, n_obj=4, xl=xl, xu=xu, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        board, kings = vector_to_board(x)
        f1 = eval_pawns(board)
        remove_backrank_pawns(board)
        f2 = eval_points(board)
        f3 = eval_empty(board)
        f4 = eval_kings(kings) + eval_proximity(board, kings) + eval_offside(board)
        out["F"] = [f1, f2, f3, f4]


def eval_empty(board):
    return abs((board == 0).sum() - 32)


def eval_pawns(board):
    back = count_backrank_pawns(board)
    w = (board[1:7] == 1).sum()
    b = (board[1:7] == 6).sum()
    wc = sum(abs((board[1:7, i] == 1).sum() - 1) for i in range(8))
    bc = sum(abs((board[1:7, i] == 6).sum() - 1) for i in range(8))
    return back + abs(w - 8) + abs(b - 8) + wc + bc


def eval_points(board):
    P = (board == 1).sum()
    N = (board == 2).sum()
    B = (board == 3).sum()
    R = (board == 4).sum()
    Q = (board == 5).sum()
    p = (board == 6).sum()
    n = (board == 7).sum()
    b = (board == 8).sum()
    r = (board == 9).sum()
    q = (board == 10).sum()
    w = P * 1 + N * 3 + B * 3 + R * 5 + Q * 9
    b = p * 1 + n * 3 + b * 3 + r * 5 + q * 9
    return abs(w - 40) + abs(b - 40)


def eval_kings(kings):
    return kings[1, 0] + 7 - kings[0, 0]


def eval_proximity(board, kings):
    total = 0
    count = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            cell = board[i, j]
            if cell > 0 and cell <= 10:
                count += 1
                if cell <= 5:
                    total += abs(i - kings[0, 0]) + abs(j - kings[0, 1])
                else:
                    total += abs(i - kings[1, 0]) + abs(j - kings[1, 1])
    if count > 0:
        return abs(total / count - 3)
    else:
        return 100


def eval_offside(board):
    return ((board[:4] > 0) * (board[:4] <= 5)).sum() + ((board[4:] > 5) * (board[4:] <= 10)).sum()


def vector_to_board(x):
    board = INDICES[x[:64]].reshape((8, 8))
    kings = x[64:].reshape((2, 2))
    board[kings[0, 0], kings[0, 1]] = 11
    board[kings[1, 0], kings[1, 1]] = 12
    return board, kings


def board_to_fen(board):
    with io.StringIO() as s:
        for row in board:
            empty = 0
            for cell in row:
                if cell > 0:
                    if empty > 0:
                        s.write(str(empty))
                        empty = 0
                    s.write(PIECES[cell])
                else:
                    empty += 1
            if empty > 0:
                s.write(str(empty))
            s.write('/')
        s.seek(s.tell() - 1)
        s.write(' w - - 0 1')
        return s.getvalue()


def backrank_pawns(fen):
    rows = fen.split(" ")[0].split("/")
    return "p" in rows[0].lower() or "p" in rows[7].lower()


def count_backrank_pawns(board):
    count = 0
    for i in range(8):
        if PIECES[board[0, i]].lower() == "p":
            count += 1
        if PIECES[board[7, i]].lower() == "p":
            count += 1
    return count


def remove_backrank_pawns(board):
    for i in range(8):
        if PIECES[board[0, i]].lower() == "p":
            board[0, i] = 0
        if PIECES[board[7, i]].lower() == "p":
            board[7, i] = 0


def evaluate(fen, final=False):
    depth = args.final_depth if final else args.depth
    if backrank_pawns(fen):
        return None, None
    engine = Stockfish(path=args.stockfish, depth=depth)
    if not engine.is_fen_valid(fen):
        return None, None
    try:
        engine.set_fen_position(fen)
        if final:
            val = engine.get_evaluation()
            if val["type"] != "cp":
                return None, None
            val = val["value"] / 100
        else:
            if args.stable:
                error = np.empty(depth)
                for d in range(depth):
                    engine.set_depth(d + 1)
                    val = engine.get_evaluation()
                    if val["type"] != "cp":
                        return None, None
                    if val["value"] == 0:
                        return None, None
                    val = val["value"] / 100
                    error[d] = abs(val - args.odds)
                error = error.mean()
                error = round(error, 4)
            else:
                val = engine.get_evaluation()
                if val["type"] != "cp":
                    return None, None
                if val["value"] == 0:
                    return None, None
                val = val["value"] / 100
                error = abs(val - args.odds)
    except StockfishException:
        return None, None
    if final:
        vis = engine.get_board_visual()
        return val, vis
    else:
        return val, error


def evolve_structure():
    problem = ChessProblem()
    algorithm = NSGA2(pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True)
    res = minimize(problem,
        algorithm,
        seed=seed,
        verbose=True)
    return res


def mutate_balance(board, kings, fen, dup, rate=1):
    val = None
    while val is None:
        board_new = board.copy()
        for i in range(rate):
            w_src_i, w_src_j = kings[0]
            while w_src_i == kings[0, 0] and w_src_j == kings[0, 1]:
                w_src_i = np.random.randint(4, 8)
                w_src_j = np.random.randint(8)
            b_src_i, b_src_j = kings[1]
            while b_src_i == kings[1, 0] and b_src_j == kings[1, 1]:
                b_src_i = np.random.randint(4)
                b_src_j = np.random.randint(8)
            w_tgt = np.random.choice(6, p=PROBS)
            b_tgt = np.random.choice(6, p=PROBS)
            if b_tgt > 0:
                b_tgt += 5
            board_new[w_src_i, w_src_j] = w_tgt
            board_new[b_src_i, b_src_j] = b_tgt
        remove_backrank_pawns(board_new)
        fen_new = board_to_fen(board_new)
        if fen_new not in dup:
            dup.add(fen_new)
            val, error = evaluate(fen_new)
    return board_new, kings, fen_new, val, error


def evolve_balance(res):
    pop = []
    dup = set()
    count = 0
    for x in res.X:
        board, kings = vector_to_board(x)
        remove_backrank_pawns(board)
        fen = board_to_fen(board)
        if fen not in dup:
            dup.add(fen)
            val, error = evaluate(fen)
            if val is None:
                error = np.inf
            pop.append((board, kings, fen, val, error))
            count += 1
            if count == 5:
                break
    pop = list(sorted(pop, key=lambda x: (x[4], np.random.rand())))
    best = pop[0]
    board, kings, fen, val, error = best
    gen = 0
    print(f"[Generation {gen}] [Evaluation {val}] [Error {error}] [FEN {fen}]")
    while error > args.error:
        gen += 1
        offspring = []
        for i, (board, kings, fen, val, error) in enumerate(pop):
            for j in range(5 - i):
                board_new, kings_new, fen_new, val_new, error_new = mutate_balance(board, kings, fen, dup)
                offspring.append((board_new, kings_new, fen_new, val_new, error_new))
        pop.extend(offspring)
        pop = list(sorted(pop, key=lambda x: (x[4], np.random.rand())))[:5]
        best = pop[0]
        board, kings, fen, val, error = best
        print(f"[Generation {gen}] [Evaluation {val}] [Error {error}] [FEN {fen}]")
    return fen


def results(fen):
    val, vis = evaluate(fen, final=True)
    assert val is not None
    print("RESULTS\n")
    print(vis)
    print("Evaluation:", val)
    print("FEN:", fen)
    print()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", type=str, default="./stockfish", help="path to stockfish binary (default ./stockfish)")
    parser.add_argument("--depth", type=int, default=20, help="balance evaluation depth (default 20)")
    parser.add_argument("--final-depth", type=int, default=30, help="final evaluation depth (default 30)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default random)")
    parser.add_argument("--odds", type=float, default=0.0, help="target evaluation (default 0.0)")
    parser.add_argument("--error", type=float, default=0.2, help="target error margin for evaluation (default 0.2)")
    parser.add_argument("--stable", action="store_true", help="search for more \"stable\" balance (might take a while!)")
    args = parser.parse_args()
    return args


def header():
    print(dna)
    print("Genetic Chess")
    # print("Version 1.0.0")
    print("By Santiago Benoit")
    print()


def main():
    global args
    global seed
    header()
    args = get_args()
    print("Arguments:", args)
    engine = Stockfish(path=args.stockfish, depth=args.depth)
    print("Stockfish version:", engine.get_stockfish_major_version())
    del engine
    seed = args.seed
    if seed is None:
        seed = np.random.randint(2**32)
    print("Random seed:", seed)
    np.random.seed(seed)
    time.sleep(1)
    print("\nEvolving structure...\n")
    res = evolve_structure()
    print("\nEvolving balance...\n")
    fen = evolve_balance(res)
    print("\nFinal analysis...\n")
    results(fen)


if __name__ == "__main__":
    main()
