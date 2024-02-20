import argparse
import chess
import chess.engine
import numpy as np
from stockfish import Stockfish


def balance(fen, args):
    engine = Stockfish(path=args.engine, depth=args.depth, parameters={"Threads": args.threads, "Hash": args.hash})
    engine.set_fen_position(fen)
    board = chess.Board(fen)
    error = float("inf")
    print(fen)
    while error > args.tolerance:
        legal = list(board.legal_moves)
        moves = engine.get_top_moves(len(legal))
        moves = [(move["Move"], move["Centipawn"] / 100) for move in moves if move["Centipawn"] is not None]
        best, val = moves[0]
        error = abs(val - args.odds)
        if error <= args.tolerance:
            break
        np.random.shuffle(moves)
        move, val = min(moves, key=lambda x: abs(x[1] - args.odds))
        error = abs(val - args.odds)
        engine.make_moves_from_current_position([move])
        board.push(chess.Move.from_uci(move))
        fen = board.fen()
        print(move)
        print(fen)
    print()
    print(engine.get_board_visual())
    print()
    print("Evaluation:", val)
    print()
    del engine
    return fen


def opposite_color_bishops(row):
    idx = 0
    for i, piece in enumerate(row):
        if piece.lower() == "b":
            idx += i
    return idx % 2 == 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="./stockfish", help="path to Stockfish engine binary (default ./stockfish)")
    parser.add_argument("--depth", type=int, default=30, help="evaluation depth (default 20)")
    parser.add_argument("--threads", type=int, default=7, help="engine CPU threads (default 7)")
    parser.add_argument("--hash", type=int, default=8192, help="engine hash size (default 8192)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default random)")
    parser.add_argument("--odds", type=float, default=0.0, help="target evaluation (default 0.0)")
    parser.add_argument("--tolerance", type=float, default=0.1, help="imbalance tolerance (default 0.1)")
    args = parser.parse_args()
    return args


def header():
    print()
    print("Funky Chess")
    # print("Version 1.0.0")
    print("By Santiago Benoit")
    print()


def main():
    header()
    args = get_args()
    print("Arguments:", args)
    engine = Stockfish(path=args.engine)
    print("Stockfish version:", engine.get_stockfish_major_version())
    del engine
    seed = args.seed
    if seed is None:
        seed = np.random.randint(1 << 32)
    print("Seed:", seed)
    np.random.seed(seed)
    print("\nGenerating position...\n")
    white = ["R", "N", "B", "Q", "K", "B", "N", "R"]
    black = ["r", "n", "b", "q", "k", "b", "n", "r"]
    shuffled = False
    while not shuffled or not opposite_color_bishops(white):
        np.random.shuffle(white)
        shuffled = True
    shuffled = False
    while not shuffled or not opposite_color_bishops(black):
        np.random.shuffle(black)
        shuffled = True
    fen = "".join(black) + "/pppppppp/8/8/8/8/PPPPPPPP/" + "".join(white) + " w - - 0 1"
    fen2 = balance(fen, args)


if __name__ == "__main__":
    main()
