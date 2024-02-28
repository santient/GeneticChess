# Nice Chess: Balanced Subset of 2880^2 / 960^2 Chess
# By Santiago Benoit


import argparse
import chess
import numpy as np
import secrets
from stockfish import Stockfish


# version = "1.0.0"


def evaluation_passed(fen, args):
    engine = Stockfish(path=args.engine, depth=args.depth, parameters={"Threads": args.threads, "Hash": args.hash, "UCI_Chess960": str(args.castling).lower()})
    engine.set_fen_position(fen)
    for depth in range(1, args.depth + 1):
        engine.set_depth(depth)
        moves = engine.get_top_moves(args.lines)
        vals = [move["Centipawn"] for move in moves]
        if any(v is None for v in vals):
            del engine
            return False
        vals = [v / 100 for v in vals]
        errs = [abs(v - args.odds) for v in vals]
        if depth == args.depth and errs[0] > args.tolerance:
            del engine
            return False
        if any(e > 2 * args.tolerance for e in errs):
            del engine
            return False
    del engine
    return True


def opposite_color_bishops(row):
    idx = 0
    for i, piece in enumerate(row):
        if piece.lower() == "b":
            idx += i
    return idx % 2 == 1


def king_between_rooks(row):
    between = False
    for piece in row:
        if piece.lower() == "r":
            between = not between
        elif piece.lower() == "k":
            if between:
                return True
            else:
                return False
    return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="./stockfish", help="path to Stockfish engine binary (default ./stockfish)")
    parser.add_argument("--depth", type=int, default=20, help="evaluation depth (default 20)")
    parser.add_argument("--lines", type=int, default=5, help="lines to evaluate (default 5)")
    parser.add_argument("--threads", type=int, default=1, help="engine CPU threads (default 1)")
    parser.add_argument("--hash", type=int, default=1024, help="engine hash size (default 1024)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (random by default)")
    parser.add_argument("--odds", type=float, default=0.0, help="target evaluation (default 0.0)")
    parser.add_argument("--tolerance", type=float, default=0.2, help="imbalance tolerance (default 0.25)")
    parser.add_argument("--castling", action="store_true", help="enable castling (disabled by default)")
    parser.add_argument("--out", type=str, default=None, help="output FEN to specified file")
    args = parser.parse_args()
    return args


def header():
    print()
    print("Nice Chess")
    # print(f"Version {version}")
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
        seed = secrets.randbits(32)
    print("Random seed:", seed)
    np.random.seed(seed)
    if args.threads > 1:
        print("Warning: random seed consistency is not guaranteed with threads > 1.")
    print("\nGenerating position...\n")
    memo = set()
    white = ["R", "N", "B", "Q", "K", "B", "N", "R"]
    black = ["r", "n", "b", "q", "k", "b", "n", "r"]
    np.random.shuffle(white)
    np.random.shuffle(black)
    if args.castling:
        fen = "".join(black) + "/pppppppp/8/8/8/8/PPPPPPPP/" + "".join(white) + " w KQkq - 0 1"
    else:
        fen = "".join(black) + "/pppppppp/8/8/8/8/PPPPPPPP/" + "".join(white) + " w - - 0 1"
    while not (opposite_color_bishops(white) and opposite_color_bishops(black) and (not args.castling or (king_between_rooks(white) and king_between_rooks(black)))) or (fen in memo) or not (memo.add(fen) is None and evaluation_passed(fen, args)):
        np.random.shuffle(white)
        np.random.shuffle(black)
        if args.castling:
            fen = "".join(black) + "/pppppppp/8/8/8/8/PPPPPPPP/" + "".join(white) + " w KQkq - 0 1"
        else:
            fen = "".join(black) + "/pppppppp/8/8/8/8/PPPPPPPP/" + "".join(white) + " w - - 0 1"
    engine = Stockfish(path=args.engine)
    engine.set_fen_position(fen)
    print(engine.get_board_visual())
    del engine
    print(fen)
    # if args.castling:
    #     print("[Variant \"Chess960\"]")
    # print("[SetUp \"1\"]")
    # print(f"[FEN \"{fen}\"]")
    print()
    if args.out is not None:
        with open(args.out, "w") as f:
            f.write(fen)
        print("Wrote FEN to", args.out)
        print()
    return fen


if __name__ == "__main__":
    main()
