#!/usr/bin/env python3
import sys

from train_and_eval_3b import main


if __name__ == "__main__":
    if "--skip-eval" not in sys.argv:
        sys.argv.append("--skip-eval")
    main()
