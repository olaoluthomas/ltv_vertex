# -*- coding: utf-8 -*-

from src.trainer import run_training


def main():
    run_training(train_data="./train_data.csv")


if __name__ == '__main__':
    main()