# VGG16
by pytorch, ImageNet

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch', default=100, type=int,
                        help='number of each process batch number')
