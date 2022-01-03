# VGG16
by pytorch, ImageNet

![vggnet](https://user-images.githubusercontent.com/92920517/147895674-0191c250-9004-4398-8961-98a7fca5ba95.png)

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
