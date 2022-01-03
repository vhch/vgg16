import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Model

import torch
import torchvision
import torchvision.transforms as transforms

import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

def train(gpu, args)->None:
    print("gpu",gpu)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.ImageFolder('./train', transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    trainset,
    num_replicas=args.world_size,
    rank=rank
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)
    
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    torch.cuda.manual_seed_all(0)
    device=gpu

    model = Model()
    model = model.cuda(device)

    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

#######################################################################################
    # checkpoint= torch.load('amp_checkpoint.pt', map_location=torch.device('cpu'))
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # amp.load_state_dict(checkpoint['amp'])
    model = DDP(model)
#######################################################################################
    if gpu == 0:
        print("training start")

    for epoch in range(args.epochs):
        loss_train = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # if not torch.isfinite(loss):
            #     print('WARNING: non-finite loss')
            #     exit()

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            # print statistics
            if gpu == 0:
                loss_train += loss
                if i % 50 == 49:    # print every 2000 mini-batches
                    print('epoch : %d, iteration : %d, loss : %.4f' %
                        (epoch + 1, i + 1, loss_train / 50))
                    loss_train = 0.0
        # torch.save(model.state_dict(), 'model.pth')
        if gpu==0:
            checkpoint = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict()
            }
            torch.save(checkpoint, 'amp_checkpoint.pt')

    if gpu == 0:
        print('training finish')