import torch
from model import Model
import torch.optim as optim


from apex import amp

def test(testloader, idx2label)->None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    checkpoint = torch.load('amp_checkpoint.pt')

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])


    # model.load_state_dict(torch.load('model.pth'))

    model.eval()

    class_correct = list(0. for i in range(1000))
    class_total = list(0. for i in range(1000))
    with torch.no_grad():
        for (images, labels) in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            #outputs : batch_num, 1000(image class num)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy_sum=0
    for i in range(1000):
        temp = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' %(
            idx2label[i], temp))
        accuracy_sum+=temp
    print('Accuracy average: ', accuracy_sum/1000)