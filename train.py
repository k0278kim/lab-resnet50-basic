import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
from nets.resnet50 import Bottleneck, ResNet
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, softmaxloss, epoch, epoch_size, epoch_size_val, gen, gen_test, Epoch, cuda, optimizer):
    total_loss = 0
    val_loss = 0

    with tqdm(total=epoch_size, desc='Epoch{}/{}'.format(epoch + 1, Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            images, targets = batch[0], batch[1]
            if cuda:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            optimizer.zero_grad()
            outputs = net(images)
            loss = softmaxloss(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss

            pbar.set_postfix(**{'total_loss': float(total_loss / (iteration + 1)),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('\nStart test')
    test_correct = 0
    with tqdm(total=epoch_size_val, desc='Epoch{}/{}'.format(epoch + 1, Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_test):
            images, targets = batch[0], batch[1]
            if cuda:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
            outputs = net(images)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == targets.data)
            pbar.set_postfix(**{'test AP': float(100 * test_correct / len(gen_test.dataset))})
            pbar.update(1)
    torch.save(net.state_dict(), 'logs/Epoch{}-Total_Loss{}.pth'.format((epoch + 1), (total_loss / ((iteration + 1)))))


if __name__ == '__main__':
    cuda = True
    pre_train = True
    CosineLR = True

    lr = 1e-3
    Batch_size = 512
    Init_Epoch = 0
    Fin_Epoch = 100

    # 모델 생성
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
    if pre_train:
        model_path = 'logs/resnet50-mnist.pth'
        model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ✅ 여러 GPU 활용 (A10 두 대 자동 분배)
    if torch.cuda.device_count() > 1:
        print("사용 가능한 GPU 개수:", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    train_dataset = datasets.MNIST(root='data/', train=True,
                                   transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='data/', train=False,
                                  transform=transforms.ToTensor(), download=False)

    gen = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True)
    gen_test = DataLoader(dataset=test_dataset, batch_size=Batch_size // 2, shuffle=False, num_workers=4, pin_memory=True)

    epoch_size = len(gen)
    epoch_size_val = len(gen_test)

    softmax_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    if CosineLR:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-10)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    for epoch in range(Init_Epoch, Fin_Epoch):
        fit_one_epoch(net=model, softmaxloss=softmax_loss, epoch=epoch, epoch_size=epoch_size,
                      epoch_size_val=epoch_size_val, gen=gen, gen_test=gen_test, Epoch=Fin_Epoch, cuda=cuda,
                      optimizer=optimizer)
        lr_scheduler.step()
