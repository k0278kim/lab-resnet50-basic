import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, distributed
from tqdm import tqdm
from nets.resnet50_imagenet import ResNet_imagenet, Bottleneck_imagenet
from nets.early_stopping import EarlyStopping

# 하이퍼파라미터
BATCH_SIZE = 32   # GPU 당 배치 크기 (총 배치 = BATCH_SIZE * world_size)
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "./resnet50-imagenet.pth"
RESUME_PATH = "checkpoint.pth"
NUM_WORKERS = 3


def train(rank, world_size):
    # -------------------
    # DDP 초기화
    # -------------------
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # -------------------
    # 모델 초기화
    # -------------------
    model = ResNet_imagenet(
        Bottleneck_imagenet, 
        [3, 4, 6, 3], 
        num_classes=1000, 
    ).to(device)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # -------------------
    # 옵티마이저 & 손실함수
    # -------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -------------------
    # 체크포인트 불러오기 (master만)
    # -------------------
    start_epoch = 0
    if rank == 0 and os.path.exists(RESUME_PATH):
        print(f"🔄 Loading checkpoint from {RESUME_PATH}...")
        checkpoint = torch.load(RESUME_PATH, map_location=device)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"✅ Resumed training from epoch {start_epoch}")
    dist.barrier()  # 모든 rank가 동기화되도록
    
    # -------------------
    # 데이터셋 & 분산 로더
    # -------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.ImageFolder("/data/imagenet/train", transform=transform)
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    # -------------------
    # 조기 종료
    # -------------------
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    # -------------------
    # 학습 루프
    # -------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_sampler.set_epoch(epoch)  # epoch별 shuffle 보장
        model.train()
        running_loss = 0.0

        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        else:
            pbar = train_loader

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if rank == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)

        # master만 로그 + 저장
        if rank == 0:
            print(f"✅ Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
            early_stopping(avg_loss)
            if early_stopping.early_stop or epoch + 1 == NUM_EPOCHS:
                print(f"⛔ Saving checkpoint at epoch {epoch+1}")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, "checkpoint.pth")

        dist.barrier()  # epoch 종료 시 동기화

    dist.destroy_process_group()


if __name__ == "__main__":
    import os

    world_size = torch.cuda.device_count()

    # 기본 환경변수 세팅
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    print(f"🌍 World Size = {world_size}")

    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
