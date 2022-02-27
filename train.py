import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from Pokemon import Pokemon
# from MobileNetV1 import MobileNetV1
# from MobileNetV2 import MobileNetV2
# from MobileNetV3 import MobileNetV3_Large, MobileNetV3_Small
from vit_model import vit_base_patch16_224_in21k

epoch_size = 5
learning_rate = 1e-3
batch_size = 32
resize = 224
root = 'E:\学习\机器学习\数据集\pokemon'
mdl_file = 'MobileNetV3_Small.mdl'

train_data = Pokemon(root=root, resize=resize, mode='train')
val_data = Pokemon(root=root, resize=resize, mode='val')
test_data = Pokemon(root=root, resize=resize, mode='test')

train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MobileNetV3_Small()
model = vit_base_patch16_224_in21k(num_classes=5, has_logits=False)
print(model)

crition = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_acc = 0
best_epoch = 0

for epoch in range(epoch_size):

    # 训练集训练
    model.train()
    for batchidx, (image, label) in enumerate(test_loader):

        # image = image.to(device)
        # label = label.to(device)

        logits = model(image)
        loss = crition(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batchidx%2 == 0:
            print("epoch:{}/{}, batch:{}/{}, loss:{}"
                  .format(epoch+1, epoch_size, batchidx, len(test_loader), loss))

    # 测试集挑选
    model.eval()
    correct = 0
    for image, label in val_loader:

        # image = image.to(device)
        # label = label.to(device)

        with torch.no_grad():
            logits = model(image)
            pred = logits.argmax(dim=1)

        correct += torch.eq(pred, label).sum().float().item()

    acc = correct/len(val_data)
    print("epoch:{}, acc:{}".format(epoch+1, acc))

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch

        torch.save(model.state_dict(), mdl_file)
        print("[get best epoch]- best_acc:{}, best_epoch:{}".format(best_acc, best_epoch))

