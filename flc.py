import copy
import threading
import time
import torch
import requests
from local_device import LocalDevice
from utils import progress_bar
import torchvision
import torchvision.transforms as transforms

def average_weights(w):
    """
    Returns the average of the weights.
    平均权重
    """
    
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            # key : conv1.bias , w_avg[key] : tensor([-0.0420,  0.0525,  0.1072,  0.1419,  0.1415, -0.0156], device='cuda:0') 
            # print(f'key : { key } , w_avg[key] : {w_avg[key]}')
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

# 单轮次训练
def train(client, epoch):
    print(f'\nEpoch: {epoch+1} | Client: {client.id}')
    client.model.train().to(device)
    train_loss = 0
    correct = 0
    total_num = 0
    for batch_idx, (imgs, targets) in enumerate(client.trainloader):
        imgs, targets = imgs.to(device), targets.to(device)
        client.optimizer.zero_grad()
        outputs = client.model(imgs)
        loss = client.loss_fn(outputs, targets)
        loss.backward()
        client.optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_num += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(client.trainloader), 'Training | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total_num, correct, total_num))     

# 总训练
def train_with_epochs(client, epochs):
    for epoch in range(epochs):
        train(client, epoch)
        client.scheduler.step()  

receive_flag = False
middle_id = 0
receive_num = 0
def send_to_mid(client, epochs):
    global receive_flag, middle_id, receive_num
    send_flag = True
    while True:
        # send flag为真 执行模型训练，并发送模型参数
        if send_flag:
            train_with_epochs(client, epochs)
            para = client.model.state_dict()
            # 变成二进制文件保存
            # state_dict_bytes = torch.save(para, open('./para/temp_model'+ str(client.id) +'.pth', 'wb'))
            torch.save(para, './para/temp_model'+ str(client.id) +'.pth')
            url = client.url + '/sendpara'
            response = requests.post(url, data={'device_id': client.id})
            response = response.json()
            send_flag = False
            if response['status'] == 'completed':
                print(f'{client.id} | Model aggregation is completed.')
                receive_flag = True
                mid_id = response['mid_id']
                middle_id = mid_id
                path = './para/aggration_model'+ str(mid_id) +'.pth'
                new_para = torch.load(path)
                client.model.load_state_dict(new_para)
                send_flag = True
            else:
                print(f'{client.id} | Model aggregation is not yet completed.')
        else:
            if receive_flag:
                path = './para/aggration_model'+ str(middle_id) +'.pth'
                new_para = torch.load(path)
                client.model.load_state_dict(new_para)
                send_flag = True
                receive_num += 1
                if receive_num == 4-1:
                    receive_flag = False
                    receive_num = 0
            else:
                # print('Waiting for the middle server to send the model parameters.')
                time.sleep(3)
                continue



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    devices = [18,12,23,32]
    clients = []
    for i in range(len(devices)):
        clients.append(LocalDevice(id=devices[i], data_type='cifar10', device_type=0, user_type=0, set_length=10000, url='127.0.0.1:5000'))

    threads = []
    for client in clients:
        # t = threading.Thread(target=train_with_epochs, args=(client, local_epochs))
        t = threading.Thread(target=send_to_mid, args=(client, 1))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
