import copy
from flask import Flask, request, jsonify
import torch
from local_device import LocalDevice
from utils import progress_bar

app = Flask(__name__)
device = 'cuda'
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

def test(client, epoch):
    # global best_acc
    client.model.eval()
    test_loss = 0
    correct = 0
    total_num = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(client.testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = client.model(inputs)
            loss = client.loss_fn(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_num += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(client.testloader), 'Testing | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total_num, correct, total_num))
        

weights = []
midserver = LocalDevice(id=10, data_type='cifar10', device_type=1, user_type=0, set_length=1, url='127.0.0.1:5001')
@app.route('/sendpara', methods=['POST'])
def update_model():
    global weights, midserver
    # 接收数据
    device_id = request.form.get('device_id')
    
    # 保存接收到的模型参数文件
    path = './para/temp_model'+ str(device_id) +'.pth'
    
    # 加载模型参数
    received_state_dict = torch.load(path)
    weights.append(received_state_dict)
    if len(weights) == 4:
        new_weight = average_weights(weights)
        weights = []
        torch.save(new_weight, './para/aggration_model'+ str(midserver.id) +'.pth')
        # 对模型进行相关验证操作
        midserver.model.load_state_dict(new_weight)
        test(midserver, 0)
        # 返回相应信息
        return jsonify({'status': 'completed', 'message': 'Model updated successfully', 'mid_id': midserver.id})
        
    else:
        return jsonify({'status': 'waiting', 'message': 'Waiting for more parameter datas'})
    

if __name__ == '__main__':
    app.run(debug=True)