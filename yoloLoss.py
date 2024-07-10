import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings

warnings.filterwarnings('ignore')  # 忽略警告消息
CLASS_NUM = 20    # （使用自己的数据集时需要更改）

class yoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        # 一般而言 l_coord = 5 ， l_noobj = 0.5
        super(yoloLoss, self).__init__()
        self.S = S  # S = 7
        self.B = B  # B = 2
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):  # box1(2,4)  box2(1,4)
        """
        这里要注意，box1包括两个边界框，box2只包括一个边界框，所以IOU的结果是box1的边界框分别与box2的边界框进行IOU操作，所以返回的IOU时2*1的
        """
        N = box1.size(0)  # 2
        M = box2.size(0)  # 1

        lt = torch.max(  # 返回张量所有元素的最大值
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, :2].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),
        )

        rb = torch.min(
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),
        )

        # 求差值
        wh = rb - lt  # [N,M,2]
        """
        wh < 0：这是一个布尔索引操作，它会生成一个与原数组wh形状相同的布尔数组。在这个布尔数组中，所有对应于原数组中值小于0的位置的元素
        都会被设置为True，其余位置为False。
        """
        wh[wh < 0] = 0  # clip at 0，去除那些可能没有交集的框
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  框重叠的部分的面积

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter) # iou=交集/并集
        return iou  # [2,1]
    """
    传入的两个参数格式为(batch_size*7*7*30)的张量，前者将图片出入神经网络得到的输出值，
    后者就是yoloData制作的target也就是ground truth。需要提取ground truth与pred_target的bbox信息，置信度信息
    以及类别信息，求取损失函数。这里有五个损失，都是在YOLOv1原文中定义的
    """
    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,7,7,30)
        target_tensor: (tensor) size(batchsize,7,7,30) --- ground truth
        '''
        N = pred_tensor.size()[0]  # batchsize
        coo_mask = target_tensor[:, :, :, 4] > 0  # 具有目标标签的索引值 true ，batchsize*7*7
        noo_mask = target_tensor[:, :, :, 4] == 0  # 不具有目标的标签索引值 false ，batchsize*7*7
        # unsqueeze(-1)：在最后面增加一个维度，expand_as：将原本的张量扩充，一般是将通道数扩充，扩充的部分就是将原来的部分复制粘贴。
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)  # 得到含物体的坐标等信息,复制粘贴 batchsize*7*7*30
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)  # 得到不含物体的坐标等信息 batchsize*7*7*30

        """
        首先选择pred_tensor中coo_mask为True的元素。这通常用于从大量预测中筛选出有效的或感兴趣的预测。
        然后，使用.view(-1, int(CLASS_NUM + 10))将这些选出的元素重新塑形。-1表示该维度的大小将自动计算，以保持总元素数量不变。
        int(CLASS_NUM + 10)指定了第二维的大小，这里假设每个预测包含CLASS_NUM个类别预测加上额外的10个值（可能是边界框的坐标或其他属性）。
        """
        coo_pred = pred_tensor[coo_mask].view(-1, int(CLASS_NUM + 10))  # view类似于reshape
        # .contiguous()确保这些元素在内存中是连续的，这对于某些PyTorch操作是必需的，尤其是在重塑（reshape）或转换设备（如CPU到GPU）时。
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # 塑造成X行5列（-1表示自动计算），一个box包含5个值
        class_pred = coo_pred[:, 10:]  # [n_coord, 20]

        coo_target = target_tensor[coo_mask].view(-1, int(CLASS_NUM + 10))
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        # 不包含物体grid ceil的置信度损失
        noo_pred = pred_tensor[noo_mask].view(-1, int(CLASS_NUM + 10))
        noo_target = target_tensor[noo_mask].view(-1, int(CLASS_NUM + 10))
        """
        创建了一个名为noo_pred_mask的PyTorch张量（tensor），它的数据类型是torch.cuda.ByteTensor，并且其形状（size）与另一个张量noo_pred相同。
        将这个新创建的张量转换为布尔类型（bool），以便它可以被用作掩码（mask）来索引或筛选其他张量。
        调用.zero_()方法将noo_pred_mask中的所有元素初始化为0（在布尔上下文中，0被视为False）。
        """
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()).bool()
        noo_pred_mask.zero_()

        # YOLOv1原文提到，如果不负责预测物体的noobj为1，负责预测物体的noobj为0
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1

        # 只会留下noo_pred_mask被置位为1的数字，也就是只有第4个位置和第9个位置的数字会被留下来，其他都为0，方便使用均方损失计算置信度的损失
        noo_pred_c = noo_pred[noo_pred_mask]  # noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)  # 均方误差

        # compute contain obj loss
        coo_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()  # ByteTensor 构建Byte类型的tensor元素全为0
        coo_response_mask.zero_()  # 全部元素置False                            bool:将其元素转变为布尔值

        no_coo_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()  # ByteTensor 构建Byte类型的tensor元素全为0
        no_coo_response_mask.zero_()  # 全部元素置False                            bool:将其元素转变为布尔值

        box_target_iou = torch.zeros(box_target.size()).cuda()

        # box1 = 预测框  box2 = ground truth
        for i in range(0, box_target.size()[0], 2):  # box_target.size()[0]：有多少bbox，并且一次取两个bbox
            box1 = box_pred[i:i + 2]  # 第一个grid ceil对应的两个bbox，取的是 i 和 i+1，2*5
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            # 这个是求bbox的左上角和右下角坐标，也就是xmin,ymin,xmax,ymax
            box1_xyxy[:, :2] = box1[:, :2] / float(self.S) - 0.5 * box1[:, 2:4]  # 原本(xc,yc)为7*7 所以要除以7
            box1_xyxy[:, 2:4] = box1[:, :2] / float(self.S) + 0.5 * box1[:, 2:4]

            box2 = box_target[i].view(-1, 5)# 因为对于真实值，两个bbox的值是完全一样的
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] / float(self.S) - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / float(self.S) + 0.5 * box2[:, 2:4]

            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4]) # 2*1
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()
            coo_response_mask[i + max_index] = 1  # IOU最大的bbox
            no_coo_response_mask[i + 1 - max_index] = 1  # 舍去的bbox
            # confidence score = predicted box 与 the ground truth 的 IOU
            """
            torch.LongTensor([4]).cuda()：这里创建一个只包含单个元素4的Long类型张量，并将其移动到GPU上。
            """
            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = max_iou.data.cuda()

        box_target_iou = Variable(box_target_iou).cuda()
        # 置信度误差（含物体的grid ceil的两个bbox与ground truth的IOU较大的一方）
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        # IOU较小的一方
        no_box_pred_response = box_pred[no_coo_response_mask].view(-1, 5)
        no_box_target_response_iou = box_target_iou[no_coo_response_mask].view(-1, 5)
        no_box_target_response_iou[:, 4] = 0  # 保险起见置0（其实原本就是0）

        box_target_response = box_target[coo_response_mask].view(-1, 5)

        # 含物体grid ceil中IOU较大的bbox置信度损失
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)
        # 含物体grid ceil中舍去的bbox损失
        no_contain_loss = F.mse_loss(no_box_pred_response[:, 4], no_box_target_response_iou[:, 4], size_average=False)
        # bbox坐标损失
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), size_average=False)

        # 类别损失
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)

        return (self.l_coord * loc_loss + contain_loss + self.l_noobj * (nooobj_loss + no_contain_loss) + class_loss) / N
