import torch
import cv2
import os
import os.path
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image
# from write_txt import VOC_CLASSES # 这个使用要谨慎，因为对应文件里面定义的两个txt是全局变量，导入的时候里面的全局变量会重新赋值

CLASS_NUM = 20  # 使用其他训练集需要更改
# CLASS_NUM=len(VOC_CLASSES) # 类别的数量
# os.chdir('/root/workspace/YOLOV1-pytorch/')

class yoloDataset(Dataset):
    image_size = 448  # 输入图片大小

    def __init__(self, img_root, list_file, train, transform):   # list_file为txt文件  img_root为图片路径
        """
        逐行读取生成的文本文件的内容，然后对其进行分类，将信息保存在fnames，boxes，labels三个列表中
        """
        self.root = img_root
        self.train = train
        self.transform = transform
        # 后续要提取txt文件信息，分类后装入以下三个列表
        self.fnames = []
        self.boxes = []
        self.labels = []

        self.S = 7   # YOLOV1
        self.B = 2   # 相关
        self.C = CLASS_NUM  # 参数
        self.mean = (123, 117, 104)  # RGB
        file_txt = open(list_file,'r')
        lines = file_txt.readlines()   # 读取txt文件每一行
        for line in lines:   # 逐行开始操作
            # strip()  # 移除首位的换行符号；split()  # 以空格为分界线，将所有元素组成一个列表
            splited = line.strip().split() # 移除首位的换行符号再生成一张列表
            self.fnames.append(splited[0])  # 存储图片的名字
            num_boxes = (len(splited) - 1) // 5  # 每一幅图片里面有多少个bbox
            box = []
            label = []
            for i in range(num_boxes): # bbox四个角的坐标
                x = float(splited[1 + 5 * i])
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                c = splited[5 + 5 * i]  # 代表物体的类别，即是20种物体里面的哪一种  值域 0-19
                box.append([x, y, x2, y2])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    # 访问坐标的时候就会直接执行这个函数
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root + fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        if self.train:  # 数据增强里面的各种变换用torch自带的transform是做不到的，因为对图片进行旋转、随即裁剪等会造成bbox的坐标也会发生变化，所以需要自己来定义数据增强
            img, boxes = self.random_flip(img, boxes) # 随机翻转
            img, boxes = self.randomScale(img, boxes) # 随机伸缩变换
            img = self.randomBlur(img)# 随机模糊处理
            img = self.RandomBrightness(img)# 随机调整亮度
            # img = self.RandomHue(img)
            # img = self.RandomSaturation(img)
            img, boxes, labels = self.randomShift(img, boxes, labels)# 平移转换
            # img, boxes, labels = self.randomCrop(img, boxes, labels)
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)  # 坐标归一化处理，为了方便训练，这个表示的bbox的宽高占整个图像的比例
        img = self.BGR2RGB(img)  # because pytorch pretrained model use RGB
        img = self.subMean(img, self.mean)  # 减去均值
        """
        这里对图像resize后不需要对boxes变化，原因一是这里不是图像增强，只是方便图片输入网络；
        原因二是YOLO原文写的是，对bbox的宽高做归一化，这个归一化是相当于整个原来图像的宽高进行归一化的（上面已经归一化了），而对bbox的中心坐标的归一化
        是相当于bbox所在的grid cell的左上角坐标进行归一化的，也就是下面的encoder操作，所以这一步是正确的。

        而且在后面使用到这个bbox的xywh的时候，是会做相应的操作的，详情可以看yoloLoss
        """
        # YOLO V1输入图像大小设置为448*448* 3
        img = cv2.resize(img, (self.image_size, self.image_size))  # 将所有图片都resize到指定大小，这里不是图像增强，而是为了方便网络的输入
        target = self.encoder(boxes, labels)  # 将图片标签编码到7x7*30的向量

        for t in self.transform:
            img = t(img)

        # 返回的img是经过图像增强的img
        return img, target

    def __len__(self):
        return self.num_samples

    # def letterbox_image(self, image, size):
    #     # 对图片进行resize，使图片不失真。在空缺的地方进行padding
    #     iw, ih = image.size
    #     scale = min(size / iw, size / ih)
    #     nw = int(iw * scale)
    #     nh = int(ih * scale)
    #
    #     image = image.resize((nw, nh), Image.BICUBIC)
    #     new_image = Image.new('RGB', size, (128, 128, 128))
    #     new_image.paste(image, ((size - nw) // 2, (size - nh) // 2))
    #     return new_image

    def encoder(self, boxes, labels):  # 输入的box为归一化形式(X1,Y1,X2,Y2) , 输出ground truth  (7*7*30)
        grid_num = 7
        target = torch.zeros((grid_num, grid_num, int(CLASS_NUM + 10)))    # 7*7*30
        # cell_size 是图像宽度和高度被划分成的等分数，用于将归一化的坐标转换为网格索引。
        cell_size = 1. / grid_num  # 1/7
        # 这个是bbox的归一化后的宽高
        wh = boxes[:, 2:] - boxes[:, :2] # wh = [w, h]  1*1

        # 物体中心坐标集合
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2  # 归一化含小数的中心坐标
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]  # 中心坐标  1*1
            """
            ij 并不是直接表示“左上角坐标（7*7)为整数，而是表示边界框中心点所在的网格的索引。ceil()表示向上取整；-1是因为python的索引是从0开始的
            ij 是一个包含两个元素的tensor，分别表示边界框中心点所在的网格的x和y索引

            下面的公式的解释可以这样理解：假设有一个图像大小为w*h，现在把图像分为7分，问坐标为（x，y）的点位于哪一个网格中，这就是小学乘法问题，
            很明显，求一下x和y占比w和h的占比，分别乘以7，最后向上取整，所以答案就是（7x/w,7y/h）.ceil(),这里再看cxcy_sample本身就是已经归一化（也就是已经除以w）
            了，所以直接乘7，也就是 / cell_size 就可以得到结果。-1是为了让索引从0开始。 
            """
            ij = (cxcy_sample / cell_size).ceil() - 1  # 左上角坐标 （7*7)为整数
            # 这里先1后0是因为坐标提取就是先行后列
            # 第一个框的置信度，4表示第一个标注框的置信度存储在下标为4的位置，下面9同理，并且这里的意义是，只有有标注框的置信度置位为1
            target[int(ij[1]), int(ij[0]), 4] = 1
            # 第二个框的置信度
            target[int(ij[1]), int(ij[0]), 9] = 1

            target[int(ij[1]), int(ij[0]), int(labels[i]) + 10] = 1  # 20个类别对应处的概率设置为1

            xy = ij * cell_size  # 归一化左上坐标  （1*1）
            # 在YOLOV1原文中，其bbox的五个参数中的x，y就是中心坐标相对于其grid cell左上角的坐标的相对值
            delta_xy = (cxcy_sample - xy) / cell_size  # 中心与左上坐标差值  （7*7）

            # 坐标w,h代表了预测的bounding box的width、height相对于整幅图像width,height的比例
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]  # w1,h1
            target[int(ij[1]), int(ij[0]), :2] = delta_xy  # x1,y1

            # 每一个网格有两个边框，在真实数据中，两个边框的值是一样的
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]  # w2,h2
            # 由此可得其实返回的中心坐标其实是相对左上角顶点的偏移，因此在进行预测的时候还需要进行解码
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy  # [5,7) 表示x2,y2
        """
        这里来解释为什么(xc,yc) = 7*7   (w,h) = 1*1
        首先解释一个简单点的，(w,h) = 1*1，是因为target保存的时候，是直接保存了前面归一化的wh，所以这里是1*1
        接下来解释(xc,yc) = 7*7，这里理一遍整个流程，首先获得了归一化的中心坐标cxcy，这个时候是1*1的，和上面wh的解释一样，
        然后取了一个cxcy作为例子，也就是cxcy_sample，那么自然cxcy_sample也是1*1的。
        然后求ij的时候是cxcy_sample*7，所以ij是7*7，
        接着求xy是ij/7，所以，xy为1*1
        最后求delta_xy的时候是(cxcy_sample - xy)*7，并且保存的也是delta_xy，那么自然也就是7*7的，这里理解很重要
        """
        return target   # (xc,yc) = 7*7   (w,h) = 1*1

    # 以下方法都是数据增强操作

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):# BGR变换为HSV
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):# 随机调整亮度
        if random.random() < 0.5:
            # 由于直接在BGR颜色空间调整亮度可能会改变图像的颜色，因此通常会在HSV（色调、饱和度、亮度）颜色空间中进行调整。
            hsv = self.BGR2HSV(bgr)
            # 使用 cv2.split() 将HSV图像分离成三个单独的通道：色调（H）、饱和度（S）和亮度（V）。
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            # 使用 np.clip() 函数确保亮度值不会超出有效范围（0到255），并将结果转换回原始HSV图像的数据类型。
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            # 使用 cv2.merge() 将调整后的亮度通道（V）与原始的色调（H）和饱和度（S）通道合并回HSV图像。
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):# 随机模糊处理
        if random.random() < 0.5:
            """
            cv2.blur() 函数实际上是一个简单的平均模糊函数，它会计算核内所有像素的平均值，并用这个平均值替换核中心的像素值。
            这种方法在去除图像噪声的同时，也会丢失一些细节信息。
            """
            bgr = cv2.blur(bgr, (5, 5))# 固定模糊核是5*5，核越大，模糊效果越明显
        return bgr

    def randomShift(self, bgr, boxes, labels):# 平移转换
        """
        主要是对输入的图像进行随机的平移变换，并且相应的更新图像中的目标框的位置，同时它还会处理平移后可能超过图像边界的情况，以及更新目标框的位置确保它们
        仍然位于图像的有效区域
        """
        """
        这里计算的是每一个bbox的center，首先boxes是一个二维数组，所以第一个冒号是取了二维数组里面所有的元素，
        而2：表示从每一个元素里面的第三第四列，也就是xmax和ymax，而：2表示取第一第二列，也就是xmin和ymin，如此计算得到center=（xcenter，ycenter）
        """
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            # 创建一个与原图相同大小和类型的全零图像，并用特定的BGR值（104, 117, 123）填充，这个值通常用于图像预处理中的均值归一化。
            after_shift_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shift_image[:, :, :] = (104, 117, 123)  # bgr
            # 随机生成水平或者垂直方向上的平移量
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            # 根据平移量的正负，分别处理图像的不同部分，分四种情况处理
            if shift_x >= 0 and shift_y >= 0:# 右下
                # 这里要注意，这里的注释里面括号是(y，x)，不是（x，y）
                # 填充，偏移后需要填充的部分是(shift_y,shift_x)到(height,width),用的是原始图像的（0,0）到（height - int(shift_y)，width - int(shift_x)）填充
                after_shift_image[int(shift_y):,int(shift_x):,:] = bgr[:height - int(shift_y),:width - int(shift_x),:]
            elif shift_x >= 0 and shift_y < 0:# 右上
                # 填充，偏移后需要填充的部分是(0，height + int(shift_y))（这里int(shift_y)是个负数），原始图像是（-int(shift_y)，0）到（height，width - int(shift_x)）
                after_shift_image[:height + int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width - int(shift_x),:]
            elif shift_x < 0 and shift_y >= 0:# 左下
                after_shift_image[int(shift_y):, :width +int(shift_x), :] = bgr[:height -int(shift_y), -int(shift_x):, :]
            elif shift_x < 0 and shift_y < 0:# 左上
                after_shift_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):, -int(shift_x):, :]

            # 扩展后的center
            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            # 检查中心点的坐标是否在图像的边界内
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            """
            view是维度变换
            这里可以看到boxes是m*4的，m是box的数量，4是坐标的数量，labels是m维的，所以view将mask展平成m维的，实际上是（m，1）
            """
            mask = (mask1 & mask2).view(-1, 1)
            # mask.expand_as(boxes)的操作等同于mask.squeeze(1)，这里操作后，boxes_in只会包含到mask中维true的部分，得到一个新的张量，也就是包含了中心的bbox会被保留
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:# 如果变换后没有包含任何的标注框，那么就不变换，返回原始的图片、bbox和labels
                return bgr, boxes, labels
            # 变换标注框的坐标
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in + box_shift
            # 保留还剩余的标注框的label
            labels_in = labels[mask.view(-1)]
            return after_shift_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):# 随机伸缩变换
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            """
            使用 expand_as(boxes) 方法将 scale_tensor 扩展到与 boxes 相同的形状，以便逐元素相乘。
            这样，boxes 中的每个边界框都会根据 scale_tensor 中的值进行相应的伸缩变换。
            expand_as方法要求 boxes 的第一维度（即批处理大小或边界框的数量）与 scale_tensor 的第一维度（这里是1）兼容
            """
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):# 减掉均值
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):# 随机翻转
        """
        给定的图像 im 和对应的边界框坐标 boxes 进行随机水平翻转。如果随机生成的数小于0.5（即有一半的概率），
        则执行翻转操作，并相应地调整边界框的坐标以反映图像的变化。
        """
        if random.random() < 0.5:
            # 使用 np.fliplr(im).copy() 对图像进行水平翻转，并复制结果以避免修改原始图像
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape# h w c
            xmin = w - boxes[:, 2]# w-xmax
            xmax = w - boxes[:, 0]# w-xmin
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im


def main():
    file_root = 'VOCdevkit/VOC2007/JPEGImages/'
    train_dataset = yoloDataset(
        img_root=file_root,
        list_file='voctrain.txt',
        train=True,
        transform=[ToTensor()])
    """
    DataLoader是一个常用的类，用于封装数据集并提供批量加载数据的功能。它支持多进程数据加载、打乱数据、自定义数据采样等。
    当创建后，DataLoader会返回一个可迭代对象，每一次迭代会返回一批数据（batch_size）

    shuffle=False：这意味着数据加载器不会打乱数据集中的样本顺序。如果您希望每个epoch的数据顺序都不同，应该将此参数设置为 True
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        drop_last=True,
        shuffle=False,
        num_workers=0)

    """
    将一个数据加载器（train_loader）转换为一个迭代器（train_iter）

    通过将 train_loader 转换为迭代器 train_iter，您可以使用 next(train_iter) 来逐个批次地获取数据。
    但是，请注意，一旦迭代器被耗尽（即，当您已经遍历了数据集中的所有批次时），再次调用 next(train_iter) 将引发 StopIteration 异常。
    """
    # train_iter = iter(train_loader)
    # for i in range(100):
    #     img, target = next(train_iter)
    #     print(img.shape)
    #     print(target)

    for img,target in train_loader:
        print(img.shape)
        print(target)


if __name__ == '__main__':
    os.chdir('/root/workspace/YOLOV1-pytorch/')
    main()
