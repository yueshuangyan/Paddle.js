import os
import sys
import math
import paddle as paddle
import numpy as np
import cv2
from PIL import Image
import paddle.fluid as fluid
import argparse
import json

paddle.enable_static()
# shape
# 模型所在目录名
# pretrained_model = "infer_model/gesture_rec_new"
# pretrained_model = "infer_model/gesture_rec_new_3"
# pretrained_model = "infer_model/MobileNetV2"
pretrained_model = "../../../models/paddleModels/MobileNetV2"
# pretrained_model = "../infer_model/pose_point"
# 模型名
modelName = "__model__"
# 参数（权重）名，若为分片权重，设置为None
paramsName = "params"
# paramsName = None
# 待保存的var名，用于将某个var的值全部输出到文件中
# saved_var_name = "output_1/fc.tmp_0"
# saved_var_name = "concat_0.tmp_0"
# saved_var_name = "gesture.tmp_1"
# saved_var_name = "save_infer_model/scale_0"
# saved_var_name = "save_infer_model/scale_0.tmp_0"
# saved_var_name = "bilinear_interp_0.tmp_0"
saved_var_name = "save_infer_model/scale_0"
# 保存var值的路径
saved_path = "result"
shape = (1, 3, 224, 224)
total_index = 245

#coding:utf-8
import os
import cv2
import glob
import numpy as np

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def process_image(img):
    size = img.shape
    h, w = size[0], size[1]
    min_side = 256
    #长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    else:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    print(int(top))
    print(int(bottom))
    print(int(left))
    print(int(right))
    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目
    #print pad_img.shape
    #cv2.imwrite("after-" + os.path.basename(filename), pad_img)
    pad_img = to_chw_bgr(pad_img)
    print(pad_img[0][0][128])
    return pad_img


def get_reader(img_path):
    resize_size = (256, 256)
    # mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    mean = np.array([0, 0, 0], np.float32).reshape(1, 1, 3)
    std = np.array([1, 1, 1], np.float32).reshape(1, 1, 3)
    image = cv2.imread(img_path)
    inp = process_image(image)
    # inp = (inp / 255. - mean) / std
    # inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    return inp

def stride_print(input):
    tensor = input.flatten().tolist()
    length = len(tensor)
    if length < 3000:
        # print(tensor)
        return
    size = 20
    stride = math.floor(length / size)
    if stride == 0:
        stride = 1
    size = math.floor(length / stride)
    nums = []
    for i in range(0, size):
        item = tensor[i * stride]
        nums.append(str(i * stride) + ": " + str(item))
    # print(nums)

def saveToFile(input, filename):
    #os.chdir("./output")
    #input.tofile('{}.txt'.format(filename), sep='\n', format='%s')
    input.tofile("./output/{}".format(filename))

def all_print(input):
    tensor = input.flatten().tolist()
    length = len(tensor)
    stride = 1
    size = math.floor(length / stride)
    nums = []
    for i in range(0, size):
        item = tensor[i * stride]
        nums.append(item)
        # print(item)
    # print(length)

# def create_input(image, shrink):
#     image_shape = [3, image.size[1], image.size[0]]
#     if shrink != 1:
#         h, w = int(image_shape[1] * shrink), int(image_shape[2] * shrink)
#         image = image.resize((w, h), Image.ANTIALIAS)
#         image_shape = [3, h, w]
#     img = np.array(image)
#     # img = to_chw_bgr(img)
#     # mean = [104., 117., 123.]
#     # scale = 0.007843
#     # img = img.astype("float32")
#     # img -= np.array(mean)[:, np.newaxis, np.newaxis].astype("float32")
#     # img = img * scale
#     img = [img]
#     img = np.array(img)
#     return img

def to_chw_bgr(image):
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # image = image[[2, 1, 0], :, :]
    return np.array(image).reshape(1, 3, 256, 256).astype('float32')

# img = Image.open("fist.jpg")
# img = create_input(img, 1)
# data = img.flatten().tolist()


def getLayerResult(shape):

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())


    # img = np.fromfile("input.dat", dtype=np.float32).reshape(shape)
    # img = np.loadtxt('nchw.txt', dtype=np.float32, delimiter=',').reshape(shape)


    [prog, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=pretrained_model, executor=exe, model_filename=modelName, params_filename=paramsName)

    feeded_vars = [prog.global_block().vars[varname] for varname in feed_target_names]
    # feedShapes = feeded_vars[0].shape


    feedData = {}
    for i, shapeItem in enumerate(shape):
        curShape = tuple(shapeItem)
        feedData[feed_target_names[i]] = np.full(curShape, 1.0, "float32")
        # feedData[feed_target_names[i]] = np.loadtxt("./inputs", dtype="float32", delimiter=',').reshape(1, 3, 128, 128)


    # img = get_reader('fist.jpg')
    # print(img)

    # image_path = "./fist.jpg"
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (256, 256))
    # img = img.reshape(1, 3, 256, 256)
    # img = img.astype("float32")
    # print(img.shape)
    # print(img)

    index = 0
    for op in prog.current_block().ops:
        # print("------------------------------------------")
        # print("【第" + str(index) + "个OP】：" + op.type)
        # print("op type:" + op.type)
        for name in op.output(op.output_names[0]):
            var = fluid.framework._get_var(name, prog)
            # print(var)
            var.persistable = True
        index = index + 1

    # print("fetchtargets: ", fetch_targets)
    outputs = exe.run(prog, feed=feedData, fetch_list=fetch_targets, return_numpy=False)
    # if name == "conv6_3_expand.tmp_0":
    #     file = open('/Users/bluebird/baidu/fluid_tools/check-temp/data.txt','a')
    #     dataList = data.flatten().tolist()
    #     print("hahahahah")
    #     for a in range(0, len(dataList)):
    #         file.write(str(dataList[a]));
    #         file.write(",")
    #         rint("hahahahah")
    #         # file.write();
    #         file.close()

    #  修改persistable属性后执行模型
    for v in prog.list_vars():
        if not v.persistable:
            v.persistable = True

    exe.run(prog, feed=feedData, fetch_list=fetch_targets, return_numpy=False)

    var = fluid.global_scope().find_var(saved_var_name)
    data = np.array(var.get_tensor())
    path = saved_path
    if os.path.exists(path):
        os.remove(path)
    file = open(path,'a')
    dataList = data.flatten().tolist()
    print(dataList)
    # print(saved_var_name)
    # print(output)

    # print(json.dumps(dataList))

    # print(data.shape)
    for a in range(0, len(dataList)):
        file.write(str(dataList[a]))
        if a != len(dataList)-1:
            file.write(",")
    file.close()
    # print("保存成功")
    sys.exit()

    index = 0
    for op in prog.current_block().ops:
        # print("------------------------------------------")
        # print("【第" + str(index) + "个OP】：" + op.type)
        index = index + 1
        try:
            for na in op.output_names:
                for name in op.output(na):
                    var = fluid.global_scope().find_var(name)
                    data = np.array(var.get_tensor())
                    if name.strip() == saved_var_name.strip():
                    # if index == total_index:
                        # print('********************')
                        # print(name)
                        path = saved_path
                        if os.path.exists(path):
                            os.remove(path)
                        file = open(path,'a')
                        dataList = data.flatten().tolist()
                        # print(data.shape)
                        for a in range(0, len(dataList)):
                            file.write(str(dataList[a]))
                            if a != len(dataList)-1:
                                file.write(",")
                        file.close()
                        # print("保存成功")
                        sys.exit()


                    # print(name)
                    # print(data.shape)
                    # data = stride_print(data)
                    # saveToFile(data, name.replace("/", "-"))

        except:
            pass

#outputas: reshape2_7.tmp_0 reshape2_23.tmp_0 reshape2_31.tmp_0
#name = "reshape2_31.tmp_0"
#var = fluid.global_scope().find_var(name)
#data = np.array(var.get_tensor())
#print(name)
#print(data.shape)
#stride_print(data)
#saveToFile(data, name.replace("/", "-"))
#
#var = fluid.framework._get_var(name, prog)
#print(var)

# print("outputs:")
# print(outputs)
# output = np.array(outputs[0][:])
# stride_print(output)
# sys.exit()
# # print(output)
# det_conf = output[:, 1]
# det_xmin = shape[3] * output[:, 2]
# det_ymin = shape[2] * output[:, 3]
# det_xmax = shape[3] * output[:, 4]
# det_ymax = shape[2] * output[:, 5]
# det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))
# print(det.shape)
# print(det)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='转化为PaddleJS模型参数解析')
    p.add_argument('--model', help='输入模型名称，默认放置在tools/infer_model下', required=True)
    p.add_argument('--inputWidth', help='输入模型的shape width, ', required=False)
    p.add_argument('--inputHeight', help='输入模型的shape height, ', required=False)
    p.add_argument('--inputShape', help='输入模型的shape, ', required=False)
    p.add_argument('--type', help='输入模型的type, ', required=False)
    p.add_argument('--modelFileName', help='输入模型文件名称，默认__model__', required=False)
    p.add_argument('--paramFileName', help='输入参数文件名称，默认__params__, 若参数文件为分片形式，指定未None', required=False)
    p.add_argument('--opName', help='输出op的outputName, 默认运行完成', required=False)

    args = p.parse_args()
    pretrained_model = os.path.join('../../../models/paddleModels/', args.model)
    if args.modelFileName:
        modelName = args.modelFileName
    if args.paramFileName:
        paramsName = args.paramFileName
    if args.opName:
        saved_var_name = args.opName

    inputWidth = args.inputWidth or 256
    inputHeight = args.inputHeight or 256
    inputShape = args.inputShape


    shape = [[1, 3, int(inputWidth), int(inputHeight)]]
    if inputShape:
        shape = json.loads(inputShape)
    getLayerResult(shape)


