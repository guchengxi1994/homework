# -*- coding: utf-8 -*-
import os
import resnet
from tqdm import tqdm
import pickle,hashlib
import numpy as np
import imageio
from PIL import Image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet201
from keras.preprocessing import image as ppimage
from numpy import linalg as LA
import tensorflow as tf
from keras import backend as K
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from keras.applications.densenet import preprocess_input as preprocess_input_densnet
# =============================================================================
# 构建特征提取网络/vgg/resnet/desnet/...     基于keras
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)



cache_root_dir = 'save_feature_resnet2'
if not os.path.exists(cache_root_dir):
	os.makedirs(cache_root_dir)
def md5(s):
	m = hashlib.md5()
	m.update(s.encode("utf8"))
	return m.hexdigest()
def cache_key(f, *args, **kwargs):
	s = '%s-%s-%s' % (f.__name__, str(args), str(kwargs))
	return os.path.join(cache_root_dir, '%s.dump' % md5(s))
def cache(f):
	def wrap(*args, **kwargs):
		fn = cache_key(f, *args, **kwargs)
		if os.path.exists(fn):
			print('loading cache')
			with open(fn, 'rb') as fr:
				return pickle.load(fr)

		obj = f(*args, **kwargs)
		with open(fn, 'wb') as fw:
			pickle.dump(obj, fw,protocol=4)
		return obj
	return wrap

class FFNet:
    def __init__(self,width, height):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (width, height, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.include_top = False
        # include_top：是否保留顶层的3个全连接网络
        # weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
        # input_tensor：可填入Keras tensor作为模型的图像输出tensor
        # input_shape：可选，仅当include_top=False有效，应为长为3的tuple，指明输入图片的shape
        # pooling：当include_top = False时，该参数指定了池化方式。None代表不池化，最后一个卷积层的输出为4D张量。‘avg’代表全局平均池化，‘max’代表全局最大值池化。
        # classes：可选，图片分类的类别数，仅当include_top = True并且不加载预训练权重时可用。
        self.model_vgg19 = VGG19(weights=self.weight,input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                               pooling=self.pooling, include_top=self.include_top)
        self.model_resnet50 = ResNet50(weights=self.weight,input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                     pooling=self.pooling, include_top=self.include_top)
        self.model_resnet152 = resnet.ResNet152(weights=self.weight,
                                       input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                       pooling=self.pooling, include_top=self.include_top)
        self.model_desnet121 = DenseNet201(weights=self.weight,
                                     pooling=self.pooling, include_top=self.include_top)

    '''
    Use vgg16/Resnet model to extract features
    Output normalized feature vector
    '''
    # 提取vgg16最后一层卷积特征
    def vgg_extract_feat(self, img):
        # img = image.load_img(img_path)
        x = np.asarray(img)
        # print(x.shape)
        x = x.transpose(1, 0, 2)
        # print('8888888888')
        # print(x.shape)
        img = ppimage.img_to_array(x)
        # print(img.shape)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_vgg(img)
        feat = self.model_vgg19.predict(img)
        # print(feat.shape)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat
    # #提取resnet50最后一层卷积特征
    def resnet_extract_feat(self, img):
        x = np.asarray(img)
        # # print(x.shape)
        x = x.transpose(1, 0, 2)
        # print('8888888888')
        # print(x.shape)
        img = ppimage.img_to_array(x)
        img = ppimage.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_resnet(img)
        feat = self.model_resnet152.predict(img)
        # print(feat.shape)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat
    # #提取densenet121最后一层卷积特征
    def densenet_extract_feat(self, img):
        # img = image.load_img(img_path)
        img = ppimage.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_densnet(img)
        feat = self.model_desnet121.predict(img)
        print(feat.shape)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat





# =============================================================================
# 预处理，切边，默认：上下200，左右1000
# =============================================================================
def cut_edges(img,size):
    height, width = img.shape[:2]
    lr = []
    for left in range(0, width):
        if img[:, left].mean() < 200:
            lr.append(left)
            break
    for right in range(width - 1, 0, -1):
        if img[:, right].mean() < 200:
            lr.append(right)
            break
    imgx = img[200:-200,lr[0]+size:lr[1]-size]  # 200:-200
    return imgx



def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


'''
 Extract features and index the images
'''
@cache
def main():
    model_15k = FFNet(13000, 300)
    model_8k = FFNet(6800, 300)
    model_2k = FFNet(2800, 300)
    weld_path = '/home/aijr/weld/D/21/'
    img_list = get_imlist(weld_path)
    # num_name = 0;num_name_dict = {}

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    feats = []
    names = []
    # txt_path = 'mix_21_406_allselect.txt'
    group = []
    error = []
    class_15k = []
    class_7k = []
    class_2d8k = []
    img_list = get_imlist(weld_path)
    # print('A total of {} welding images were found to be extracted !'.format(len(pic_list)))
    for pic in tqdm(img_list):
        # num_name+=1
        # nn = str(num_name).zfill(7)
        # num_name_dict[nn] = pic_name
        pic_path = pic
        pic_name = os.path.split(pic)[1]
        # print(pic_name)
        try:
            image = imageio.imread(pic_path)
        except:
            error.append(pic_path)
            continue

        if image.shape[1] > 15000:
            imagec = cut_edges(image, 1000)
            imagep = Image.fromarray(imagec)
            imageps = imagep.resize((13000, 300), Image.NEAREST)
            # print('+++++')
            # print(imageps.size)
            norm_feat = model_15k.resnet_extract_feat(imageps)
            # imagem = cv2.resize(imagec, (13000, 300))
            # class_15k.append(pic_name)
            # ss = 13000
        elif image.shape[1] < 15000 and image.shape[1] > 6500:
            imagec = cut_edges(image, 100)
            imagep = Image.fromarray(imagec)
            imageps = imagep.resize((6800, 300), Image.NEAREST)
            # imagem = cv2.resize(imagec, (6800, 300))
            # class_7k.append(pic_name)
            norm_feat = model_8k.resnet_extract_feat(imageps)
            # ss = 6800
        else:
            imagec = cut_edges(image, 0)
            imagep = Image.fromarray(imagec)
            imageps = imagep.resize((2800, 300), Image.NEAREST)
            # imagem = cv2.resize(imagec, (2800, 300))
            # class_2d8k.append(pic_name)
            norm_feat = model_2k.resnet_extract_feat(imageps)
            # ss = 2800
          # 修改此处改变提取特征的网络
        feats.append(norm_feat)
        names.append(pic_name)
        print("extracting feature from image No. %d , %d images in total" % (len(names), len(img_list)))
    feats = np.array(feats)
    return (feats, names)
if __name__ == "__main__":
    main()
