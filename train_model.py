from __future__ import division, print_function
from keras.applications.resnet50 import preprocess_input, decode_predictions
from theano.sandbox import cuda
import utils
import importlib
importlib.reload(utils)
# In case we are going to use the TensorFlow backend we need to explicitly set the Theano image ordering
from keras import backend as K
K.set_image_dim_ordering('th')


from  vgg16bn import Vgg16BN
from utils import *

path = "data/fish/"
batch_size = 64

batches = get_batches(path+'train', batch_size=batch_size)
val_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)

(val_classes, trn_classes, val_labels, trn_labels,
    val_filenames, filenames, test_filenames) = get_classes(path)

raw_filenames = [f.split('/')[-1] for f in filenames]
raw_test_filenames = [f.split('/')[-1] for f in test_filenames]
raw_val_filenames = [f.split('/')[-1] for f in val_filenames]

sizes = [PIL.Image.open(path+'train/'+f).size for f in filenames]
id2size = list(set(sizes))
size2id = {o:i for i,o in enumerate(id2size)}


# trn = get_data(path+'train')
# val = get_data(path+'valid')
# test = get_data(path+'test')

# save_array(path+'results/trn.dat', trn)
# save_array(path+'results/val.dat', val)
# save_array(path+'results/test.dat', test)

trn = load_array(path+'results/trn.dat')
val = load_array(path+'results/val.dat')

test = load_array(path+'results/test.dat')
gen = image.ImageDataGenerator()

trn_sizes_orig = to_categorical([size2id[o] for o in sizes], len(id2size))
raw_val_sizes = [PIL.Image.open(path+'valid/'+f).size for f in val_filenames]
val_sizes = to_categorical([size2id[o] for o in raw_val_sizes], len(id2size))

trn_sizes = trn_sizes_orig-trn_sizes_orig.mean(axis=0)/trn_sizes_orig.std(axis=0)
val_sizes = val_sizes-trn_sizes_orig.mean(axis=0)/trn_sizes_orig.std(axis=0)

import ujson as json
anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']


def get_annotations():
    annot_urls = {
        '5458/bet_labels.json': 'bd20591439b650f44b36b72a98d3ce27',
        '5459/shark_labels.json': '94b1b3110ca58ff4788fb659eda7da90',
        '5460/dol_labels.json': '91a25d29a29b7e8b8d7a8770355993de',
        '5461/yft_labels.json': '9ef63caad8f076457d48a21986d81ddc',
        '5462/alb_labels.json': '731c74d347748b5272042f0661dad37c',
        '5463/lag_labels.json': '92d75d9218c3333ac31d74125f2b380a'
    }
    cache_subdir = os.path.abspath(os.path.join(path, 'annos'))
    url_prefix = 'https://kaggle2.blob.core.windows.net/forum-message-attachments/147157/'

    if not os.path.exists(cache_subdir):
        os.makedirs(cache_subdir)

    for url_suffix, md5_hash in annot_urls.items():
        fname = url_suffix.rsplit('/', 1)[-1]
        get_file(fname, url_prefix + url_suffix, cache_subdir=cache_subdir, md5_hash=md5_hash)

# 获取annotations
# get_annotations()

bb_json = {}
for c in anno_classes:
    if c == 'other':continue # no annotation file for "other" class
    j = json.load(open('{}annos/{}_labels.json'.format(path, c), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]

print(bb_json['img_04908.jpg'])

file2idx = {o:i for i,o in enumerate(raw_filenames)}
val_file2idx = {o:i for i,o in enumerate(raw_val_filenames)}
empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}
for f in raw_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox
for f in raw_val_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox

bb_params = ['height', 'width', 'x', 'y']
def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (224. / size[0])
    conv_y = (224. / size[1])
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb

trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, sizes)],
                   ).astype(np.float32)
val_bbox = np.stack([convert_bb(bb_json[f], s)
                   for f,s in zip(raw_val_filenames, raw_val_sizes)]).astype(np.float32)

def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def show_bb(i):
    bb = val_bbox[i]
    plot(val[i])
    plt.gca().add_patch(create_rect(bb))

# show_bb(0)
# -------------train-----------
model = vgg_ft_bn(8)

conv_layers, fc_layers = split_at(model, Convolution2D)
conv_model = Sequential(conv_layers)

conv_feat = conv_model.predict(trn)
conv_val_feat = conv_model.predict(val)
conv_test_feat = conv_model.predict(test)


p = 0.6
inp = Input(conv_layers[-1].output_shape[1:])
x = MaxPooling2D()(inp)
x = BatchNormalization(axis=1)(x)
x = Dropout(p/4)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p/2)(x)
x_bb = Dense(4, name='bb')(x)
x_class = Dense(8, activation='softmax', name='class')(x)

model = Model([inp], [x_bb, x_class])
model.compile(Adam(lr=0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'],
             loss_weights=[.001, 1.])
# print (conv_feat[0], trn_bbox[0], trn_labels[0])
model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=batch_size, nb_epoch=3,
             validation_data=(conv_val_feat, [val_bbox, val_labels]))

model.optimizer.lr = 1e-5
model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=batch_size, nb_epoch=10,
             validation_data=(conv_val_feat, [val_bbox, val_labels]))

pred = model.predict(conv_val_feat[0:10])
i = 6
bb = val_bbox[i]
bb_pred = pred[0][i]
class_pre = pred[1][i]
print("bb_pred----->", bb_pred)
print ("class_pred----->", class_pre)
def show_bb_pred(i):
    bb = val_bbox[i]
    bb_pred = pred[0][i]
    plt.figure(figsize=(6,6))
    plot(val[i])
    ax=plt.gca()
    ax.add_patch(create_rect(bb_pred, 'yellow'))
    ax.add_patch(create_rect(bb))


# show_bb_pred(6)
# 评估
model.evaluate(conv_val_feat, [val_bbox, val_labels])
model.save_weights(path+'models/bn_anno.h5')
# model.load_weights(path+'models/bn_anno.h5', by_name=True)
# # val = load_array(path+'results/val.dat')
# # conv_layers,fc_layers = split_at(model, Convolution2D)
# # conv_model = Sequential(conv_layers)
# # conv_val_feat = conv_model.predict(val)
# test_image_path = './data/fish/train/DOL/img_00165.jpg'
#
# img = image.load_img(test_image_path, target_size=(224, 224))
#
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# pred = model.predict(x)
# print(pred)
