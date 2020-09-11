# --coding:utf-8--
import os
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 訓練資料
train_dir = 'E:/google drive/iii_course/HOT/img_distinct/train/outside'
test_dir = 'E:/google drive/iii_course/HOT/img_distinct/test/outside'

# 分類數
class_numbers = len(os.listdir(train_dir))
# print(class_numbers)
# 設定網路結構, 使用在 imagenet 上訓練的參數作為初始參數
# include_top=False 不使用預先的分類器
Backbone = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

Backbone.trainable = True  # 設定所有層為可訓練

set_trainable = False   # 凍結布林變數

# 249層以前全部凍結, 只微調訓練249層之後
for layer in Backbone.layers[:249]:
   layer.trainable = False
for layer in Backbone.layers[249:]:
   layer.trainable = True

model = Sequential()

# 訓練自己的分類器
model.add(Backbone)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # class_numbers=4 總共有四種葉子

model.summary()

# 資料增強增加學習樣本
train_datagen = ImageDataGenerator(
  rescale=1./255,    # 指定將影象像素縮放到0~1之間
  # preprocessing_function=preprocess_input,
  rotation_range=30,  # 角度值，0~180，影象旋轉
  width_shift_range=0.45,  # 水平平移，相對總寬度的比例
  fill_mode='wrap',
  zoom_range=0.6,
  shear_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 訓練資料與測試資料  # 分類超過兩類 使用categorical, 若分類只有兩類使用binary
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(299, 299),
batch_size=32,
class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(299, 299),
batch_size=32,
class_mode='categorical',
shuffle=False)

print('='*30)
print(train_generator.class_indices)
print('='*30)

checkpoint = ModelCheckpoint('model_testtest.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
estop = EarlyStopping(monitor='val_loss', patience=3)

# 使用批量生成器 訓練模型
H = model.fit_generator(
train_generator,
steps_per_epoch=train_generator.samples/train_generator.batch_size,  # 每一回合從訓練集中抓取訓練樣本訓練, 總共訓練30次
epochs=10
,  # 一共訓練回合
validation_data=test_generator,
validation_steps=test_generator.samples/test_generator.batch_size,
callbacks=[checkpoint, estop],
verbose=1
)

epochs = range(len(H.history['acc']))

plt.figure()
plt.plot(epochs, H.history['acc'], 'b', label='Training acc')
plt.plot(epochs, H.history['val_acc'], 'r', label='validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('acc_iv3.png')
plt.show()

plt.figure()
plt.plot(epochs, H.history['loss'], 'b', label='Training loss')
plt.plot(epochs, H.history['val_loss'], 'r', label='validation loss')
plt.title('Training and validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_iv3.png')

plt.show()

del model
