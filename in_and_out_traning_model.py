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

# �V�m���
train_dir = 'E:/google drive/iii_course/HOT/img_distinct/train/outside'
test_dir = 'E:/google drive/iii_course/HOT/img_distinct/test/outside'

# ������
class_numbers = len(os.listdir(train_dir))
# print(class_numbers)
# �]�w�������c, �ϥΦb imagenet �W�V�m���ѼƧ@����l�Ѽ�
# include_top=False ���ϥιw����������
Backbone = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

Backbone.trainable = True  # �]�w�Ҧ��h���i�V�m

set_trainable = False   # �ᵲ���L�ܼ�

# 249�h�H�e�����ᵲ, �u�L�հV�m249�h����
for layer in Backbone.layers[:249]:
   layer.trainable = False
for layer in Backbone.layers[249:]:
   layer.trainable = True

model = Sequential()

# �V�m�ۤv��������
model.add(Backbone)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # class_numbers=4 �`�@���|�ظ��l

model.summary()

# ��ƼW�j�W�[�ǲ߼˥�
train_datagen = ImageDataGenerator(
  rescale=1./255,    # ���w�N�v�H�����Y���0~1����
  # preprocessing_function=preprocess_input,
  rotation_range=30,  # ���׭ȡA0~180�A�v�H����
  width_shift_range=0.45,  # ���������A�۹��`�e�ת����
  fill_mode='wrap',
  zoom_range=0.6,
  shear_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# �V�m��ƻP���ո��  # �����W�L���� �ϥ�categorical, �Y�����u�������ϥ�binary
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

# �ϥΧ�q�ͦ��� �V�m�ҫ�
H = model.fit_generator(
train_generator,
steps_per_epoch=train_generator.samples/train_generator.batch_size,  # �C�@�^�X�q�V�m��������V�m�˥��V�m, �`�@�V�m30��
epochs=10
,  # �@�@�V�m�^�X
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
