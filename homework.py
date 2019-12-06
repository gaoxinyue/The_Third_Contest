import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

#卷积函数
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME',name='conv2d')
#池化函数
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='maxpool')
#权重生成函数
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
#偏置生成函数
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

import string
#建立一个字典，将大小写字母转成数字
label_dict={}
characters = string.digits + string.ascii_letters
for i,x in enumerate(characters):
    label_dict[x]=i
#{'0:0,'1:1,'2:2,'3:3,'4:4,'5:5,'6:6,'7:7,'8:8,'9:9,'a:10,'b:11,'c:12,'d:13,'e:14,'f:15,'g:16,'h:17,'i:18,'j:19,'k:20,'l:21,'m:22,'n:23,'o:24,'p:25,'q:26,'r:27,'s:28,'t:29,'u:30,'v:31,'w:32,'x:33,'y:34,'z:35,'A:36,'B:37,'C:38,'D:39,'E:40,'F:41,'G:42,'H:43,'I:44,'J:45,'K:46,'L:47,'M:48,'N:49,'O:50,'P:51,'Q:52,'R:53,'S:54,'T:55,'U:56,'V:57,'W:58,'X:59,'Y:60,'Z:61}

#对验证码图片进行处理，进行灰度转化、二值化、高斯滤波后再次进行二值化，最终得到需要的输入数据
imagepath='train/'
def get_x_data(filepath):
    data=[]
    image_name=[]
    for filename in os.listdir(filepath):
        image_name.append(filename)
        im=cv2.imread(filepath+filename)
        #灰度
        im_gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #二值化
        ret, im_inv = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY_INV)
        #高斯滤波去除颗粒噪声
        kernel = 1/16*np.array([[1,2,1], [2,4,2], [1,2,1]])
        im_blur = cv2.filter2D(im_inv,-1,kernel)
        #再次进行二值化
        ret, im_res = cv2.threshold(im_blur,127,255,cv2.THRESH_BINARY)
        im_data=im_res.reshape(1,30*150)/255
        image_data=im_data.tolist()
        data.append(image_data)
    return data,image_name
data,image_name=get_x_data(imagepath)

def get_y_data(imagename):
    label_data=[]
    for x in imagename:
        labels=np.zeros([5,62])
        result1,result2=x.split('.')
        for i,chara in enumerate(result1):
            col_index=label_dict[chara]
            labels[i,col_index]=1
        label_data.append(labels.reshape(1,-1).tolist())   
    return label_data
label_data=get_y_data(image_name)

#再次进行处理
m,k,n=np.array(data).shape
new_data=np.array(data).reshape(m,n)
m,k,n=np.array(label_data).shape
new_label=np.array(label_data).reshape(m,n)

#将整个数据集划分成训练集和测试集
train_x,test_x,train_y,test_y=train_test_split(new_data,new_label,test_size=0.2)

xs=tf.placeholder(tf.float32,shape=[None,30*150],name='x')
ys=tf.placeholder(tf.float32,shape=[None,62*5],name='label-input')
keep_prob = tf.placeholder(tf.float32, name='keep-prob')

#搭建输入层—>隐藏层1—>隐藏层2—>隐藏层3—>全连接层—>输出层的神经网络结构
#第一层卷积，卷积核为5*5
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
x_image=tf.reshape(xs,[-1,30,150,1],name='x-input')
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
pool_1=max_pool(h_conv1)
pool_1=tf.nn.dropout(pool_1,keep_prob)

#第二层卷积
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(pool_1,w_conv2)+b_conv2)
pool_2=max_pool(h_conv2)
pool_2=tf.nn.dropout(pool_2,keep_prob)

#第三层卷积
w_conv3=weight_variable([5,5,64,64])
b_conv3=bias_variable([64])
h_conv3=tf.nn.relu(conv2d(pool_2,w_conv3)+b_conv3)
pool_3=max_pool(h_conv3)
pool_3=tf.nn.dropout(pool_3,keep_prob)

#全连接层
w_fc_1=weight_variable([4*19*64,1024])
b_fc_1=bias_variable([1024])

pool_2_flat=tf.reshape(pool_3,[-1,4*19*64])
h_fc_1=tf.nn.relu(tf.matmul(pool_2_flat,w_fc_1)+b_fc_1)
h_fc_prob=tf.nn.dropout(h_fc_1,keep_prob)

#输出层
w_fc_2=weight_variable([1024,62*5])
b_fc_2=bias_variable([62*5])
output = tf.add(tf.matmul(h_fc_prob, w_fc_2),b_fc_2)

def get_next_batch(data,size):
    m,n=data.shape
    index=[]
    for i in range(m):
        if i % size==0:
            index.append(i)
    index.append(m-1)
    return index

#设置损失函数以及优化器使得神经网络可以反馈调节
#定义损失以及优化器
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=output))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

predict = tf.reshape(output, [-1, 5, 62], name='predict')
labels = tf.reshape(ys, [-1, 5, 62], name='labels')

#计算准确率
predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))

#训练模型
#每把所有的图片填到神经网络后，计算一次神经网络在测试集上的准确率，如果大于0.5，就终止训练，保存参数
path = 'model/'
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        index=get_next_batch(train_x,64)
        order=np.random.permutation(train_x.shape[0])
        for i in range(len(index)-1):
            _,loss_=sess.run([optimizer,loss],feed_dict={xs:train_x[order[index[i]:index[i+1]],:],ys:train_y[order[index[i]:index[i+1]],:],keep_prob:0.75})
        acc=sess.run(accuracy,feed_dict={xs:test_x,ys:test_y,keep_prob:1.0})
        print('step=%d,iter=%d,accuracy=%f'%(step,i,acc))
        if acc>0.5:
            saver.save(sess,path+'digit_captcha.model',global_step=step)
            break

#对test中的验证码图片你进行验证
validpath='test/'
validata,validname=get_x_data(validpath)

m,k,n=np.array(validata).shape
valid_data=np.array(validata).reshape(m,n)

labeldata=get_y_data(validname)
m,k,n=np.array(labeldata).shape
label_data=np.array(labeldata).reshape(m,n)

graph=tf.get_default_graph()
input_holder=graph.get_tensor_by_name('x:0')
label_holder=graph.get_tensor_by_name('label-input:0')
keep_prob_holder=graph.get_tensor_by_name('keep-prob:0')
predict_max_idx=graph.get_tensor_by_name('predict_max_idx:0')
labels_max_idx =graph.get_tensor_by_name('labels_max_idx:0')
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint(path))
    predict=sess.run(predict_max_idx,feed_dict={input_holder:valid_data,keep_prob_holder:1.0})
    labels= sess.run(labels_max_idx,feed_dict={label_holder:label_data})

digit_character = dict(zip(label_dict.values(), label_dict.keys()))
predcharacter=[]
for i in range(len(predict)):
    yzm=''.join(str(digit_character[x]) for x in predict[i])
    predcharacter.append(yzm)

# 将得到的list结果保存到csv文件中
m, = np.shape(predcharacter)
data = pd.DataFrame(predcharacter, index=range(0, m))
data.to_csv('train_labels.csv')




