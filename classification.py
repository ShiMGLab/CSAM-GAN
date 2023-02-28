import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout,Reshape,merge,Add,Concatenate
from keras.models import Model
from keras.layers import Dot
import tensorflow
from tensorflow.keras.optimizers import Adam,SGD,Adagrad
from sklearn.model_selection import StratifiedKFold, train_test_split
import xlrd
from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib
from sklearn.metrics import auc, f1_score, roc_curve, precision_score, recall_score,accuracy_score,fbeta_score
from lifelines.utils import concordance_index
from sklearn.model_selection import cross_val_score
from keras import regularizers
from keras.regularizers import l2
def ROC(label,predict):
    fpr, tpr, threshold = roc_curve(label, predict)
    AUC = auc(fpr, tpr)
    return AUC

def get_os_time():
    path = '/Users/limingna/PycharmProjects/p1/CBAM-GAN/KIRC/os.time.xlsx'
    data = xlrd.open_workbook(path)
    table = data.sheet_by_name('Sheet1')
    colum = table.ncols
    row = table.nrows
    patientID = []
    os_time = []
    for i in range(1,row):
        os_time.append(table.cell(i,1).value)
        patientID.append(table.cell(i, 0).value)
    return os_time
#
def load_gene():
    dataPath = '/Users/limingna/PycharmProjects/p1/CBAM-GAN/KIRC/KIRC.xlsx'
    data = xlrd.open_workbook(dataPath)
    table = data.sheet_by_name('Sheet1')
    colum = table.ncols
    row = table.nrows
    data_gene = []
    for i in range(row):
        data_gene.append(table.row_values(i))
    return data_gene
#
def get_label():
    dataPath = '/Users/limingna/PycharmProjects/p1/CBAM-GAN/KIRC/label.xlsx'
    data = xlrd.open_workbook(dataPath)
    table = data.sheet_by_name('Sheet1')
    colum = table.ncols
    row = table.nrows
    patientID = []
    label = []
    for i in range(1,row):
        label.append(table.cell(i, 1).value)
        patientID.append(table.cell(i, 0).value)
    return label
#
def get_state():
    path = '/Users/limingna/PycharmProjects/p1/CBAM-GAN/KIRC/os.state.xlsx'
    data = xlrd.open_workbook(path)
    table = data.sheet_by_name('Sheet1')
    colum = table.ncols
    row = table.nrows
    patientID = []
    os_state = []
    for i in range(1,row):
        os_state.append(table.cell(i,1).value)
        patientID.append(table.cell(i, 0).value)
    return os_state

def get_cv_data():
    dataPath = '/Users/limingna/PycharmProjects/p1/CBAM-GAN/KIRC/label1.xlsx'
    data = xlrd.open_workbook(dataPath)
    table = data.sheet_by_name('Sheet1')
    colum = table.ncols
    row = table.nrows
    patientID = []
    label = []
    for i in range(1, row):
        label.append(table.cell(i, 1).value)
        patientID.append(table.cell(i, 0).value)

    data_gene=load_gene()
    skf = StratifiedKFold(n_splits=5)
    i = 1
    train_index = []
    test_index = []
    for train_indx, test_indx in skf.split(data_gene, label):
        print(i, 'fold #####')
        train_index.append(train_indx)
        test_index.append(test_indx)
        print(test_indx)
        i += 1
    print('\n')
    print(test_index[0])
    np.save('/Users/limingna/PycharmProjects/p1/CBAM-GAN/KIRC/train_index.npy', train_index)
    np.save('/Users/limingna/PycharmProjects/p1/CBAM-GAN/KIRC/test_index.npy', test_index)
# if __name__ == '__main__':
#     get_cv_data()
def fusion():
    input_mirna = Input(shape=(200,), name='mirna_input')
    x = Dense(500, activation='relu', name='FC_1')(input_mirna)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', name='FC_5')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', name='FC_3')(x)
    x = Dropout(0.1)(x)
    # x = Dense(64, activation='relu', name='FC_2')(x)
    # x = Dropout(0.1)(x)
    x = Dense(32, activation='relu', name='FC_4')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(2, activation='softmax', name='output')(x)
    model = Model(inputs=input_mirna,outputs=outputs)
    return model

def fusion_test():
    path = '/Users/limingna/PycharmProjects/p1/CBAM-GAN/KIRC/'
    train_index = np.load(path + 'train_index.npy', allow_pickle=True, encoding='latin1')
    test_index = np.load(path + 'test_index.npy', allow_pickle=True, encoding='latin1')
    os_time = get_os_time()
    os_state = get_state()
    label = get_label()
    data_gene=load_gene()
    predict = []  # predict label
    ori_label = []  # original label
    ori_os_time = []
    ori_os_state = []
    predict_score = [] # predict scores

    for i in range(5):
        print('\n')
        print(i + 1, 'th fold ######')
        x_train_gene, x_test_gene = np.array(data_gene)[train_index[i]], np.array(data_gene)[test_index[i]]
        y_train_gene, y_test_gene = np.array(label)[train_index[i]], np.array(label)[test_index[i]]

        y_test_gene_onehot = to_categorical(y_test_gene)  # 将类别标签向量转换成二进制矩阵类型
        y_train_gene_onehot = to_categorical(y_train_gene)
        y_test_gene_os_time = np.array(os_time)[test_index[i]]
        y_test_gene_os_state = np.array(os_state)[test_index[i]]

        fusion_model = fusion()

        fusion_model.summary()

        adam = Adam(lr=0.00004, beta_1=0.8, beta_2=0.999, epsilon=1e-08, decay=0.01)

        loss_function = 'binary_crossentropy'

        fusion_model.compile(loss=loss_function, optimizer=adam, metrics=['accuracy'])
        history = fusion_model.fit(x_train_gene , y_train_gene_onehot, epochs=600, batch_size=16,
                               validation_split=0.2)
        ori_label.extend(y_test_gene)
        ori_os_time.extend(y_test_gene_os_time)
        ori_os_state.extend(y_test_gene_os_state)
        y_pred = fusion_model.predict(x_test_gene)
        predict.extend(np.argmax(y_pred, 1))
        predict_score.extend(y_pred[:, 1].tolist())




    label_path = '/Users/limingna/PycharmProjects/p1/CBAM-GAN/KIRC/'
    np.save(label_path + 'inter_intra_ori_label.npy', ori_label)
    np.save(label_path + 'inter_intra_predict_score.npy',predict_score)
    np.save(label_path + 'inter_intra_predict_label.npy', predict)

    #print('\n')


    auc = ROC(ori_label, predict_score)
    acc=accuracy_score(ori_label,predict)
    f1=f1_score(ori_label,predict,average="weighted")
    precision=precision_score(ori_label,predict,average="weighted")
    recall=recall_score(ori_label,predict,average="weighted")
    #
    print('\n')
    print('auc=', auc)
    print('acc=', acc)
    print('f1=', f1)
    print('precision=', precision)
    print('recall=', recall)
    return auc,acc,f1,precision,recall

if __name__ == '__main__':
    fusion_test()

