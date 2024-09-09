import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


def model(data_path):
    data=pd.read_csv(data_path,header=0)
    y_use_col=['travel_purpose']
    x_use_col=['sex','age','social_class','local_size','starting_point','ending_point','travel_time','start_time','end_time','waiting_time']
    sample_index=range(0,data.shape[0],5)
    y=data.loc[sample_index,y_use_col]
    X=data.loc[sample_index,x_use_col]
    y=np.array(y)
    X=np.array(X)

    # 对特征进行标签编码（Label Encoding）
    label_encoder = LabelEncoder()
    for i in range(X.shape[1]):
        X[:, i] = label_encoder.fit_transform(X[:, i])
    print(X.shape)
    print(X)

    # 对标签进行标签编码
    y = label_encoder.fit_transform(y)

    # 对特征进行独热编码（One-Hot Encoding）
    onehot_encoder = OneHotEncoder()
    X = onehot_encoder.fit_transform(X).toarray()

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_test_class=np.unique(y_test)
    y_train_class=np.unique(y_train)

    # 将标签转换为独热编码格式
    y_train = to_categorical(y_train, num_classes=len(y_train_class))
    y_test = to_categorical(y_test, num_classes=len(y_test_class))


    # 构建ANN模型
    model = Sequential()

    # 输入层和第一层隐藏层
    model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))

    # 第二层隐藏层
    model.add(Dense(units=32, activation='relu'))

    # 输出层
    model.add(Dense(units=8, activation='softmax'))

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    history=model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


    # 预测结果
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,linewidths=0.1, linecolor='black', cbar=True)  # 增加边框线
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(y_test_class)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

if __name__=='__main__':
    path_read = 'test_Local_Small.csv'
    model(path_read)
