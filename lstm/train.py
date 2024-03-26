import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

# 读取CSV文件
df = pd.read_excel('spatiotemporal_corpus.xlsx')

# 转换分类标签为数字
labels = ['Time Interval Query', 'Time Point Query', 'Spatial Range Query', 'Time Range Query', \
          'Spatio-temporal Range Query', 'Spatial Nearest Neighbor Query', 'Moving Objects Nearest Neighbor Query', \
          'Spatial Join Query', 'Spatio-temporal Join Query', 'Similarity Query', 'Spatial Basic-distance Query', \
          'Spatial Basic-direction Query', 'Spatial Basic-length Query', 'Spatial Basic-area Query', 'Spatial Aggregation-count Query', \
          'Spatial Aggregation-sum Query', 'Spatial Aggregation-max Query', 'Temporal Aggregation Query', 'Normal Basic Query']
df['cat'] = df['cat'].apply(lambda x: labels.index(x))

# 分割数据集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 构建词汇表和文本嵌入向量
words = df['review'].str.split(expand=True).unstack().value_counts()
word_to_idx = {word: idx + 2 for idx, word in enumerate(words.index)}
word_to_idx['<PAD>'] = 0
word_to_idx['<UNK>'] = 1
max_length = df['review'].str.split().apply(len).max()
train_vectors = np.stack(train_df['review'].apply(lambda x: np.array([word_to_idx.get(word, 1) for word in x.split()] + [0]*(max_length-len(x.split())))))
test_vectors = np.stack(test_df['review'].apply(lambda x: np.array([word_to_idx.get(word, 1) for word in x.split()] + [0]*(max_length-len(x.split())))))

# 将数据转换为Tensor
train_labels = torch.tensor(train_df['cat'].values)
test_labels = torch.tensor(test_df['cat'].values)
train_vectors = torch.LongTensor(train_vectors)
test_vectors = torch.LongTensor(test_vectors)

# 定义LSTMCNN模型
class LSTMCNN(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size, num_classes, num_filters=100, kernel_sizes=[3,4,5]):
        super(LSTMCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.convolution_layers = nn.ModuleList([
            nn.Conv1d(in_channels=2*hidden_size, out_channels=num_filters, kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_output, (h_n, c_n) = self.lstm(x)
        x = lstm_output.permute(0, 2, 1)
        convolution_outputs = []
        for convolution in self.convolution_layers:
            convolution_output = convolution(x)
            convolution_output = nn.functional.relu(convolution_output)
            max_pool_output = nn.functional.max_pool1d(convolution_output, kernel_size=convolution_output.size(2))
            convolution_outputs.append(max_pool_output)
        concatenated_tensor = torch.cat(convolution_outputs, dim=1)
        flatten_tensor = concatenated_tensor.view(concatenated_tensor.size(0), -1)
        dropout_output = self.dropout(flatten_tensor)
        logits = self.linear(dropout_output)
        return logits

# 定义交叉验证的折数
num_folds = 5

# 定义交叉验证
skf = StratifiedKFold(n_splits=num_folds, shuffle=True)

# 定义参数组合
parameters = [
    {'embedding_size': 100, 'hidden_size': 64, 'num_filters': 100, 'kernel_sizes': [3, 4, 5]},
    {'embedding_size': 200, 'hidden_size': 128, 'num_filters': 200, 'kernel_sizes': [3, 4, 5]},
    {'embedding_size': 150, 'hidden_size': 100, 'num_filters': 150, 'kernel_sizes': [3, 4, 5]},
    {'embedding_size': 100, 'hidden_size': 128, 'num_filters': 150, 'kernel_sizes': [3, 4, 6]},
    {'embedding_size': 200, 'hidden_size': 64, 'num_filters': 100, 'kernel_sizes': [4, 5, 6]},
    # 其他参数组合...
]

# 定义存储性能指标的列表
accuracies = []

# 进行交叉验证
for parameter in parameters:
    fold_accuracies = []
    for train_index, val_index in skf.split(train_vectors, train_labels):
        # 划分训练集和验证集
        train_vectors_fold, val_vectors_fold = train_vectors[train_index], train_vectors[val_index]
        train_labels_fold, val_labels_fold = train_labels[train_index], train_labels[val_index]

        # 创建模型
        model = LSTMCNN(len(word_to_idx), embedding_size=parameter['embedding_size'], hidden_size=parameter['hidden_size'], num_classes=len(labels), num_filters=parameter['num_filters'], kernel_sizes=parameter['kernel_sizes'])

        # 定义优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 训练模型
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            logits = model(train_vectors_fold)
            loss = criterion(logits, train_labels_fold)
            loss.backward()
            optimizer.step()

        # 在验证集上评估模型性能
        model.eval()
        with torch.no_grad():
            logits = model(val_vectors_fold)
            predicted_classes = torch.argmax(logits, dim=1)
            accuracy = (predicted_classes == val_labels_fold).float().mean().item()
            fold_accuracies.append(accuracy)

    # 计算平均性能
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    accuracies.append(avg_accuracy)

# 打印不同参数组合的性能指标
for i, parameter in enumerate(parameters):
    print(f"Parameter combination {i+1}: Accuracy = {accuracies[i]:.4f}")

# 选择最佳参数组合
best_parameter_index = accuracies.index(max(accuracies))
best_parameter = parameters[best_parameter_index]
print(f"Best parameter combination: {best_parameter}")

# 使用最佳参数组合重新训练模型并评估性能
model = LSTMCNN(len(word_to_idx), embedding_size=best_parameter['embedding_size'], hidden_size=best_parameter['hidden_size'], num_classes=len(labels), num_filters=best_parameter['num_filters'], kernel_sizes=best_parameter['kernel_sizes'])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    logits = model(train_vectors)
    loss = criterion(logits, train_labels)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    logits = model(test_vectors)
    predicted_classes = torch.argmax(logits, dim=1)
    accuracy = (predicted_classes == test_labels).float().mean().item()
    print(f"Test accuracy with best parameter combination: {accuracy:.4f}")

# 存储模型和相关信息到文件中
torch.save(model.state_dict(), 'model.pth')
torch.save(word_to_idx, 'word_to_idx.pth')
torch.save(max_length, 'max_length.pth')
