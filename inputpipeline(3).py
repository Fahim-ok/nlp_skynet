#!/usr/bin/env python
# coding: utf-8

# In[1]:


data = open("train.txt", encoding="UTF-8")


# In[2]:


d_list = []
for i in range(207197):
    d_list.append(data.readline())


# In[3]:


import re
d_list = [item.replace("\n", "") for item in d_list]


# In[4]:


new_dl = []
buffer = []
for i in range(len(d_list)):
    if d_list[i] == "":
        new_dl.append(buffer)
        buffer = []
    else:
        buffer.append(d_list[i])
new_dl.append(buffer)


# In[5]:


for i in range(len(new_dl)):
    new_dl[i] = [(item.split(" _ _ ")[0], item.split(" _ _ ")[1]) for item in new_dl[i]]


# In[6]:


#sentences -> new_dl


# # Analysis

# In[7]:


largest_sen = max(len(sen) for sen in new_dl)


# In[8]:


"Maximum sentence length "+str(largest_sen)


# In[9]:


import matplotlib.pyplot as plt
plt.hist([len(sen) for sen in new_dl], bins = 60)
plt.title("Histogram Plot for the train set")
plt.xlabel("Number of Words")
plt.ylabel("Number of Sentences")
plt.show()


# In[10]:


labels = []
for i in range(len(new_dl)):
    for item in new_dl[i]:
        labels.append(item[1])


# In[11]:


import pandas as pd
names = pd.DataFrame(labels).value_counts().index.values


# In[12]:


values = pd.DataFrame(labels).value_counts().values


# In[13]:


plt.figure(figsize=(12,12))
plt.bar([i[0] for i in names][1:], values[1:])


# # Preprocessing

# In[14]:


sequences = []
labels = []
for i in range(len(new_dl)):
    seq = []
    lab = []
    for item in new_dl[i]:
        seq.append(item[0])
        lab.append(item[1])
    sequences.append(seq)
    labels.append(lab)


# In[15]:


get_ipython().system('pip install tensorflow==2.11')


# In[16]:


import tensorflow as tf


# In[17]:


# vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
#     max_tokens=1000,
#     output_mode='int',
#     output_sequence_length=40)


# In[18]:


# import numpy as np
# vectorize_layer.adapt(np.array(" ".join(sequences)))
wordVectors = []
# seq_labels = []
for i in range(len(new_dl)):
    wordVectors.append(" ".join(sequences[i]))


# In[19]:


tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=1000,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=False,
    split=' ',
    char_level=False,
    oov_token=None,
    analyzer=None,
)


# In[20]:


tokenizer.fit_on_texts(wordVectors)


# In[21]:


X = tokenizer.texts_to_sequences(wordVectors)
y = labels

MAX_LEN = largest_sen+10



all_labels = []
for i in range(len(new_dl)):
    for item in new_dl[i]:
        all_labels.append(item[1])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(all_labels)
print(le.classes_)


for i in range(len(y)):
    y[i] = le.transform(y[i])


# In[22]:


X_train = tf.keras.utils.pad_sequences(X, 
                            maxlen=MAX_LEN, 
                            dtype='int32', 
                            padding="post",
                            truncating="post", 
                            value=0)

y_train = tf.keras.utils.pad_sequences(y, 
                            maxlen=MAX_LEN, 
                            dtype='int32', 
                            padding="post",
                            truncating="post", 
                            value=len(le.classes_))


# In[23]:


path = "dev.txt"

def preprocess_test_data(file_path):
    test_data = open(file_path, encoding="UTF-8")
    d_list = []
    for i in range(11133):
        d_list.append(test_data.readline())
    d_list = [item.replace("\n", "") for item in d_list]
    new_dl = []
    buffer = []
    for i in range(len(d_list)):
        if d_list[i] == "":
            new_dl.append(buffer)
            buffer = []
        else:
            buffer.append(d_list[i])
    new_dl.append(buffer)
    for i in range(len(new_dl)):
        new_dl[i] = [(item.split(" _ _ ")[0], item.split(" _ _ ")[1]) for item in new_dl[i]]
    sequences = []
    labels = []
    for i in range(len(new_dl)):
        seq = []
        lab = []
        for item in new_dl[i]:
            seq.append(item[0])
            lab.append(item[1])
        sequences.append(seq)
        labels.append(lab)
    wordVectors = []
    for i in range(len(new_dl)):
        wordVectors.append(" ".join(sequences[i]))
    X = tokenizer.texts_to_sequences(wordVectors)
    y = labels
    for i in range(len(y)):
        y[i] = le.transform(y[i])
    X_test = tf.keras.utils.pad_sequences(X, 
                            maxlen=MAX_LEN, 
                            dtype='int32', 
                            padding="post",
                            truncating="post", 
                            value=0)

    y_test = tf.keras.utils.pad_sequences(y, 
                            maxlen=MAX_LEN, 
                            dtype='int32', 
                            padding="post",
                            truncating="post", 
                            value=len(le.classes_))
    return X_test, y_test


# In[24]:


X_test, y_test = preprocess_test_data(path)


# In[25]:


print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)


# In[26]:


import numpy as np
X_combined = np.append(X_train, X_test, axis = 0)
y_combined = np.append(y_train, y_test, axis = 0)


# In[27]:


print("X_test: ", X_combined.shape)
print("y_test: ", y_combined.shape)


# # Model Deployment

# In[28]:


get_ipython().system('pip install transformers')


# In[29]:


from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer_ = AutoTokenizer.from_pretrained("sagorsarker/mbert-bengali-ner")
model = AutoModelForTokenClassification.from_pretrained("sagorsarker/mbert-bengali-ner")


# In[30]:


from torch import nn
import torch.nn.functional as F


# In[31]:


model.classifier = nn.Linear(in_features=768, out_features=13)
# model = nn.Sequential(
#     model,
#     F.softmax(13)
# )


# In[ ]:





# In[32]:


# bnlptk = BasicTokenizer()
# X_combined_string_format = 
# bnlptk.tokenize(tokenizer.sequences_to_texts(X_combined)[0])


# In[ ]:





# In[33]:


# tokenizer.sequences_to_texts(X_combined)


# In[34]:


X_combined_string_format = tokenizer_(tokenizer.sequences_to_texts(X_combined), padding="max_length", max_length=63 )["input_ids"]


# In[35]:


# X_combined_string_format


# In[ ]:





# In[36]:


from torch.utils.data import (
    Dataset,
    DataLoader,
)

class NER_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self,):
        return self.X.shape[0]
    def __getitem__(self, index):
        sample_x = self.X[index]
        sample_y = self.y[index]
        return (sample_x, sample_y)

dataset = NER_Dataset(X =np.array(X_combined_string_format), y = y_combined)


# In[37]:


batch_size = 32
import torch
train_set, test_set = torch.utils.data.random_split(dataset, [15301, 801])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# In[38]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[39]:


num_epochs = 100
learning_rate = 1e-3


# In[40]:


import torch.optim as optim 
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)


# In[ ]:


def train_loop(model, df_train, df_val):

#     train_dataset = DataSequence(df_train)
#     val_dataset = DataSequence(df_val)

#     train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)
    train_dataloader = df_train
    val_dataloader = df_val

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][train_label[i] != -100]
              label_clean = train_label[i][train_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_train += acc
              total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][val_label[i] != -100]
              label_clean = val_label[i][val_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_val += acc
              total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')

LEARNING_RATE = 5e-3
EPOCHS = 5
BATCH_SIZE = 2
from tqdm import tqdm

train_loop(model, train_loader, test_loader)


# In[ ]:


def evaluate(model, df_test):

    test_dataset = DataSequence(df_test)

    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0.0

    for test_data, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_data['attention_mask'].squeeze(1).to(device)

            input_id = test_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, test_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][test_label[i] != -100]
              label_clean = test_label[i][test_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_test += acc

    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {total_acc_test / len(df_test): .3f}')


evaluate(model, df_test)


# In[ ]:





# In[ ]:




