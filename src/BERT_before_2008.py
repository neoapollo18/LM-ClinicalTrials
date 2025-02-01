import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import wandb
import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from torch.autograd import Variable
from sklearn.metrics import f1_score
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.set_float32_matmul_precision('high')


sweep_config = {
    'name': 'Before2008OneHotSweep',
    'method': 'random',  # or 'grid' or 'bayes'
    'metric': {
      'name': 'val-loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-3, 1e-4, 5e-5, 5e-4]
        },
        'dropout_rate': {
            'values': [0.4, 0.5, 0.6]
        },
        'batch_size': {
            'values': [64, 128]
        },
        # Add other hyperparameters here
    }
}
sweep_id = wandb.sweep(sweep_config, project="Clinical Trials")


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
df = pd.read_csv('TrialData/before_2008.csv')

#Preprocessing Function
def train():
    def preprocess_bert(data, labels, textual_fields, categorical_fields, maximum_length):
        in_ids = []
        att_masks = []
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        onehot_encoder = OneHotEncoder(sparse=False)
        
        onehot_encoded = onehot_encoder.fit_transform(data[categorical_fields])
        
        for i in tqdm(range(len(data))):
            concatenated_text = ' '.join([str(data[field][i]) for field in textual_fields])
            
            encoded = tokenizer.encode_plus(
                concatenated_text,
                add_special_tokens = True,
                padding = 'max_length',
                return_attention_mask = True,
                max_length = maximum_length,
                truncation=True
            )
            in_ids.append(encoded['input_ids'])
            att_masks.append(encoded['attention_mask'])
        
        return np.array(in_ids), np.array(att_masks), labels, onehot_encoded


    #Model

    class CombinedModel(nn.Module):
        def __init__(self, num_extra_features, num_classes, dropout_rate=0.5):
            super(CombinedModel, self).__init__()
            
            self.bert_model = BertForSequenceClassification.from_pretrained(
                "bert-base-cased",
                num_labels=256,  
                output_attentions=False,
                output_hidden_states=False,
            )        
            
            # for p in self.bert_model.bert.parameters():
            #     p.requires_grad = False

            # hidden layers and dropout layers
            self.fc1 = nn.Linear(256 + num_extra_features, 256)
            self.dropout1 = nn.Dropout(dropout_rate)
            
            self.fc2 = nn.Linear(256, 128)
            self.dropout2 = nn.Dropout(dropout_rate)
            
            # self.fc3 = nn.Linear(256, 128)
            # self.dropout3 = nn.Dropout(0.5)
            
            self.fc4 = nn.Linear(128, num_classes)

        def forward(self, input_ids, attention_mask, extra_features):
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits 
            
            combined_input = torch.cat((logits, extra_features), dim=1)
            
            x = F.relu(self.fc1(combined_input))
            x = self.dropout1(x)
            
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            
            # x = F.relu(self.fc3(x))
            # x = self.dropout3(x)
            
            x = self.fc4(x)
            
            return x
    




    # Preprocessing and Splitting
    wandb.init(config=sweep_config)

    bins = [0, 100, 500, float('inf')]
    labels = ['Low', 'Medium', 'High']
    df['EnrollmentCategory'] = pd.cut(df['StudyEnrollmentCount'], bins=bins, labels=labels)


    textual_fields = ['BriefSummary', 'DetailedDescription', 'EligibilityCriteria']
    categorical_fields = ["OrgFullName", "LeadSponsorName", "Phase", "Condition", "InterventionType", "DesignPrimaryPurpose", "DesignMasking", "EnrollmentCategory"]


    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['Status'].values)

    in_ids, att_masks, labels, onehot_encoded = preprocess_bert(df, df['Status'].values, textual_fields, categorical_fields, maximum_length=128)


    unique_drugs = df['InterventionName'].unique().tolist()
    random.shuffle(unique_drugs)

    train_size = int(0.8 * len(unique_drugs))
    val_size = int(0.1 * len(unique_drugs))

    train_drugs = unique_drugs[:train_size]
    val_drugs = unique_drugs[train_size:train_size + val_size]
    test_drugs = unique_drugs[train_size + val_size:]


    train_df = df[df['InterventionName'].isin(train_drugs)].reset_index(drop=True)
    val_df = df[df['InterventionName'].isin(val_drugs)].reset_index(drop=True)
    test_df = df[df['InterventionName'].isin(test_drugs)].reset_index(drop=True)


    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(df[categorical_fields])  # Fit only on training data

    train_onehot_encoded = one_hot_encoder.transform(train_df[categorical_fields])
    val_onehot_encoded = one_hot_encoder.transform(val_df[categorical_fields])
    test_onehot_encoded = one_hot_encoder.transform(test_df[categorical_fields])

    # if not os.path.exists(/path/to/saved/data.pt)
    # do the stuff to process
    # else
    # load the data from /path/to/saved/data.pt

    train_in_ids, train_att_masks, train_labels, _ = preprocess_bert(train_df, train_df['Status'].values, textual_fields, categorical_fields, maximum_length=128)
    val_in_ids, val_att_masks, val_labels, _ = preprocess_bert(val_df, val_df['Status'].values, textual_fields, categorical_fields, maximum_length=128)
    test_in_ids, test_att_masks, test_labels, _ = preprocess_bert(test_df, test_df['Status'].values, textual_fields, categorical_fields, maximum_length=128)

    train_inputs = torch.tensor(train_in_ids)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_att_masks)
    train_features = torch.tensor(train_onehot_encoded).float()

    validation_inputs = torch.tensor(val_in_ids)
    validation_labels = torch.tensor(val_labels)
    validation_masks = torch.tensor(val_att_masks)
    validation_features = torch.tensor(val_onehot_encoded).float()

    test_inputs = torch.tensor(test_in_ids)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_att_masks)
    test_features = torch.tensor(test_onehot_encoded).float()
    
    learning_rate = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    dropout_rate = wandb.config.dropout_rate
    
    train_data = TensorDataset(train_inputs, train_masks, train_features, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=32)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_features, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=256, num_workers=32)

    test_data = TensorDataset(test_inputs, test_masks, test_features, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_model = CombinedModel(num_extra_features=train_onehot_encoded.shape[1], num_classes=2, dropout_rate=dropout_rate)
    combined_model = combined_model.to(device)

    optimizer = AdamW(combined_model.parameters(), lr = learning_rate,)

    epochs = 40
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    # total_steps = len(train_dataloader) * epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer, 
    #                                             num_warmup_steps = 0, 
    #                                             num_training_steps = total_steps)

    all_training_losses = []
    all_validation_losses = []




    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 3
    # Model Training

    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        total_train_loss = 0
        combined_model.train()

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_features = batch[2].to(device)
            b_labels = batch[3].to(device)

            optimizer.zero_grad()
            
            logits = combined_model(b_input_ids, b_input_mask, b_features)
            # class_weights = compute_class_weight('balanced', classes=np.unique(train_labels.cpu().numpy()), y=train_labels.cpu().numpy())
            # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
            # Compute the loss
            loss = nn.CrossEntropyLoss()(logits, b_labels)

            total_train_loss += loss.item()
            all_training_losses.append(loss.item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(combined_model.parameters(), 1.0)
            optimizer.step()
            wandb.log({"loss": loss})
        scheduler.step()
        



        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")
        combined_model.eval()


        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        combined_model.eval()


        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_features = batch[2].to(device)
            b_labels = batch[3].to(device)

            with torch.no_grad():        
                logits = combined_model(b_input_ids, b_input_mask, b_features)
                loss = nn.CrossEntropyLoss()(logits, b_labels)
                
            wandb.log({"val-loss": loss})
            eval_loss += loss.item()
            all_validation_losses.append(loss.item())

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

        
            eval_accuracy += accuracy_score(np.argmax(logits, axis=1).flatten(), label_ids.flatten())
            nb_eval_steps += 1
        avg_val_loss = eval_loss / len(validation_dataloader)
        print(f"Average validation loss: {avg_val_loss}")
        print(" Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(combined_model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            break


    probabilities = []
    combined_model.eval()
    predictions , true_labels = [], []


    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_features = batch[2].to(device)
        b_labels = batch[3].to(device)
        
        with torch.no_grad():
            logits = combined_model(b_input_ids, b_input_mask, b_features)
            loss = nn.CrossEntropyLoss()(logits, b_labels)

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
    
        predictions.append(logits)
        true_labels.append(label_ids)
        probabilities.append(F.softmax(torch.from_numpy(logits), dim=1).cpu().numpy())




    #Results

    predictions = [item for sublist in predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]


    predicted_labels = np.argmax(predictions, axis=1).flatten()
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
                preds=predicted_labels,
                y_true=true_labels,
                class_names=label_encoder.classes_
            )})

    normalized_probabilities = np.vstack(probabilities)
    unique_classes = np.unique(true_labels)
    num_unique_classes = len(unique_classes)

    roc_auc = roc_auc_score(true_labels, normalized_probabilities[:, 1])

    print("Test Accuracy: {0:.2f}".format(accuracy_score(predicted_labels, true_labels)))
    print("Test F1 Score: {0:.2f}".format(f1_score(predicted_labels, true_labels, average='weighted')))
    print(f"Test ROC AUC Score: {roc_auc:.2f}")
wandb.agent(sweep_id, function=train)
wandb.finish()