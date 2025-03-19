import argparse
import os
import pandas as pd
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F

from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


#####################
# YOU MUST WRITE YOUR STUDENT ID IN THE VARIABLE STUDENT_ID
# EXAMPLE: STUDENT_ID = "12345678"
#####################
STUDENT_ID = "20251053"

class TextDatasetforTrainableEmbedding(Dataset):
    def __init__(self, data_path, pad_token = "PAD", max_length = 20, train=True):
        self.inputs = []
        self.labels = []
        self.vocab = {}
        self.vocab_label = []
        self.pad_token = pad_token

        #### 데이터 열어서 input, label에 추가
        with open(data_path, "r", encoding="utf-8") as f :
            full_text = f.readlines()

        for e in tqdm(full_text, desc= "Preparing Dataset...") :
            
            # 분리
            e_split = e.strip().split(",")
            
            # 맨뒤 공백 제거
            while e_split[-1] == '':
                e_split = e_split[:-1]

            # train인 경우 맨 뒤는 레이블
            if train :
                inp = e_split[:-1]
                lab = e_split[-1]
            # infer인 경우 전부 다 input
            else :
                inp = e_split
                lab = ""

            # max len에 맞게 padding
            if len(inp) < max_length :
                inp.extend([self.pad_token]*(max_length-len(inp)))
            
            assert len(inp) == max_length

            # 데이터로 추가
            self.inputs.append(inp)
            self.labels.append(lab)
        

        ### one-hot으로 vocab 구성하기
        # 모든 input을 합쳐서
        all_inputs = []
        for inp in self.inputs :
            all_inputs.extend(inp)
        
        # vocab 만들기
        all_tokens = sorted(list(set(all_inputs)))
        for i, token in enumerate(all_tokens) :
            self.vocab[token] = i
        
        ### Label vocab
        self.vocab_label = sorted(list(set(self.labels)))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx) :
        return torch.tensor([self.vocab[token] for token in self.inputs[idx]]), torch.tensor(self.vocab_label.index(self.labels[idx])).unsqueeze(0)

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float32))
        nn.init.kaiming_normal_(self.weight.data)

        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Perform linear transformation: y = xW^T + b
        output = x @ self.weight.T  # Equivalent to torch.mm(x, self.weight.T)
        output += self.bias
        return output

class CustomEmbeddings(nn.Module) :
    def __init__(self, vocab_size, embedding_dim) :
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim, dtype=torch.float32))
        nn.init.normal_(self.embeddings)

    def forward(self, x) :
        return self.embeddings[x]

class CustomModelwithTrainableEmbedding(nn.Module) :
    def __init__(self, vocab_size, embedding_dim, context_length=20, num_class=19, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        self.token_embedding_table = CustomEmbeddings(vocab_size, embedding_dim)
        
        self.layer1 = nn.Sequential(
            CustomLinear(context_length*embedding_dim, 1000),
            nn.ReLU(),
            nn.LayerNorm(1000),
            nn.Dropout(self.dropout),
        )
        
        self.layer2 = nn.Sequential(
            CustomLinear(1000, 100),
            nn.ReLU(),
            nn.LayerNorm(100),
            nn.Dropout(self.dropout),
        )

        self.layer3 = nn.Sequential(
            CustomLinear(100, num_class),
            nn.ReLU(), # ReLU 추가해야되나?
        )
    
    def forward(self, idx, target=None):

        # (BATCH, MAX_LENGTH) -> # (BATCH, MAX_LENGTH, EMBED_SIZE)
        tok_emb = self.token_embedding_table(idx)
        B,T,C = tok_emb.shape

        # (BATCH, MAX_LENGTH * EMBED_SIZE) -> (BATCH, 1, 1000)
        x = self.layer1(tok_emb.view(B, 1, T*C))
        
        # (BATCH, 1, 1000) -> (BATCH, 1, 100)
        x = self.layer2(x)
        
        # (BATCH, 1, 1000) -> (BATCH, 1, NUM_CLASS)
        logits = self.layer3(x)

        # target이 있을 경우에만 loss 계산
        if target is None :
            loss = None
        else :
            B,T,C = logits.shape # (BATCH, 1, NUM_CLASS)
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            
            loss = F.cross_entropy(logits, target)

        return logits, loss

def load_data(path):
    return path

def save_data(preds):
    # EXAMPLE
    # Save the data to a csv file
    # You can change function
    # BUT you should keep the file name as "{STUDENT_ID}_simple_seq.p#.answer.csv"

    tot_len = 100
    id = ["S{}".format(str(i+1).zfill(3)) for i in range(tot_len)]

    df_sub = pd.DataFrame(columns=['id', 'pred'])
    df_sub["id"] = id
    df_sub["pred"] = preds

    # df_sub.to_csv(f'{STUDENT_ID}_simple_seq.p1.answer.csv', index=False)
    df_sub.to_csv(f'{STUDENT_ID}_simple_seq.p2.answer.csv', index=False)

def save_checkpoint(model, optimizer, epoch, loss, filename='ckpt.pth') :
    checkpoint = {  
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),  
        'loss': loss,
    }  
    torch.save(checkpoint, filename)  
    print(f"Checkpoint saved to {filename}")  

def load_checkpoint(filename, model, optimizer=None):  
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])  
    
    if optimizer is not None :
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print("Model {} Loaded! Loss {} on Epoch {}".format(filename, loss, epoch))
    return epoch, loss

def parse_arguments() :
    parser = argparse.ArgumentParser(description='Argparse')
    
    parser.add_argument('--seed', type=int, default=486)
    
    parser.add_argument('--train_data_path', type=str, default="./dataset/simple_seq.train.csv")
    parser.add_argument('--test_data_path', type=str, default="./dataset/simple_seq.test.csv")

    parser.add_argument('--mode', type=str, default="infer")
    
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--embedding_dim', type=int, default=512)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=1000)

    parser.add_argument('--out_folder', type=str, default="./output_emb512_all")
    parser.add_argument('--model_path', type=str, default="./output_emb512_all/ckpt_1000.pth")
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument('--save_step', type=int, default=100)
    
    args = parser.parse_args()

    return args

def main():

    args = parse_arguments()

    print("===========================================")
    print("\tNow Training model...")
    print("===========================================")

    ### =======================================================================
    ### Device Setting 
    ### =======================================================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    ### =======================================================================
    ### Data 
    ### =======================================================================    
    train_data = TextDatasetforTrainableEmbedding(load_data(args.train_data_path))
    # train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size)

    train_dataset, test_dataset = train_test_split(train_data, test_size=0.1, random_state=args.seed)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    ### =======================================================================
    ### Model
    ### =======================================================================
    VOCAB_SIZE = len(train_data.vocab)
    print("Vocab Size : {}".format(VOCAB_SIZE))
    model = CustomModelwithTrainableEmbedding(VOCAB_SIZE, args.embedding_dim).to(DEVICE)

    # x,y = train_data[0]
    # model.eval()
    # logits, loss = model(x.unsqueeze(0).to(DEVICE), y.to(DEVICE))
    # print(logits)
    # print(loss)

    ## =======================================================================
    ## Training
    ## =======================================================================
    if args.mode == "train" :

        if not os.path.exists(args.out_folder):
            os.makedirs(args.out_folder)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        for steps in tqdm(range(1,args.epoch+1), desc= "TRAINING STEP") :
                
            for xb, yb in train_loader :
                
                # x : (batch, MAX_LENGTH)
                # y : (batch, 1)
            
                _, loss = model(xb.to(DEVICE), yb.to(DEVICE))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
            # Logging % Save
            if steps % args.eval_step == 0 :
                
                model.eval()
                total_eval_loss = 0
                for xb_eval, yb_eval in test_loader :
                    _, loss_eval = model(xb_eval.to(DEVICE), yb_eval.to(DEVICE))
                    total_eval_loss += loss_eval.item()
                model.train()
                print("Loss on {} / {} step ----- \tTrain : {}\tEval : {}".format(steps, args.epoch, loss.item(), total_eval_loss))
                
                # print("Loss on {} / {} step ----- \t{}".format(steps, args.epoch, loss.item()))

            if steps % args.save_step == 0 :
                save_checkpoint(model, optimizer, steps, loss.item(), "{}/ckpt_{}.pth".format(args.out_folder, steps))
                print("Model Saved!")

        print("Final Loss : {}".format(loss.item()))


    ## =======================================================================
    ## Infer
    ## =======================================================================
    else :

        _, _ = load_checkpoint(args.model_path, model)
        model.eval()

        # 이걸로 구축된 inputs만 사용해서, train_data의 vocab으로 토큰화.
        test_data = TextDatasetforTrainableEmbedding(args.test_data_path, train=False)
        preds = []

        for x in tqdm(test_data.inputs, desc= "Inferring...") :
            
            # train_dataset의 vocab으로 토큰화한다.
            # 이때 unk일 경우 pad를 넣어줌

            input_x = []
            for token in x :
                try :
                    input_x.append(train_data.vocab[token])
                except :
                    input_x.append(train_data.vocab["PAD"])
            
            input_x = torch.tensor(input_x).unsqueeze(0).to(DEVICE)
            logits, _ = model(input_x)
            preds.append(train_data.vocab_label[torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()])

        save_data(preds)

        print("Done Saving!")

if __name__ == "__main__":
    main()
