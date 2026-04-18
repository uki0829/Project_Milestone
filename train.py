import os
import torch
import torch.nn as nn
from torch.nn import functional as F
'''this is a decoder only model'''
block_size = 256
batch_size = 32
eval_iters = 200
max_iters = 5000
n_embed = 128
dropout = 0.2
learning_rate = 3e-4
num_heads = 8 # n_embed (128) must be divisible by num_heads
n_layer = 4
torch.manual_seed(1001)
#----------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
#----------------
text = ""
for fname in os.listdir("data"):
    if fname.endswith(".txt"):
        with open(os.path.join("data", fname), "r") as f:
            text += f.read()
print(len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# convert data into tensor
data = torch.tensor(encode(text), dtype=torch.long) # torch.long is int64
n = int(0.9*len(data))
train_data = data[:n] #90 percent of the dataset
test_data = data[n:] #10 percent of the dataset

# creating function to generate batches
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # this create a (4,)random starting point, torch.randint(max, dims)
    x = torch.stack([data[i:i+block_size] for i in ix]) #x is the context
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #y is the target in relation with the context
    x, y = x.to(device), y.to(device)
    return x, y

#creating the estimate loss function
@torch.no_grad() #this is the decorator which don't track the gradients
def estimate_loss():
    out = {} #create a empty dict to store our loss
    model.eval() #eval doesen't activate dropout 
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)#create a fix tensor to hold losses
        for k in range(eval_iters):#k bacth is extract from the data
            X,Y = get_batch(split)#we get our context and target from both train and test
            logits, loss = model(X,Y)#compute the loss and attention score
            losses[k] = loss.item()#replace the losses tensor witht the actuall loss in python float
        out[split] = losses.mean() #this will average the loss on chosen split
    model.train()# going back to train activate dropout
    return out

#define the single head attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_dim = head_size #this clearify the headsize later
        self.key = nn.Linear(n_embed, head_size, bias=False) #each linear transformation takes 384 input and generate 64 output, internally form a weight (64, 384) then transpose internally 
        self.query = nn.Linear(n_embed, head_size, bias=False) # Linear layer flatten the x to (BxT, C) mathch the n_embed, then internally adjust the parameter out put B, T, headsize weight matrix 
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #we are dot product the input x with the weight.T
        q = self.query(x) #same as key
        wei = q @ k.transpose(-2, -1) #this makes the dimension (B, 64, T) and get a (B, T, T)
        wei = wei * (self.head_dim ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #this will mask the TxT matrix with inf so after softmax the probability is 0
        wei = F.softmax(wei, dim=-1) #this calculate the actual attention weight
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
       
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #creating list of Head in the given num_heads each head has head_size by n_embed / num_heads
        self.proj = nn.Linear(n_embed, n_embed) #apply linear transformation to mix the heads output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #the heads are concat together along the last dimension (B, T, [head_size]), head_size*num_heads
        out = self.dropout(self.proj(out)) #we apply the dropout layer and map the output using the linear layer to something reasonable
        return out

class FeedForward(nn.Module):
    '''we are creating a feedforward network 
    allowing the tokens to learn richer features internally'''
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embed, 4*n_embed),
                # this will expand the dimension letting the tokens learns more about the features
                nn.ReLU(),
                nn.Linear(4*n_embed, n_embed),
                nn.Dropout(dropout)
                 )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module): # This is a container for of the modules
    def __init__(self, n_embed, num_heads): 
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffw = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed) # LayerNorm will stablize the training to avoid gradient explode and vanishing off features.
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        '''residual connection block'''
        x = x + self.sa(self.ln1(x)) # With the residual connections even in the model didn't learn useful feature we can still the original input x 
        x = x + self.ffw(self.ln2(x))#here we use pre-norm 
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads) for _ in range(n_layer)])
        # we are using * to upack the list[blocks] in to individual layer
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # Final linear transformation get produce the logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embed = self.token_embed(idx) # give the input x a token embedding with n_embe dim
        position_embed = self.position_embed(torch.arange(T, device=device)) # create the position table with B,T
        x = token_embed + position_embed # this will adds up the two shape(B,T,C) + (B,T) with pytorch broadcasting
        x = self.blocks(x) # passing through the blocks
        x = self.ln_f(x) # final layernorm
        logits = self.lm_head(x) # apply the linear layer to get the logits prediction raw scores

        if targets == None:
            loss = None # if targets is none we don't perform loss calculation
        else:
            B, T, C = logits.shape # first we give the variable it's value
            logits = logits.view(B*T, C) 
            # we use view to flatten the tensor by changing its shape but not change the data
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) 
            # the cross entropy loss will take N, C shape logits and C shape targets to calculate the correctness 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens): # Iterate over the max_new_tokens
            idx_count = idx[:, -block_size:] # this create a context window of block_size
            logits, loss = self(idx_count) # this passing the input into the forward function
            logits = logits[:, -1, :] # We are selecting only the last logits score 
            prob = F.softmax(logits, dim=-1) # passing that last logits of the sequence and predict the probability distribution
            idx_next = torch.multinomial(prob, num_samples=1) # randomly select a sample for the probability distribution append to the idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = Transformer() # instantiate the model 
m = model.to(device) # move model to device
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # create optimizer

# Define the training loop
for epoch in range(max_iters):
    if epoch % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))

torch.save(m.state_dict(), "../models/model.pt")
print("model saved")





        
    