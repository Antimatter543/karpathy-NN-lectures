import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T) || Scaled attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) || Makes this a decoder block (cannot talk to future)
        wei = F.softmax(wei, dim=-1) # (B, T, T) || Convert to probabilities
        
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concat across the channels (last dim)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """ literally a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), # projection layer going back into the residual pathwayz
        )

    def forward(self, x): # This is on a per token level btw.
        return self.net(x)
# super simple bigram model


class Block(nn.Module):
    """ Transformer block: communication followed by computation 
    > This is the big block (like the right side dark outlined bit) on the legendary transformer diagram.
    """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size) # communication (tokens talk to each other, figure out their impact on each other)
        self.ffwd = FeedForward(n_embd) #  computation part

        # Layernorms
        self.ln1 = nn.LayerNorm(n_embd) # n_embd =32 remember, so this is basically acting across the channel of each (B,T). I.e token wise.
        self.ln2 = nn.LayerNorm(n_embd)
        # We deviate from OG paper (as is done in modern times) by putting this layer before the self-attention/self forward parts (hehe I had a feeling this would be better).

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # apply one head of self-attention || Multiple heads now (B,T,C) || Masked multi-head attention. No multi-head attention (i.e block above it in the diagram) since that connects to encoder (cross attention to encoder -- we haven't used that so yea) || Added residual connection (like a highway for the gradient to flow straight back to inputs!!!)

        # Feed forward -- lets it 'think' about the data that the tokens have given it. Before, they looked at each other but didn't have a lot of time to process wtf they each are. Kinda??

        x == self.ffwd(self.ln2(x)) ## || So The self attention is the communication between tokens and gathers their data (and how they impact each other). This layer  processes each token by itself || (B,T,C) || Added residual/skip connections
        
        return x
class BigramLanguageModel(nn.Module): # well, it's the GPT model now lol

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Number of dimensions we want for each vocab. Legit just a lookup table.
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # || We want to embed the positions too
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size) # Final linear layer that decodes into the vocab_size
        # self.count = 0 
		

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers. B is batch size, T is block size (context length) -- how long each training example should be; how many tokens are considered?
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C). Embeds basically every number 0 - T-1 into the table to make (T,C).
        x = tok_emb + pos_emb # (B,T,C) + (T,C) -> right aligns TC to (1,T,C), then broadcasts across the batch dimension (because well, over the batch the position embeds would be the same). -> (B,T,C)|| ON THE DIAGRAM, THIS IS POS ENCODING + OUTPUT EMBEDDING ADD

        x = self.blocks(x) # (B,T,C)
  

        logits = self.lm_head(x) # (B,T, vocab_size)  || This is the final linear (outside the big block)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # C is vocab_size 
            # self.count+=1
            # print(f'{C=}, {self.count}')
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # crop idx to last #block_size tokens (like 'scrolling') bc we have positional embeddings now (as our positional emb. table only has embeds for up to block size)
            idx_cond = idx[:, -block_size:]


            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    # print(xb.shape, "HEYYYYYYY")
    
    # evaluate the loss and gradient descend
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))