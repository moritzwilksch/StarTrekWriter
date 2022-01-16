# Star Trek Writer 🖖

> I forced a Neural Network to "watch" the entire Star Trek: The Next Generation series to generate new scripts

## Corpus
[A dump of all Star Trek scripts on Kaggle](https://www.kaggle.com/gjbroughton/start-trek-scripts)

## The Model
- tokenized corpus with BERT tokenizer and embedded tokens into 128-dimensional space
- used GRU for next-token prediction

**Hyperparameter Configuration**
```yaml
# learning
batch_size: 128
learning_rate: 0.01
epochs: 20
use_scheduler: True

# architecture
embedding_dim: 128
recurrent_size: 256
hidden_size: 512
```

## Example Script
<p align="center" style="font-family:monospace;">
PICARD<br>
Mister LaForge, can you get a frequency to engineering? <br>
<br>
LAFORGE<br>
I'm on my way.<br>
<br>
RIKER<br>
Geordi, I've finished the injury to the holodeck. <br>
<br>
CRUSHER<br>
I'm sorry, sir. we're going to have to succeed. <br>
<br>
PICARD<br>
Then it's a waste of time. I'd like to shut down the mission. I want you to know that I'm right. <br>
<br>
CRUSHER<br>
You look wonderful. I’m really not worried about this mission.<br>
<br>
RIKER<br>
I’ve been very close to caution. what's that? <br>
<br>
(but it's a table)<br>
</p>
