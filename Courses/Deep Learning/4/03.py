# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://leakyrelu.com/2019/10/18/using-glove-word-embeddings-with-seq2seq-encoder-decoder-in-pytorch/

import string
import re 
import random

import torch 
import torch.nn as nn 
from torch import optim 
import torch.nn.functional as F 
from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cpu")

SOS_token = 0
EOS_token = 1

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1:"EOS"}
		self.n_words = 2

	def addsentence(self, sentence):
		for word in sentence.split(' '):
			self.addword(word)

	def addword(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words +=1
		else:
			self.word2count[word] +=1

def prepareData(lang1, lang2,ls1,ls2):
	inp_lang = Lang(lang1)
	out_lang = Lang(lang2)
	pairs = []
	for i in range(len(ls1)):
		inp_lang.addsentence(ls1[i])
		out_lang.addsentence(ls2[i])
		pairs.append([ls1[i],ls2[i]])
	print("Counting words...")
	print("Counted words:")
	print(inp_lang.name, inp_lang.n_words)
	print(out_lang.name, out_lang.n_words)
	return inp_lang, out_lang, pairs


en_path = 'Grp_35/train_en.txt'
hi_path = 'Grp_35/train_hi.txt'

valid_en = 'Grp_35/train_en.txt'
valid_hi = 'Grp_35/train_hi.txt'

max_len_en = 0
max_len_hi = 0
with open(en_path,'r') as file_handle:
	en_lines = []
	for line in file_handle:
		max_len_en = max(max_len_en,len(line))
		ln = line.strip()     #.split(' ')
		en_lines.append(ln)

with open(hi_path,'r') as file_handle:
	hi_lines = []
	for line in file_handle:
		max_len_hi = max(max_len_hi,len(line))
		ln = line.strip()       #.split(' ')
		hi_lines.append(ln)

with open(valid_en,'r') as file_handle:
	valid_en = []
	for line in file_handle:
		max_len_en = max(max_len_en,len(line))
		ln = line.strip()     #.split(' ')
		valid_en.append(ln)

with open(valid_hi,'r') as file_handle:
	valid_hi = []
	for line in file_handle:
		max_len_hi = max(max_len_hi,len(line))
		ln = line.strip()       #.split(' ')
		valid_hi.append(ln)

input_lang, output_lang, pairs = prepareData('eng', 'hin',en_lines,hi_lines)
valid_eng, valid_hin, valid_pairs = prepareData('eng', 'hin',valid_en,valid_hi)

print("max length is",max_len_en,max_len_hi)

class EncoderRNN(nn.Module):
	def __init__(self, inp_size, hid_size):
		super(EncoderRNN, self).__init__()
		self.hid_size = hid_size 
		self.embedding = nn.Embedding(inp_size, hid_size)
		self.lstm = nn.LSTM(hid_size, hid_size)

	def forward(self, inp, hidden):
		# print("IN ENCODER inp",inp.shape)
		embedded = self.embedding(inp)
		# print("Embedded shape",embedded.shape)
		out = embedded.view(1,1,-1)
		# print("After view",out.shape)
		out, hidden = self.lstm(out, hidden)
		# print("out, hidden",out.shape)
		return out, hidden

	def initHidden(self):
		ho, co = (torch.zeros(1,1,self.hid_size, device=device),torch.zeros(1,1,self.hid_size, device=device))
		return (ho,co)

class DecoderRNN(nn.Module):
	def __init__(self, hid_size, out_size):
		super(DecoderRNN, self).__init__()
		self.hid_size = hid_size
		self.embedding = nn.Embedding(out_size, hid_size)
		self.relu = nn.ReLU()
		self.lstm = nn.LSTM(hid_size, hid_size)
		self.out = nn.Linear(hid_size, out_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, enc_out, hidden):
		out = self.embedding(enc_out)
		out = out.view(1,1,-1)
		out = self.relu(out)
		out, hidden = self.lstm(out, hidden)
		out = self.softmax(self.out(out[0]))
		return out, hidden

	def initHidden(self):
		return torch.zeros(1,1,self.hid_size, device=device)


MAX_LENGTH=300
class AttnDecoderRNN(nn.Module):
	def __init__(self, hid_size, out_size, dropout_p=0.1, max_length=MAX_LENGTH):
		super(AttnDecoderRNN, self).__init__()
		self.hid_size = hid_size
		self.dropout_p = dropout_p
		self.max_len = max_length
		self.embedding = nn.Embedding(out_size, hid_size)
		self.attn = nn.Linear(hid_size*2,max_length)
		self.attn_combine = nn.Linear(hid_size*2, hid_size)
		self.dropout = nn.Dropout(dropout_p)
		self.relu = nn.ReLU()
		self.lstm = nn.LSTM(hid_size, hid_size)
		self.out = nn.Linear(hid_size, out_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, inp, hidden, enc_out):
		embedded = self.embedding(inp).view(1,1,-1)
		embedded = self.dropout(embedded)

		attn_wts = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_wts.unsqueeze(0), enc_out.unsqueeze(0)) 
		
		out = torch.cat((embedded[0], attn_applied[0]), 1)
		out = self.attn_combine(out).unsqueeze(0)
		out = self.relu(out)
		out, hidden = self.lstm(out, hidden)
		out = self.softmax(self.out(out[0]))
		return out, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1,1,self.hid_size, device=device)

def indexesFromSentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
	indexes = indexesFromSentence(lang, sentence)
	indexes.append(EOS_token)
	return torch.tensor(indexes, dtype=torch.long, device="cpu").view(-1, 1)


def tensorsFromPair(pair):
	input_tensor = tensorFromSentence(input_lang, pair[0])
	target_tensor = tensorFromSentence(output_lang, pair[1])
	return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	# print("here",input_tensor,input_tensor.shape)
	# print("here",target_tensor,target_tensor.shape)

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(max_length, encoder.hid_size, device=device)

	loss = 0

	for i in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
		encoder_outputs[i] = encoder_output[0, 0]
	decoder_input = torch.tensor([[SOS_token]], device=device)
	decoder_hidden = encoder_hidden

	for di in range(target_length):
		# decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
		decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
		topv, topi = decoder_output.topk(1)
		decoder_input = topi.squeeze().detach() 
		loss += criterion(decoder_output, target_tensor[di])
		if decoder_input.item() == EOS_token:
			break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math
import time

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
	plt.figure()
	fig, ax = plt.subplots()
	loc = ticker.MultipleLocator(base=0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)
	plt.savefig('plots/Loss-{}.png'.format(1))
	plt.show()

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
	start = time.time()
	plot_losses = []
	print_loss_total = 0  
	plot_loss_total = 0  
	
	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
	training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
	criterion = nn.NLLLoss()

	for iter in range(1, n_iters + 1):
		training_pair = training_pairs[iter - 1]
		input_tensor = training_pair[0]
		target_tensor = training_pair[1]

		loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
		print_loss_total += loss
		plot_loss_total += loss

		if iter % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print("iter",iter)
			print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

		if iter % plot_every == 0:
			evaluateRandomly(encoder_, decoder_)
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0

	showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hid_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # decoder_output, decoder_hidden, decoder_attention = decoder(
                # decoder_input, decoder_hidden, encoder_outputs)
            # decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        # return decoded_words, decoder_attentions[:di + 1]
        return decoded_words

def evaluateRandomly(encoder, decoder, n=10):
	score = 0.
	score1,score2,score3,score4 = 0.0,0.0,0.0,0.0
	for i in range(n):
		pair = random.choice(pairs)
		# print('>', pair[0])
		# print('=', pair[1])
		output_words = evaluate(encoder, decoder, pair[0])
		# output_words, attentions = evaluate(encoder, decoder, pair[0])
		output_sentence = ' '.join(output_words)
		# print('<', output_sentence)
		# print('')

		score += sentence_bleu([output_sentence.strip().split()], pair[1].strip().split(),weights = (0.25, 0.25, 0.25, 0.25))
		score1 += sentence_bleu([output_sentence.strip().split()], pair[1].strip().split(),weights = (1.,))
		score2 += sentence_bleu([output_sentence.strip().split()], pair[1].strip().split(),weights = (0.5, 0.5))
		score3 += sentence_bleu([output_sentence.strip().split()], pair[1].strip().split(),weights = (0.3333, 0.3333, 0.3333, 0.3333))
		score4 += sentence_bleu([output_sentence.strip().split()], pair[1].strip().split(),weights = (0.25, 0.25, 0.25, 0.25))

		print(score,score1,score2,score3,score4,output_sentence,pair[1])

	score /= n
	print("The bleu score is: "+str(score))

print("num_words",input_lang.n_words,output_lang.n_words)
print("length ",len(input_lang.word2count),len(output_lang.word2index))

hidden_size = 256
encoder_ = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder_ = DecoderRNN(hidden_size, output_lang.n_words).to(device)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
# trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
trainIters(encoder_, decoder_, 7000, print_every=1000)
# evaluateRandomly(encoder_, decoder_)

