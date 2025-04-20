import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Encoder using ResNet50
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]  # Remove the last fc layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

# Decoder with LSTM
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

# Simple vocabulary class
class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.idx = 4

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)

# Dummy data and test image preprocessing
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image).unsqueeze(0)
    return image

# Inference function (greedy decoding)
def caption_image(encoder, decoder, image_tensor, vocab, max_len=20):
    with torch.no_grad():
        features = encoder(image_tensor)
        inputs = features.unsqueeze(1)
        states = None
        output_caption = []

        input_token = torch.tensor([[vocab('<start>')]])
        for _ in range(max_len):
            embeddings = decoder.embed(input_token)
            inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
            hiddens, states = decoder.lstm(embeddings, states)
            output = decoder.linear(hiddens.squeeze(1))
            predicted = output.argmax(1)
            word = vocab.idx2word[predicted.item()]
            if word == '<end>':
                break
            output_caption.append(word)
            input_token = predicted.unsqueeze(0)

        return ' '.join(output_caption)

# Example usage (you'll need to train the model first)
if __name__ == '__main__':
    # Initialize components
    embed_size = 256
    hidden_size = 512
    vocab = Vocabulary()
    # Example vocab (should be built from dataset in real use)
    for word in ['a', 'man', 'with', 'red', 'shirt', '<start>', '<end>']:
        vocab.add_word(word)

    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dummy image path
    image_path = 'example.jpg'  # Replace with an actual image path
    image = load_image(image_path, transform)

    # Generate caption
    caption = caption_image(encoder, decoder, image, vocab)
    print("Generated Caption:", caption)
