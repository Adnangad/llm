import re

# Retreives the data textand stores it into a var
with open("wizard_of_oz.txt", "r", encoding='utf-8') as f:
    raw_text = f.read()

print("Length before:: ", len(raw_text))

# Remove the whitespaces from the data
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Retreive all the unique words and sort them
all_words = sorted(list(set(preprocessed)))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)


vocab = {token:integer for integer, token in enumerate(all_words)}


# Tokenizes the data
class SimpleTokenizerV1:
    def __init__(self, vocab):
        # vocab is a key val pair of token and int/id
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # Handles text not in the vocab
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
    
        # this returns the id of the input text thats in vocab
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
tokenizer = SimpleTokenizerV1(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
encoded = tokenizer.encode(text)

print(encoded)

print("Decoded is:: ", tokenizer.decode(encoded))